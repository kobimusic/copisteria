"""The evidence model: a small transformer over a window of tokens (a few consecutive systems).

Every head that has a detector reading behind it is a *residual on that reading's log-probability*:

    final logits = a_head * log P_detector(value) + evidence(value | context)

so ``evidence`` is literally what the context adds to (or takes from) the detector's local reading, in nats; with
no evidence the detector stands. Heads without a detector reading (sounding alteration, chord, tie, tuplet, and
the bar's clef / key / meter) are read from context alone.

Attention carries a learned bias per (same part?, bar distance) so locality is a prior the model can override,
not a wall.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import MAX_STAVES, N_GEO
from .vocab import (CLS_FAM, N_ALTER, N_CLEF, N_CLS, N_DOTS, N_FAM, N_KEY, N_POS, N_TIME, N_TUP, N_VOICE,
                    POS_MAX)

MAX_DCOL = 8
DCOL_BANDS = (0, 1, 2, 3, 4)                 # |bar distance| 0, 1, 2, 3, 4..MAX_DCOL
N_BASIS = 2 * len(DCOL_BANDS) + 3            # (same part?, band) + far + "later bar" + same system
EPS_LOGP = math.log(1e-4)

PRIOR_HEADS = ("real", "cls", "dots", "pos", "voice", "grace")
ONSET_BEATS, ONSET_FRAC = 16, 48          # onset in the bar: whole quarters (clipped) and 48ths of a quarter
FREE_HEADS = {"alter": N_ALTER, "chord": 1, "tie": 1, "tup": N_TUP, "key": N_KEY, "time": N_TIME, "clef": N_CLEF,
              "onset_beat": ONSET_BEATS, "onset_frac": ONSET_FRAC}


class Block(nn.Module):
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)
        self.rel = nn.Parameter(torch.zeros(heads, N_BASIS))           # attention bias per relation, per head

    def forward(self, x, basis, pad, need_attn=False):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        bias = torch.einsum("hk,bkij->bhij", self.rel.to(basis.dtype), basis)          # B, H, L, L
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))
        if need_attn:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h) + bias
            att = att.softmax(-1)
            y = att @ v
        else:
            att = None
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.drop(self.o(y.transpose(1, 2).reshape(B, L, D)))
        x = x + self.drop(self.ff(self.n2(x)))
        return x, att


class EvidenceNet(nn.Module):
    def __init__(self, d=160, layers=6, heads=8, ff=640, drop=0.1, feat_dim=0):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff, drop=drop, feat_dim=feat_dim)
        self.cls_emb = nn.Embedding(N_CLS, d)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.fam_emb = nn.Embedding(N_FAM, d)
        self.staffpos_emb = nn.Embedding(MAX_STAVES, d)
        self.nstaves_emb = nn.Embedding(MAX_STAVES + 1, d)
        self.pos_emb = nn.Embedding(N_POS + 1, d)               # the detector's staff position (+1: none)
        n_num = N_GEO + 3 + 4 + N_VOICE + 3 + 3 + 2
        self.num = nn.Sequential(nn.Linear(n_num, d), nn.GELU(), nn.Linear(d, d))
        self.feat_dim = feat_dim
        if feat_dim:
            # the detector's image features at the token (all zero: none); zero-initialised, so a model
            # fine-tuned from one without them starts out the same
            self.feat_norm = nn.LayerNorm(feat_dim)
            self.feat_proj = nn.Linear(feat_dim, d)
            nn.init.zeros_(self.feat_proj.weight); nn.init.zeros_(self.feat_proj.bias)
            self.cand_emb = nn.Parameter(torch.zeros(d))          # a sub-threshold candidate of the detector's
        self.blocks = nn.ModuleList(Block(d, heads, ff, drop) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        out = {"real": 1, "cls": N_CLS, "dots": N_DOTS, "pos": N_POS, "voice": N_VOICE, "grace": 3, **FREE_HEADS}
        self.heads = nn.ModuleDict({k: nn.Linear(d, n) for k, n in out.items()})
        self.prior_scale = nn.ParameterDict({k: nn.Parameter(torch.ones(1)) for k in PRIOR_HEADS})
        fm = torch.zeros(N_FAM, N_CLS, dtype=torch.bool)
        for c, f in enumerate(CLS_FAM):
            fm[f, c] = True
        self.register_buffer("fam_mask", fm, persistent=False)

    # ---------------------------------------------------------------------------------------------- priors
    def priors(self, b):
        fam, cls_ids, cls_p, p = b["fam"].long(), b["cls_ids"].long(), b["cls_p"], b["p"]
        B, L = fam.shape
        logp = torch.full((B, L, N_CLS), EPS_LOGP, device=p.device)
        logp.scatter_(2, cls_ids, torch.log(cls_p.clamp_min(1e-4)))
        fm = self.fam_mask[fam]                                       # B, L, N_CLS
        size = fm.sum(-1, keepdim=True).clamp_min(1)
        uni = torch.where(fm, -torch.log(size.float()), torch.full_like(logp, EPS_LOGP))
        m = b["cls_mask"].bool()[..., None]
        pri = {"cls": torch.where(m, uni, logp)}
        pri["real"] = torch.logit(p.clamp(1e-3, 1 - 1e-3))[..., None]
        pri["dots"] = torch.log(b["dots_p"] + 1e-3)
        pri["voice"] = torch.log(b["voice_p"] + 1e-3)
        pri["grace"] = torch.log(b["grace_p"] + 1e-3)
        pv, pp = b["pos_v"].long(), b["pos_p"]
        has = (pv > -99)
        idx = (pv.clamp(-POS_MAX, POS_MAX) + POS_MAX)
        soft = torch.zeros(B, L, N_POS, device=p.device)
        soft.scatter_(2, idx[..., None], pp[..., None])
        side = ((1 - pp) / 2)[..., None]
        soft.scatter_add_(2, (idx - 1).clamp(0, N_POS - 1)[..., None], side)
        soft.scatter_add_(2, (idx + 1).clamp(0, N_POS - 1)[..., None], side)
        pri["pos"] = torch.where(has[..., None], torch.log(soft + 1e-3), torch.zeros_like(soft))
        none = b["dots_p"].sum(-1, keepdim=True) == 0                 # no attribute reading at all
        for k in ("dots", "voice", "grace"):
            pri[k] = torch.where(none, torch.zeros_like(pri[k]), pri[k])
        return pri

    # ---------------------------------------------------------------------------------------------- embedding
    def embed(self, b):
        fam, cls_ids, cls_p = b["fam"].long(), b["cls_ids"].long(), b["cls_p"]
        e = (self.cls_emb(cls_ids) * cls_p[..., None]).sum(2)
        m = b["cls_mask"].float()[..., None]
        e = e * (1 - m) + self.mask_emb * m
        p = b["p"]
        pv = b["pos_v"].long()
        pos_idx = torch.where(pv > -99, pv.clamp(-POS_MAX, POS_MAX) + POS_MAX, torch.full_like(pv, N_POS))
        num = torch.cat([b["geo"], p[..., None], torch.logit(p.clamp(1e-3, 1 - 1e-3))[..., None] / 5,
                         b["pos_p"][..., None], b["dots_p"], b["voice_p"], b["grace_p"], b["stem_p"],
                         b["cls_mask"].float()[..., None], b["attr_mask"].float()[..., None]], -1)
        x = (e + self.fam_emb(fam) + self.staffpos_emb(b["staffpos"].long())
             + self.nstaves_emb(b["nstaves"].long()) + self.pos_emb(pos_idx) + self.num(num))
        if self.feat_dim and "feat" in b:
            f = b["feat"].float()
            has = (f.abs().sum(-1, keepdim=True) > 0).float()
            x = x + self.feat_proj(self.feat_norm(f)) * has + self.cand_emb * b["cand"].float()[..., None]
        return x

    @staticmethod
    def basis(b, dtype):
        """Fixed relation maps between every query and key token: [B, N_BASIS, L, L] of 0/1."""
        col, sp, sysi = b["col"].long(), b["staffpos"].long(), b["sys"].long()
        dcol = col[:, None, :] - col[:, :, None]
        ad = dcol.abs()
        same = sp[:, None, :] == sp[:, :, None]
        maps = []
        for part in (same, ~same):
            for i, lo in enumerate(DCOL_BANDS):
                hi = DCOL_BANDS[i + 1] - 1 if i + 1 < len(DCOL_BANDS) else MAX_DCOL
                maps.append(part & (ad >= lo) & (ad <= hi))
        maps += [ad > MAX_DCOL, dcol > 0, sysi[:, None, :] == sysi[:, :, None]]
        return torch.stack(maps, 1).to(dtype)

    def forward(self, b, need_attn=False):
        x = self.embed(b)
        dtype = torch.bfloat16 if x.is_cuda else torch.float32
        basis = self.basis(b, dtype)
        pad = b["pad"].bool()
        atts = []
        for blk in self.blocks:
            x, att = blk(x, basis, pad, need_attn)
            if need_attn:
                atts.append(att)
        h = self.norm(x)
        pri = self.priors(b)
        out, ev = {}, {}
        for k, head in self.heads.items():
            z = head(h).float()
            if k in pri:
                ev[k] = z
                z = z + self.prior_scale[k] * pri[k]
            out[k] = z
        if need_attn:
            out["_attn"] = atts
        out["_evidence"] = ev
        return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
