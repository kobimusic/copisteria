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
from .vocab import (CLASSES, CLS_FAM, FAMILIES, N_ALTER, N_CLEF, N_CLS, N_DOTS, N_FAM, N_KEY, N_POS, N_TIME, N_TUP,
                    N_VOICE, POS_MAX, TIMES, TUPLETS, TYPE_QUARTERS)

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


N_FB = 39                                    # refinement feedback features per token (EvidenceNet.feedback)
# where an accidental glyph's centre sits above the head it belongs to, in half staff spaces: a flat's bowl is at the
# head's height and its stem reaches up (sharps and naturals are centred on the head)
ACC_RISE = {"flat": 0.92, "dflat": 0.92}
ACC_TYPES = ("sharp", "flat", "natural", "dsharp", "dflat")    # accidental glyph kinds the feedback names


def _class_tables():
    """Per class: nominal duration in quarters (notes / rests; 0 otherwise) and whether it is a note or rest;
    per tuplet: the factor it scales durations by."""
    dur, nr, acc, note = [], [], [], []
    for c in CLASSES:
        fam, _, sub = c.partition("_")
        nr.append(fam in ("note", "rest"))
        note.append(fam == "note")
        dur.append(float(TYPE_QUARTERS.get(sub, 0)) if fam in ("note", "rest") else 0.0)
        acc.append(ACC_TYPES.index(sub) + 1 if fam == "accid" and sub in ACC_TYPES else (len(ACC_TYPES) + 1 if fam == "accid" else 0))
    fac = []
    for t in TUPLETS:
        try:
            a, n = (int(v) for v in t.split("/"))
            fac.append(n / a)
        except ValueError:
            fac.append(1.0)
    return torch.tensor(dur), torch.tensor(nr), torch.tensor(fac), torch.tensor(acc), torch.tensor(note)


def _time_lengths():
    """Per meter of the bar head: the bar's length in quarters (0: other / not set here)."""
    out = []
    for t in TIMES:
        try:
            a, b = (int(v) for v in str(t).split("/"))
            out.append(a * 4.0 / b)
        except ValueError:
            out.append(0.0)
    return torch.tensor(out)


class EvidenceNet(nn.Module):
    def __init__(self, d=160, layers=6, heads=8, ff=640, drop=0.1, feat_dim=0, refine=0, fb=17):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff, drop=drop, feat_dim=feat_dim, refine=refine, fb=fb)
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
        self.refine = refine
        if refine:
            # a second reading of the window with the first one's conclusions as input: each note's onset and how
            # sure it was, its voice, and where the durations read before it in its voice put it (the running
            # sum), so the model can see and fix readings that do not add up. Zero-initialised: an identity at first.
            self.n_fb = fb                       # the first n of the N_FB feedback features (17: onsets and voices)
            self.fb = nn.Sequential(nn.Linear(fb, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.fb[-1].weight); nn.init.zeros_(self.fb[-1].bias)
            self.rblocks = nn.ModuleList(Block(d, heads, ff, drop) for _ in range(refine))
            for blk in self.rblocks:
                for lin in (blk.o, blk.ff[-1]):
                    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
            self.rnorm = nn.LayerNorm(d)
            dur, nr, fac, acc, note = _class_tables()
            self.register_buffer("cls_acc", acc, persistent=False)
            self.register_buffer("cls_note", note, persistent=False)
            self.register_buffer("cls_dur", dur, persistent=False)
            self.register_buffer("cls_nr", nr, persistent=False)
            self.register_buffer("tup_fac", fac, persistent=False)
            self.register_buffer("time_len", _time_lengths(), persistent=False)
            self.register_buffer("acc_rise", torch.tensor([0.0] + [ACC_RISE.get(k, 0.0) for k in ACC_TYPES] + [0.0]),
                                 persistent=False)
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

    def init_refine_from_base(self):
        """After loading a model without the refinement pass: its final norm becomes the second pass's too."""
        if self.refine:
            self.rnorm.load_state_dict(self.norm.state_dict())

    def _heads(self, h, pri):
        out, ev = {}, {}
        for k, head in self.heads.items():
            z = head(h).float()
            if k in pri:
                ev[k] = z
                z = z + self.prior_scale[k] * pri[k]
            out[k] = z
        out["_evidence"] = ev
        return out

    @torch.no_grad()
    def feedback(self, out, b):
        """The first reading's conclusions per token, as numbers (N_FB): onset (quarters / 16) and its
        confidence, the running sum of the durations read before it in its staff, bar and voice, how far that sum
        is from its onset, whether they agree, how many other notes of its voice it collides with, the voice's
        total in the bar, its own duration, P(real), its voice, and whether it is a note or rest."""
        pad = b["pad"].bool()
        real = torch.sigmoid(out["real"][..., 0].float())
        cls = out["cls"].argmax(-1)
        dots = out["dots"].argmax(-1).float()
        dur = self.cls_dur[cls] * (2 - 0.5 ** dots) * self.tup_fac[out["tup"].argmax(-1)]
        nr = self.cls_nr[cls] & ~pad
        voice = out["voice"].argmax(-1)
        chord = torch.sigmoid(out["chord"][..., 0].float()) > 0.5
        pb, pf = out["onset_beat"].float().softmax(-1), out["onset_frac"].float().softmax(-1)
        on = pb.argmax(-1).float() + pf.argmax(-1).float() / pf.shape[-1]
        conf = pb.max(-1).values * pf.max(-1).values
        root = nr & (real > 0.5) & ~chord
        same = (b["sys"][:, :, None] == b["sys"][:, None, :]) & (b["staffpos"][:, :, None] == b["staffpos"][:, None, :]) & \
               (b["col"][:, :, None] == b["col"][:, None, :]) & (voice[:, :, None] == voice[:, None, :])
        L = same.shape[-1]
        before = torch.ones(L, L, dtype=torch.bool, device=same.device).tril(-1)
        rootj = root[:, None, :]
        cum = ((same & before & rootj).float() @ dur[..., None])[..., 0]
        fill = ((same & rootj).float() @ dur[..., None])[..., 0]
        eye = torch.eye(L, dtype=torch.bool, device=same.device)
        coll = (same & rootj & ((on[:, :, None] - on[:, None, :]).abs() < 1 / 96) & ~eye).float().sum(-1)
        delta = cum - on
        # the nearest note or rest of another voice in the same staff and bar: notes struck together line up, so
        # its onset is evidence for this one's (its x distance in staff spaces says how much)
        bar = (b["sys"][:, :, None] == b["sys"][:, None, :]) & (b["staffpos"][:, :, None] == b["staffpos"][:, None, :]) & \
              (b["col"][:, :, None] == b["col"][:, None, :])
        other = bar & (voice[:, :, None] != voice[:, None, :]) & (nr & (real > 0.5))[:, None, :]
        dx = (b["geo"][:, :, None, 0] - b["geo"][:, None, :, 0]).abs().float() * 10       # dx_bar is x / sp / 10
        dx = dx.masked_fill(~other, 1e4)
        near = dx.argmin(-1)
        has_other = other.any(-1)
        near_on = on.gather(1, near)
        near_dx = dx.gather(2, near[..., None])[..., 0]
        aligned = (has_other & (near_dx < 0.5)).float()
        f = torch.stack([on / 16, conf, cum / 16, delta.clamp(-4, 4) / 4, (delta.abs() < 1 / 96).float(),
                         torch.log1p(coll), fill / 16, dur / 4, real, nr.float(),
                         aligned, aligned * (near_on - on).clamp(-4, 4) / 4,
                         has_other.float() * torch.exp(-near_dx.clamp(max=20))], -1)
        # accidentals: the nearest accidental glyph at this note's height to its left in the bar (its kind, weighted
        # by nearness), and the alteration read for the last note before it on the same line of the bar (what an
        # accidental carries)
        acc_kind = self.cls_acc[cls]                                               # 0: not an accidental
        isacc = (acc_kind > 0) & (real > 0.5) & ~pad
        y = b["geo"][..., 2].float() * 10                                         # staff position, half spaces
        x = b["geo"][..., 0].float() * 10                                         # x in the bar, staff spaces
        dxl = x[:, :, None] - x[:, None, :]                                       # this token's x minus the other's
        sameline = (y[:, :, None] - y[:, None, :]).abs() < 0.75
        cand = bar & sameline & isacc[:, None, :] & (dxl > 0) & (dxl < 5)
        dacc = dxl.masked_fill(~cand, 1e4)
        nearest = dacc.argmin(-1)
        has_acc = cand.any(-1)
        kind = acc_kind.gather(1, nearest).clamp(max=len(ACC_TYPES) + 1)
        w = has_acc.float() * torch.exp(-dacc.gather(2, nearest[..., None])[..., 0].clamp(max=20))
        acc_f = torch.nn.functional.one_hot(kind, len(ACC_TYPES) + 2)[..., 1:].float() * w[..., None]   # 6
        isnote = self.cls_note[cls] & ~pad
        prevm = bar & sameline & (isnote & (real > 0.5))[:, None, :] & (dxl > 0.3)
        dprev = dxl.masked_fill(~prevm, 1e4)
        prev = dprev.argmin(-1)
        has_prev = prevm.any(-1)
        alt = out["alter"].float().softmax(-1)                                     # 6: key, -2..2
        carry = alt.gather(1, prev[..., None].expand(-1, -1, alt.shape[-1])) * has_prev[..., None].float()
        f = torch.cat([f, torch.nn.functional.one_hot(voice, N_VOICE).float(), acc_f, carry], -1)
        if self.n_fb > f.shape[-1]:
            # the meter: the bar's length (the last meter a bar head read sets, at or before this token in reading
            # order, where bars come before their contents), and where the voice's durations put this note and the
            # voice against it -- a voice that overfills its bar is read wrong somewhere (an unmarked tuplet)
            ismeas = (b["fam"] == FAMILIES.index("measure")) & ~pad
            tlen = self.time_len[out["time"].argmax(-1)]
            setter = ismeas & (tlen > 0)
            idx = torch.arange(L, device=pad.device)[None, :].expand_as(setter)
            last = torch.where(setter, idx, torch.full_like(idx, -1)).cummax(-1).values
            blen = tlen.gather(1, last.clamp(min=0)) * (last >= 0).float()
            known = (blen > 0).float()
            over = ((fill - blen) / 4).clamp(-4, 4) * known
            ends = ((cum + dur - blen) / 4).clamp(-4, 4) * known
            # accidentals again, a flat at its bowl's height (the glyph's centre sits above the head it marks)
            ya = y - self.acc_rise[acc_kind]
            cand2 = bar & ((y[:, :, None] - ya[:, None, :]).abs() < 0.75) & isacc[:, None, :] & (dxl > 0) & (dxl < 5)
            dacc2 = dxl.masked_fill(~cand2, 1e4)
            nearest2 = dacc2.argmin(-1)
            kind2 = acc_kind.gather(1, nearest2).clamp(max=len(ACC_TYPES) + 1)
            w2 = cand2.any(-1).float() * torch.exp(-dacc2.gather(2, nearest2[..., None])[..., 0].clamp(max=20))
            acc_f2 = torch.nn.functional.one_hot(kind2, len(ACC_TYPES) + 2)[..., 1:].float() * w2[..., None]
            f = torch.cat([f, torch.stack([blen / 16, known, over, ends], -1), acc_f2], -1)
        return f * (~pad)[..., None].float()

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
        pri = self.priors(b)
        out = self._heads(self.norm(x), pri)
        if self.refine:
            first = out
            x = x + self.fb(self.feedback(first, b)[..., :self.n_fb].to(x.dtype))
            for blk in self.rblocks:
                x, att = blk(x, basis, pad, need_attn)
                if need_attn:
                    atts.append(att)
            out = self._heads(self.rnorm(x), pri)
            out["_pass1"] = first
        if need_attn:
            out["_attn"] = atts
        return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
