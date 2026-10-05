"""Round 2 of the 2M-parameter search: six designs that differ in KIND from round 1's centre-heatmap
family, not in width. Every model keeps the full capability set (266 classes + boxes, staff_position / stem_dir /
dots / grace / voice_slot on notes and rests) and plugs into ``copisteria.detector.small.train`` through the same three calls as
the round-1 models: ``forward(x) -> dict``, ``loss(out, t, attr_cls_mask)`` and ``decode_out(out, ...)``.

  peak-relational  two stages: a class-agnostic objectness map proposes centres; the candidates (RoI features +
                   position) then reason about each other in a 2-layer transformer before each is classified
                   (266 + background), its box refined and its attributes read. The precision gap on real pages
                   (rests / accidentals / articulations over-fired) is a relational question: a mark is an
                   accidental because a note follows it. Round 1 classified every cell alone.
  seg-vote         dense semantic segmentation at stride 4 (every cell of a glyph carries its class) + a centre
                   vote (offset to the owner's centre) + a size at every cell. Instances = clusters of votes
                   (Hough-style); the centre and size are the voters' average, so localisation is averaged over the
                   glyph's cells instead of read from one cell. Supervision density: every glyph cell, not one.
  cnet-recur       weight-shared recurrent refinement: the heads run twice, the second pass sees the first pass's
                   own heatmap / size / attribute maps next to the features. Compute is not capped, parameters are.
  cnet-distill     round-1's flat model with feature-hint distillation from the deployed v6 (26M): its stride-4
                   and stride-8 neck features are regressed by 1x1 adapters. Does a 2M trunk lack capacity or only
                   supervision?
  cnet-twostream   BiSeNet-style trunk: a shallow full-resolution detail path (stride 4, receptive field a few
                   staff spaces) and a deep, wide context path on the 4x-downsampled page (strides 8-64), fused
                   once with channel attention. Round 1's encoders were sequential: context could only come from
                   the same features that carried the detail.
  cnet-embed       the class heatmap as a dot product between a per-cell embedding and a FACTORISED class table
                   (family vector + subtype vector): the 200 rare classes share their family's direction, so the
                   tail of the 266-class taxonomy learns from its siblings' centres too.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR
from . import models as M
from .models import Basic, ConvBNAct, DW, DWDown, Encoder, FPN, attr_loss, focal, gather_at, head

BG = "background"


# ----------------------------------------------------------------------------------------------------------------
# shared bits

def _sincos_pos(cy, cx, d):
    """2-D sine / cosine position code for cell coordinates: [n, d] (d/4 frequencies per axis)."""
    q = d // 4
    freq = torch.exp(-math.log(2000.0) * torch.arange(q, device=cy.device).float() / q)     # wavelengths 1 .. 2000 cells
    ay = cy.float()[:, None] * freq[None]; ax = cx.float()[:, None] * freq[None]
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], 1)


def _box_l1(reg, t):
    m = t["mask"].unsqueeze(-1).float(); n = t["mask"].sum().clamp(min=1)
    wh = 0.1 * (F.l1_loss(reg[..., :2], t["wh"], reduction="none") * m).sum() / n
    off = (F.l1_loss(reg[..., 2:], t["off"], reduction="none") * m).sum() / n
    return wh, off


def _peaks(hm, K, thresh):
    """[C, H, W] sigmoid map -> flat indices of 3x3 maxima over ``thresh`` (top K)."""
    pooled = F.max_pool2d(hm[None], 3, 1, 1)[0]
    hm = hm * (pooled == hm)
    scores, idx = hm.reshape(-1).topk(min(K, hm.numel()))
    keep = scores > thresh
    return scores[keep], idx[keep]


def _dets(boxes, scores, cls, attrs, cy, cx, nms_iou):
    if nms_iou and len(boxes):
        from torchvision.ops import batched_nms
        k = batched_nms(boxes, scores, cls, nms_iou)
        boxes, scores, cls, attrs, cy, cx = boxes[k], scores[k], cls[k], attrs[k], cy[k], cx[k]
    return {"boxes": boxes, "scores": scores, "cls": cls, "attrs": attrs, "cell": torch.stack([cy, cx], 1)}


class Base(nn.Module):
    """What copisteria.detector.small.train / predict_real read off a model."""
    mode = "x"
    staff_head = False
    stride = 4
    n_fam = 0

    def __init__(self, n_cls):
        super().__init__()
        self.n_cls = n_cls


# ----------------------------------------------------------------------------------------------------------------
# 1. peak-relational

class RelationalDecoder(nn.Module):
    def __init__(self, d=96, layers=2, heads=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, tok, pad_mask):
        return self.enc(tok, src_key_padding_mask=pad_mask)


class PeakRelational(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, d=96, layers=2, roi=3,
                 n_neg=192, k_infer=2000):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.obj = head(fpn, hd, 1, dilations=(1, 2), prior=0.01)
        self.reg = head(fpn, hd, 4)
        self.roi, self.d, self.n_neg, self.k_infer = roi, d, n_neg, k_infer
        self.proj = nn.Linear(fpn * roi * roi + fpn, d)
        self.dec = RelationalDecoder(d, layers)
        self.cls = nn.Linear(d, n_cls + 1)                 # + background
        self.refine = nn.Linear(d, 4)                      # dx, dy (cells), dlogw, dlogh
        self.attr = nn.Linear(d, N_ATTR)
        nn.init.zeros_(self.refine.weight); nn.init.zeros_(self.refine.bias)
        nn.init.constant_(self.cls.bias, 0.0); self.cls.bias.data[n_cls] = 2.0     # background prior

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"f": f, "obj": self.obj(f), "reg": self.reg(f)}

    # candidates -----------------------------------------------------------------------------------------------
    def _tokens(self, f, reg, cy, cx):
        """RoI features (roi x roi grid over the dense box at each candidate) + the centre feature + position."""
        B, C, H, W = f.shape
        n = cy.shape[1]
        b = torch.arange(B, device=f.device)[:, None].expand(B, n)
        cell = cy * W + cx
        r = gather_at(reg, cell)                                                  # [B, n, 4]
        w = torch.exp(r[..., 0].clamp(max=6)) * self.stride; h = torch.exp(r[..., 1].clamp(max=6)) * self.stride
        px = (cx.float() + r[..., 2]) * self.stride; py = (cy.float() + r[..., 3]) * self.stride
        boxes = torch.stack([b.float().reshape(-1), (px - w / 2).reshape(-1), (py - h / 2).reshape(-1), (px + w / 2).reshape(-1), (py + h / 2).reshape(-1)], 1)
        from torchvision.ops import roi_align
        roi = roi_align(f.float(), boxes.detach(), (self.roi, self.roi), spatial_scale=1.0 / self.stride, sampling_ratio=2, aligned=True)
        roi = roi.reshape(B, n, -1)
        centre = gather_at(f, cell)
        tok = self.proj(torch.cat([roi, centre], -1)) + _sincos_pos(cy.reshape(-1), cx.reshape(-1), self.d).reshape(B, n, -1)
        return tok, r

    def _run(self, out, cy, cx, pad):
        tok, r = self._tokens(out["f"], out["reg"], cy, cx)
        pad = pad.clone(); pad[:, 0] = False                 # a fully padded row would attend over nothing (NaN)
        h = self.dec(tok, pad)
        return self.cls(h), self.refine(h), self.attr(h), r

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["obj"].shape
        losses = {}
        obj_gt = t["hm"].amax(1, keepdim=True)
        losses["obj"] = focal(out["obj"], obj_gt)
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        # candidates: every GT centre + the strongest non-GT peaks of the objectness map (hard negatives)
        with torch.no_grad():
            obj = torch.sigmoid(out["obj"].detach().float())[:, 0]
            pooled = F.max_pool2d(obj[:, None], 3, 1, 1)[:, 0]
            peaks = obj * (pooled == obj)
            gt_cells = torch.zeros(B, H * W, dtype=torch.bool, device=obj.device)
            gt_cells.scatter_(1, t["ind"], t["mask"])
            peaks = peaks.reshape(B, -1).masked_fill(gt_cells, 0.0)
            neg_s, neg_i = peaks.topk(self.n_neg, 1)                                   # [B, n_neg]
            neg_m = neg_s > 0.02
        ind = torch.cat([t["ind"], neg_i], 1); mask = torch.cat([t["mask"], neg_m], 1)
        cy, cx = ind // W, ind % W
        cls_logit, ref, attr, _ = self._run(out, cy, cx, ~mask)
        K = t["ind"].shape[1]
        # classification: GT candidates -> their class, negatives -> background
        target = torch.cat([t["cls"], torch.full_like(neg_i, self.n_cls)], 1)
        losses["cls"] = F.cross_entropy(cls_logit[mask].float(), target[mask])
        # refinement on GT candidates: the residual between the dense box and the truth
        m = t["mask"]
        if m.any():
            dense = gather_at(out["reg"], t["ind"]).detach()
            res = torch.cat([t["off"] - dense[..., 2:], t["wh"] - dense[..., :2]], -1)     # dx, dy, dlogw, dlogh
            losses["ref"] = F.l1_loss(ref[:, :K][m].float(), res[m])
        # attributes on GT candidates, through round 1's masks: the token logits laid out as a [B, N_ATTR, K, 1] "map"
        # whose flat cell index is the candidate index
        tt = {"ind": torch.arange(K, device=attr.device)[None].expand(B, K), "mask": t["mask"], "cls": t["cls"], "attrs": t["attrs"]}
        al, parts = attr_loss(attr[:, :K].permute(0, 2, 1)[..., None], tt, attr_cls_mask)
        losses["attr"] = al
        total = sum(losses.values())
        return total, {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        B, _, H, W = out["obj"].shape
        obj = torch.sigmoid(out["obj"].float())
        res = []
        for b in range(B):
            s, idx = _peaks(obj[b], self.k_infer, thresh * 0.5)
            if len(idx) == 0:
                res.append(_dets(torch.zeros(0, 4, device=obj.device), s, idx, torch.zeros(0, N_ATTR, device=obj.device), idx, idx, 0)); continue
            cy, cx = (idx // W)[None], (idx % W)[None]
            cls_logit, ref, attr, r = self._run({k: v[b:b + 1] for k, v in out.items()}, cy, cx, torch.zeros_like(cy, dtype=torch.bool))
            p = torch.softmax(cls_logit[0].float(), -1)
            pc, cls = p[:, :self.n_cls].max(-1)
            score = s * pc
            r = r[0].float() + ref[0].float()[:, [2, 3, 0, 1]]
            w = torch.exp(r[:, 0]) * self.stride; h = torch.exp(r[:, 1]) * self.stride
            x = (cx[0].float() + r[:, 2]) * self.stride; y = (cy[0].float() + r[:, 3]) * self.stride
            boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
            keep = score > thresh
            res.append(_dets(boxes[keep], score[keep], cls[keep], attr[0].float()[keep], cy[0][keep], cx[0][keep], nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 2. seg-vote

class SegVote(Base):
    WIN = 33            # cells: a glyph's cells within this window of its centre carry its class and vote

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.sem = head(fpn, hd, n_cls + 1)                # + background (index n_cls)
        self.geo = head(fpn, hd, 4)                        # vote dx, dy (cells to the owner's centre); log w, log h
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        nn.init.constant_(self.sem[-1].bias, 0.0); self.sem[-1].bias.data[n_cls] = 4.0

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"sem": self.sem(f), "geo": self.geo(f), "attr": self.attr(f)}

    def targets(self, t, H, W):
        """Per-cell owner from the object lists: the smallest box wins a cell. Returns sem [B, H, W] (n_cls = bg),
        vote [B, 2, H, W], wh [B, 2, H, W], fg [B, H, W] bool."""
        B, K = t["ind"].shape
        dev = t["ind"].device
        cy0, cx0 = t["ind"] // W, t["ind"] % W
        cx = cx0.float() + t["off"][..., 0]; cy = cy0.float() + t["off"][..., 1]
        w = torch.exp(t["wh"][..., 0]); h = torch.exp(t["wh"][..., 1])
        area = (w * h).masked_fill(~t["mask"], float("inf"))
        r = self.WIN // 2
        dy, dx = torch.meshgrid(torch.arange(-r, r + 1, device=dev), torch.arange(-r, r + 1, device=dev), indexing="ij")
        gy = cy0[:, :, None, None] + dy; gx = cx0[:, :, None, None] + dx                       # [B, K, win, win]
        inside = ((gx.float() + 0.5 - cx[..., None, None]).abs() <= w[..., None, None] / 2) & ((gy.float() + 0.5 - cy[..., None, None]).abs() <= h[..., None, None] / 2)
        inside |= (dx == 0) & (dy == 0)
        ok = inside & (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W) & t["mask"][..., None, None]
        bi = torch.arange(B, device=dev)[:, None, None, None].expand_as(gy)
        flat = (bi * H + gy) * W + gx
        prio = torch.full((B * H * W,), float("inf"), device=dev)
        a4 = area[..., None, None].expand_as(gy)
        prio.scatter_reduce_(0, flat[ok], a4[ok], reduce="amin")
        own = ok & (a4 == prio.reshape(B, H, W)[bi.clamp(0, B - 1), gy.clamp(0, H - 1), gx.clamp(0, W - 1)])
        sem = torch.full((B * H * W,), self.n_cls, dtype=torch.long, device=dev)
        sem[flat[own]] = t["cls"][..., None, None].expand_as(gy)[own]
        vote = torch.zeros(B * H * W, 2, device=dev); whm = torch.zeros(B * H * W, 2, device=dev)
        vote[flat[own], 0] = (cx[..., None, None] - gx.float() - 0.5).expand_as(gy.float())[own]
        vote[flat[own], 1] = (cy[..., None, None] - gy.float() - 0.5).expand_as(gy.float())[own]
        whm[flat[own], 0] = t["wh"][..., 0][..., None, None].expand_as(gy.float())[own]
        whm[flat[own], 1] = t["wh"][..., 1][..., None, None].expand_as(gy.float())[own]
        sem = sem.reshape(B, H, W)
        return sem, vote.reshape(B, H, W, 2).permute(0, 3, 1, 2), whm.reshape(B, H, W, 2).permute(0, 3, 1, 2), sem.ne(self.n_cls)

    @staticmethod
    def _sem_sample(logits, sem, fg, ratio=3):
        """One sample's CE: every glyph cell + the hardest ``ratio`` x as many background cells (OHEM)."""
        ce = F.cross_entropy(logits.float()[None], sem[None], reduction="none")[0]
        n_fg = int(fg.sum())
        fg_loss = ce[fg].sum()
        bg = ce.masked_fill(fg, 0.0).reshape(-1)
        k = min(bg.numel(), max(256, ratio * n_fg))
        bg_loss = bg.topk(k).values.sum()
        return (fg_loss + bg_loss) / max(1, n_fg + k)

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["sem"].shape
        sem, vote, whm, fg = self.targets(t, H, W)
        losses = {}
        total = out["sem"].new_zeros((), dtype=torch.float32)
        for b in range(B):
            total = total + checkpoint(self._sem_sample, out["sem"][b], sem[b], fg[b], use_reentrant=False)
        losses["sem"] = total / B
        m = fg[:, None].float(); n = fg.sum().clamp(min=1)
        geo = out["geo"].float()
        losses["vote"] = (F.l1_loss(geo[:, :2], vote, reduction="none") * m).sum() / n
        losses["wh"] = 0.1 * (F.l1_loss(geo[:, 2:], whm, reduction="none") * m).sum() / n
        al, parts = attr_loss(out["attr"], t, attr_cls_mask)
        losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7, fg_min=0.3):
        B, C1, H, W = out["sem"].shape
        res = []
        for b in range(B):
            p = torch.softmax(out["sem"][b].float(), 0)                                 # [C+1, H, W]
            fgp = 1.0 - p[self.n_cls]
            geo = out["geo"][b].float()
            voter = (fgp >= fg_min).reshape(-1)
            if voter.sum() == 0:
                z = torch.zeros(0, device=p.device)
                res.append(_dets(torch.zeros(0, 4, device=p.device), z, z.long(), torch.zeros(0, N_ATTR, device=p.device), z.long(), z.long(), 0)); continue
            ys, xs = torch.meshgrid(torch.arange(H, device=p.device), torch.arange(W, device=p.device), indexing="ij")
            vx = (xs.float() + 0.5 + geo[0]).reshape(-1)[voter]; vy = (ys.float() + 0.5 + geo[1]).reshape(-1)[voter]
            tx = vx.floor().long().clamp(0, W - 1); ty = vy.floor().long().clamp(0, H - 1)
            tgt = ty * W + tx
            wgt = fgp.reshape(-1)[voter]
            mass = torch.zeros(H * W, device=p.device).scatter_add_(0, tgt, wgt)
            # every cell that collected votes is an instance (no 3x3 max-pool: chord seconds vote for adjacent cells
            # and a max-pool kept only one of them -- note recall 0.67 at epoch 8); box NMS removes the split glyphs
            idx = (mass > 0.5).nonzero()[:, 0]
            if len(idx) > K:
                idx = idx[mass[idx].topk(K).indices]
            s = mass[idx]
            if len(idx) == 0:
                z = torch.zeros(0, device=p.device)
                res.append(_dets(torch.zeros(0, 4, device=p.device), z, z.long(), torch.zeros(0, N_ATTR, device=p.device), z.long(), z.long(), 0)); continue
            # aggregate per target cell: centre (weighted mean of votes), size (weighted mean of log wh), class (summed probs)
            def agg(v):
                return torch.zeros(H * W, device=p.device).scatter_add_(0, tgt, wgt * v)
            cxm = agg(vx) / mass.clamp(min=1e-6); cym = agg(vy) / mass.clamp(min=1e-6)
            lw = agg(geo[2].reshape(-1)[voter]) / mass.clamp(min=1e-6); lh = agg(geo[3].reshape(-1)[voter]) / mass.clamp(min=1e-6)
            pv = p[:self.n_cls].reshape(self.n_cls, -1)[:, voter] * wgt[None]                    # [C, nv]
            csum = torch.zeros(self.n_cls, H * W, device=p.device).scatter_add_(1, tgt[None].expand(self.n_cls, -1), pv)
            cls_p = csum[:, idx] / mass[idx].clamp(min=1e-6)                                     # [C, n]
            pc, cls = cls_p.max(0)
            w = torch.exp(lw[idx]) ; h = torch.exp(lh[idx])
            cover = (mass[idx] / (w * h).clamp(min=1.0, max=float(self.WIN ** 2))).clamp(max=1.0)
            score = cover * pc
            x = cxm[idx] * self.stride; y = cym[idx] * self.stride
            boxes = torch.stack([x - w * self.stride / 2, y - h * self.stride / 2, x + w * self.stride / 2, y + h * self.stride / 2], 1)
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, idx].t()
            keep = score > thresh
            res.append(_dets(boxes[keep], score[keep], cls[keep], attrs[keep], (idx // W)[keep], (idx % W)[keep], nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 3. cnet-recur

class RecurNet(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, passes=2):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.fb = 1 + 4 + 2                                 # max class prob, the 4 regressions, stem-dir up/down probs
        self.refine = nn.Sequential(ConvBNAct(fpn + self.fb, fpn, 3), ConvBNAct(fpn, fpn, 3, act=False))
        nn.init.zeros_(self.refine[-1][1].weight)           # the refiner starts as identity on the features
        self.passes = passes

    def _heads(self, f):
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    def _feedback(self, o):
        p = torch.sigmoid(o["hm"].float()).amax(1, keepdim=True)
        st = ATTR_OFFSETS["stem_dir"]
        sd = torch.softmax(o["attr"].float()[:, st:st + ATTR_SIZES["stem_dir"]], 1)[:, 1:3]
        return torch.cat([p, o["reg"].float().clamp(-6, 6), sd], 1).to(o["hm"].dtype)

    def forward(self, x):
        f = self.fpn(self.enc(x))
        outs = [self._heads(f)]
        for _ in range(self.passes - 1):
            f = f + self.refine(torch.cat([f, self._feedback(outs[-1]).detach()], 1))
            outs.append(self._heads(f))
        return {"passes": outs, **outs[-1]}

    def loss(self, out, t, attr_cls_mask):
        losses = {}; parts = {}
        n = len(out["passes"])
        for k, o in enumerate(out["passes"]):
            w = 1.0 if k == n - 1 else 0.5
            losses[f"hm{k}"] = w * focal(o["hm"], t["hm"])
            reg = gather_at(o["reg"], t["ind"]); wh, off = _box_l1(reg, t)
            losses[f"box{k}"] = w * (wh + off)
            al, parts = attr_loss(o["attr"], t, attr_cls_mask)
            losses[f"attr{k}"] = w * al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), {k: out[k] for k in ("hm", "reg", "attr")}, K, thresh, nms_iou)


class _Flat:
    """Duck-typed view for M.decode: a flat-mode model with a stride."""
    mode = "flat"

    def __init__(self, m):
        self.stride = m.stride; self.n_cls = m.n_cls


# ----------------------------------------------------------------------------------------------------------------
# 4. cnet-distill

_TEACHER = {}


def _teacher(path, device):
    """The deployed v6 (ultralytics DetectionModel), run layer by layer so the neck features can be read: layer 24
    is the stride-4 P2 neck output (96 ch), layer 15 the stride-8 P3 (192 ch)."""
    key = (path, str(device))
    if key not in _TEACHER:
        from ultralytics import YOLO
        m = YOLO(path).model.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _TEACHER[key] = m
    return _TEACHER[key]


@torch.no_grad()
def teacher_feats(m, x, layers=(24, 15)):
    y = []
    x = x.expand(-1, 3, -1, -1)
    for l in m.model:
        if l.f != -1:
            x = y[l.f] if isinstance(l.f, int) else [x if j == -1 else y[j] for j in l.f]
        x = l(x); y.append(x)
        if l.i == max(layers):
            break
    return [y[i].float() for i in layers]


def _standardise(f):
    mu = f.mean((0, 2, 3), keepdim=True); sd = f.std((0, 2, 3), keepdim=True) + 1e-3
    return (f - mu) / sd


class DistillNet(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64,
                 teacher="/root/omr/models/yolo_v6_best.pt", w_hint=1.0):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.adapt4 = nn.Conv2d(fpn, 96, 1); self.adapt8 = nn.Conv2d(widths[2], 192, 1)
        self.teacher_path, self.w_hint = teacher, w_hint

    def forward(self, x):
        p2, p3, p4, p5 = self.enc(x)
        f = self.fpn((p2, p3, p4, p5))
        out = {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}
        if self.training:
            t4, t8 = teacher_feats(_teacher(self.teacher_path, x.device), x)
            out["hint"] = (self.adapt4(f).float(), self.adapt8(p3).float(), _standardise(t4), _standardise(t8))
        return out

    def loss(self, out, t, attr_cls_mask):
        losses = {"hm": focal(out["hm"], t["hm"])}
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        s4, s8, t4, t8 = out["hint"]
        losses["hint"] = self.w_hint * (F.mse_loss(s4, t4) + F.mse_loss(s8, t8))
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), {k: out[k] for k in ("hm", "reg", "attr")}, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 5. cnet-twostream

class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), nn.SiLU(inplace=True), nn.Conv2d(c // r, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class TwoStream(Base):
    def __init__(self, n_cls, detail=(24, 48), detail_depth=3, ctx=(48, 112, 192, 256), ctx_depth=(2, 3, 4, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        s, cd = detail
        self.detail = nn.Sequential(ConvBNAct(1, s, 3, s=2), ConvBNAct(s, cd, 3, s=2), *[Basic(cd) for _ in range(detail_depth)])
        c8, c16, c32, c64 = ctx
        self.ctx_stem = ConvBNAct(1, c8, 3, s=2)                            # on the 4x-downsampled page: stride 8
        self.c8 = nn.Sequential(*[DW(c8) for _ in range(ctx_depth[0])])
        self.c16 = nn.Sequential(DWDown(c8, c16), *[DW(c16) for _ in range(ctx_depth[1])])
        self.c32 = nn.Sequential(DWDown(c16, c32), *[DW(c32) for _ in range(ctx_depth[2])])
        self.c64 = nn.Sequential(DWDown(c32, c64), *[DW(c64) for _ in range(ctx_depth[3])])
        self.lat = nn.ModuleList([nn.Conv2d(c, fpn, 1) for c in ctx])
        self.smooth = nn.ModuleList([ConvBNAct(fpn, fpn, 3, g=fpn) for _ in ctx[:-1]])
        self.fuse = nn.Sequential(ConvBNAct(cd + fpn, fpn, 1), SE(fpn), ConvBNAct(fpn, fpn, 3))
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        d = self.detail(x)
        xs = F.avg_pool2d(x, 4)
        c8 = self.c8(self.ctx_stem(xs)); c16 = self.c16(c8); c32 = self.c32(c16); c64 = self.c64(c32)
        y = self.lat[3](c64)
        for k, f in ((2, c32), (1, c16), (0, c8)):
            y = F.interpolate(y, size=f.shape[-2:], mode="nearest") + self.lat[k](f)
            y = self.smooth[k](y)
        y = F.interpolate(y, size=d.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([d, y], 1))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    def loss(self, out, t, attr_cls_mask):
        losses = {"hm": focal(out["hm"], t["hm"])}
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 6. cnet-embed

class EmbedNet(Base):
    def __init__(self, n_cls, fam_of_cls, n_fam, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, d=48):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.emb = nn.Sequential(ConvBNAct(fpn, hd, 3), ConvBNAct(hd, hd, 3, d=2), nn.Conv2d(hd, d, 1))
        self.fam_emb = nn.Parameter(torch.randn(n_fam, d // 2) * 0.05)
        self.sub_emb = nn.Parameter(torch.randn(n_cls, d // 2) * 0.05)
        self.bias = nn.Parameter(torch.full((n_cls,), -math.log(99.0)))
        self.register_buffer("fam_of_cls", torch.as_tensor(fam_of_cls, dtype=torch.long)); self.n_fam = n_fam
        self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def table(self):
        return torch.cat([self.fam_emb[self.fam_of_cls], self.sub_emb], 1)               # [n_cls, d]

    def forward(self, x):
        f = self.fpn(self.enc(x))
        e = self.emb(f)
        hm = (torch.einsum("bdhw,cd->bchw", e, self.table().to(e.dtype)) + self.bias.to(e.dtype)[None, :, None, None]).contiguous()
        return {"hm": hm, "reg": self.reg(f), "attr": self.attr(f)}

    def loss(self, out, t, attr_cls_mask):
        losses = {"hm": focal(out["hm"], t["hm"])}
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------

CONFIGS2 = {
    "peak-relational": (PeakRelational, {}),
    "seg-vote":        (SegVote, {}),
    "cnet-recur":      (RecurNet, {}),
    "cnet-distill":    (DistillNet, {}),
    "cnet-twostream":  (TwoStream, {}),
    "cnet-embed":      (EmbedNet, {}),
}


def build2(name, classes, **override):
    cls, kw = CONFIGS2[name]
    kw = {**kw, **override}
    if cls is EmbedNet:
        fams, fam_of_cls = M.families(classes)
        return cls(len(classes), fam_of_cls, len(fams), **kw)
    return cls(len(classes), **kw)
