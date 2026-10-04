"""Round 6 of the 2M-parameter search: the precision hypothesis, tested six ways.

Rounds 2 and 3 found that the only designs to close the precision gap on handwritten scans were the ones that
made a SECOND decision per candidate with an explicit background class (peak-relational on features, fovea on
pixels). Everything one-stage -- heatmap, dense per-cell, distributional -- shared one precision ceiling. Round 6
asks what exactly buys that precision, with designs that each isolate one candidate answer:

  softmax-bg    the one-stage readout with a softmax over 266 classes + background at every cell (SSD-style
                competition against "nothing") instead of 266 independent sigmoids. If the background class is
                what matters, this should show it without any second stage.
  embed-nms     a learnt per-cell instance embedding: cells of one glyph pull together, centres of different glyphs
                push apart; at decode, duplicate peaks are merged by EMBEDDING distance, not box overlap. If false
                positives are mostly duplicates and fragments, a learnt notion of "same object" beats IoU.
  duo-vote      two independent 1.0M models (half-width trunks) trained side by side; a detection both make (same
                class, IoU >= 0.5) keeps its full score, one only one makes is halved. Committee precision for the
                same parameter total.
  cls-prior-box the flat model whose box is a residual on a learnt per-class size prior: the head predicts
                log(w / mu_c), not log w. A class knows its shape; the regressor should only correct it.
  gfnet-ctx     global context by a learnt spectral filter (GFNet: FFT, multiply by a smooth learnt filter,
                inverse FFT) at strides 16 and 32 -- global mixing at the cost of a small conv, and the fourth
                mechanism for context after axial attention, a row GRU and windowed attention.
  scribble-aug  the flat model trained with random pen strokes and blots drawn on every page: explicit ink that is
                NOT a symbol. Handwritten scans are full of it; the synthetic corpus is not. A training-time input
                change, not an architecture -- kept because it targets the same gap as the five above.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR
from . import models as M
from .arch2 import Base, _box_l1, _dets, _peaks
from .arch4 import _Flat
from .arch5 import _flat_loss
from .models import ConvBNAct, Encoder, FPN, attr_loss, focal, gather_at, head


# ----------------------------------------------------------------------------------------------------------------
# 1. softmax-bg

class SoftmaxBg(Base):
    IGNORE = 0.3                 # cells with a Gaussian value above this (next to a centre) are neither positive nor negative

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.cls = head(fpn, hd, n_cls + 1)
        nn.init.constant_(self.cls[-1].bias, 0.0); self.cls[-1].bias.data[n_cls] = 4.0
        self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"cls": self.cls(f), "reg": self.reg(f), "attr": self.attr(f)}

    def loss(self, out, t, attr_cls_mask):
        B, C1, H, W = out["cls"].shape
        # per-cell target: the class at exact centres, background elsewhere, ignore the Gaussian halo
        tgt = torch.full((B, H * W), self.n_cls, dtype=torch.long, device=out["cls"].device)
        tgt.scatter_(1, t["ind"], torch.where(t["mask"], t["cls"], torch.full_like(t["cls"], self.n_cls)))
        tgt = tgt.reshape(B, H, W)
        halo = (t["hm"].amax(1) > self.IGNORE) & tgt.eq(self.n_cls)
        pos = tgt.ne(self.n_cls)
        ce = F.cross_entropy(out["cls"].float(), tgt, reduction="none")                     # [B, H, W]
        n_pos = int(pos.sum())
        pos_loss = ce[pos].sum()
        neg = ce.masked_fill(pos | halo, 0.0).reshape(B, -1)
        k = min(neg.shape[1], max(256, 3 * max(1, n_pos // B)))
        neg_loss = neg.topk(k, 1).values.sum()                                             # OHEM: the hardest 3 : 1 negatives
        losses = {"cls": (pos_loss + neg_loss) / max(1, n_pos + k * B) * 8.0}
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k_: float(v) for k_, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        p = torch.softmax(out["cls"].float(), 1)[:, :self.n_cls]                            # [B, C, H, W]
        # the heatmap the flat decoder expects: peaks of the class probabilities
        hm = torch.log(p.clamp(1e-6) / (1 - p).clamp(1e-6))                                 # logits of p, for M.decode's sigmoid
        return M.decode(_Flat(self), {"hm": hm, "reg": out["reg"], "attr": out["attr"]}, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 2. embed-nms

class EmbedNms(Base):
    D = 8
    RADIUS = 6                   # cells: peaks closer than this, same class, similar embedding -> one object

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 240), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.emb = head(fpn, 32, self.D)

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f), "emb": self.emb(f)}

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = _flat_loss(self, out, t, attr_cls_mask)
        B, _, H, W = out["emb"].shape
        e = out["emb"].float()
        ec = gather_at(e, t["ind"])                                                            # [B, K, D] centre embeddings
        # pull: the cells in each object's halo (Gaussian > 0.5) towards its centre embedding
        pull = e.new_zeros(()); push = e.new_zeros(()); n = 0
        d = torch.arange(-2, 3, device=e.device)
        gy = (t["ind"] // W)[:, :, None, None] + d[None, None, :, None]; gx = (t["ind"] % W)[:, :, None, None] + d[None, None, None, :]
        gy = gy.expand(B, -1, 5, 5); gx = gx.expand(B, -1, 5, 5)
        ok = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W) & t["mask"][:, :, None, None]
        for b in range(B):
            if not t["mask"][b].any():
                continue
            g = t["hm"][b].amax(0)[gy[b].clamp(0, H - 1), gx[b].clamp(0, W - 1)]                # [K, 5, 5]
            sel = ok[b] & (g > 0.5)
            cell_e = e[b][:, gy[b].clamp(0, H - 1), gx[b].clamp(0, W - 1)].permute(1, 2, 3, 0)  # [K, 5, 5, D]
            diff = ((cell_e - ec[b][:, None, None, :]) ** 2).sum(-1)
            pull = pull + (diff * sel).sum() / sel.sum().clamp(min=1)
            m = t["mask"][b]; c = ec[b][m]
            if len(c) > 1:
                dist = torch.cdist(c, c)
                push = push + (F.relu(1.0 - dist).triu(1).sum() / max(1, len(c) * (len(c) - 1) / 2))
            n += 1
        parts["pull"] = float(pull / max(1, n)); parts["push"] = float(push / max(1, n))
        return total + (pull + push) / max(1, n), parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.9):
        dets = M.decode(_Flat(self), {k: out[k] for k in ("hm", "reg", "attr")}, K, thresh, nms_iou)
        for b, d in enumerate(dets):
            n = len(d["boxes"])
            if n < 2:
                continue
            W = out["emb"].shape[-1]
            e = gather_at(out["emb"][b:b + 1], (d["cell"][:, 0] * W + d["cell"][:, 1])[None])[0]     # [n, D]
            order = d["scores"].argsort(descending=True)
            cy, cx = d["cell"][order, 0].float(), d["cell"][order, 1].float()
            same = d["cls"][order][:, None] == d["cls"][order][None, :]
            near = ((cy[:, None] - cy[None, :]) ** 2 + (cx[:, None] - cx[None, :]) ** 2) < self.RADIUS ** 2
            alike = torch.cdist(e[order], e[order]) < 0.5
            dup = (same & near & alike).triu(1)                                                       # [n, n]: j duplicates i (i stronger)
            keep = ~dup.any(0)
            idx = order[keep]
            for k in ("boxes", "scores", "cls", "attrs", "cell"):
                d[k] = d[k][idx]
        return dets


# ----------------------------------------------------------------------------------------------------------------
# 3. duo-vote

class DuoVote(Base):
    def __init__(self, n_cls, widths=(16, 32, 64, 128, 176), depths=(1, 2, 3, 2), fpn=48, hd=48, attr_hd=48):
        super().__init__(n_cls)
        self.nets = nn.ModuleList()
        for _ in range(2):
            m = nn.Module()
            m.enc = Encoder(widths, depths); m.fpn = FPN(m.enc.out_channels, fpn)
            m.hm = head(fpn, hd, n_cls, prior=0.01); m.reg = head(fpn, hd, 4)
            m.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
            self.nets.append(m)

    def forward(self, x):
        outs = []
        for m in self.nets:
            f = m.fpn(m.enc(x))
            outs.append({"hm": m.hm(f), "reg": m.reg(f), "attr": m.attr(f)})
        return {"nets": outs, **outs[0]}

    def loss(self, out, t, attr_cls_mask):
        total = 0.0; parts = {}; aparts = {}
        for i, o in enumerate(out["nets"]):
            l, p, a = _flat_loss(self, o, t, attr_cls_mask)
            total = total + l
            parts.update({f"{k}{i}": v for k, v in p.items()}); aparts = a
        return total, parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        from torchvision.ops import box_iou
        d0 = M.decode(_Flat(self), out["nets"][0], K, thresh, nms_iou)
        d1 = M.decode(_Flat(self), out["nets"][1], K, thresh, nms_iou)
        res = []
        for a, b in zip(d0, d1):
            if len(a["boxes"]) == 0 or len(b["boxes"]) == 0:
                for d in (a, b):
                    d["scores"] = d["scores"] * 0.5
                res.append({k: torch.cat([a[k], b[k]]) for k in a}); continue
            iou = box_iou(a["boxes"], b["boxes"]) * (a["cls"][:, None] == b["cls"][None, :])
            best, j = iou.max(1)
            agree = best >= 0.5
            # matched: averaged box and score from a's side; unmatched from either side at half score
            boxes = torch.where(agree[:, None], (a["boxes"] + b["boxes"][j]) / 2, a["boxes"])
            scores = torch.where(agree, (a["scores"] + b["scores"][j]) / 2, a["scores"] * 0.5)
            used = torch.zeros(len(b["boxes"]), dtype=torch.bool, device=iou.device); used[j[agree]] = True
            merged = {"boxes": torch.cat([boxes, b["boxes"][~used]]), "scores": torch.cat([scores, b["scores"][~used] * 0.5]),
                      "cls": torch.cat([a["cls"], b["cls"][~used]]), "attrs": torch.cat([a["attrs"], b["attrs"][~used]]),
                      "cell": torch.cat([a["cell"], b["cell"][~used]])}
            res.append(merged)
        return res


# ----------------------------------------------------------------------------------------------------------------
# 4. cls-prior-box

class ClsPriorBox(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.prior = nn.Parameter(torch.full((n_cls, 2), math.log(2.5)))             # per-class log (w, h) in cells, learnt

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    def loss(self, out, t, attr_cls_mask):
        losses = {"hm": focal(out["hm"], t["hm"])}
        reg = gather_at(out["reg"], t["ind"])
        # the head predicts the residual on the labelled class's prior; the prior itself learns the class mean
        pri = self.prior[t["cls"]]                                                          # [B, K, 2]
        m = t["mask"].unsqueeze(-1).float(); n = t["mask"].sum().clamp(min=1)
        losses["wh"] = 0.1 * (F.l1_loss(reg[..., :2] + pri, t["wh"], reduction="none") * m).sum() / n
        losses["prior"] = 0.1 * (F.l1_loss(pri, t["wh"], reduction="none") * m).sum() / n
        losses["off"] = (F.l1_loss(reg[..., 2:], t["off"], reduction="none") * m).sum() / n
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        dets = M.decode(_Flat(self), out, K, thresh, nms_iou)
        # M.decode used the raw residual as log w / log h: add the decoded class's prior back
        for b, d in enumerate(dets):
            if len(d["boxes"]) == 0:
                continue
            pri = torch.exp(self.prior[d["cls"]].float())                                   # [n, 2] multiplicative on the residual box
            cx = (d["boxes"][:, 0] + d["boxes"][:, 2]) / 2; cy = (d["boxes"][:, 1] + d["boxes"][:, 3]) / 2
            w = (d["boxes"][:, 2] - d["boxes"][:, 0]) * pri[:, 0]; h = (d["boxes"][:, 3] - d["boxes"][:, 1]) * pri[:, 1]
            d["boxes"] = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        return dets


# ----------------------------------------------------------------------------------------------------------------
# 5. gfnet-ctx

class GlobalFilter(nn.Module):
    """FFT -> multiply by a learnt complex filter (stored at a small fixed size, interpolated to the map) -> iFFT."""

    def __init__(self, c, fh=16, fw=9):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(c, fh, fw, 2))
        nn.init.normal_(self.w, std=0.02)
        self.w.data[..., 0] += 1.0                                                        # starts near identity
        self.n = nn.BatchNorm2d(c)

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x.float(), norm="ortho")                                        # [B, C, H, W//2+1]
        w = F.interpolate(self.w.permute(3, 0, 1, 2), size=xf.shape[-2:], mode="bilinear", align_corners=False)
        w = torch.complex(w[0], w[1])
        y = torch.fft.irfft2(xf * w[None], s=(H, W), norm="ortho")
        return x + self.n(y.to(x.dtype))


class GfnetCtx(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths)
        self.gf4 = GlobalFilter(widths[3]); self.gf5 = GlobalFilter(widths[4])
        self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        p2, p3, p4, p5 = self.enc(x)
        f = self.fpn((p2, p3, self.gf4(p4), self.gf5(p5)))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 6. scribble-aug

def scribble(h, w, rng):
    """A [h, w] float mask of random pen strokes and blots (darkness 0..1): 0-6 polylines of 2-5 points, width 1-3
    px, and 0-3 filled ellipses, drawn with cv2 on the CPU."""
    import cv2
    m = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(0, 7)):
        n = rng.integers(2, 6)
        pts = np.stack([rng.integers(0, w, n), rng.integers(0, h, n)], 1)
        # keep strokes local: a random walk of 20-150 px steps from the first point
        pts[1:] = pts[0] + np.cumsum(rng.integers(-150, 151, (n - 1, 2)), 0)
        cv2.polylines(m, [pts.astype(np.int32).reshape(-1, 1, 2)], False, float(rng.uniform(0.4, 1.0)), int(rng.integers(1, 4)), cv2.LINE_AA)
    for _ in range(rng.integers(0, 4)):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        ax = (int(rng.integers(2, 12)), int(rng.integers(2, 12)))
        cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(0.4, 1.0)), -1, cv2.LINE_AA)
    return m


class ScribbleAug(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, p=0.7):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.p = p; self._rng = np.random.default_rng(0)

    def forward(self, x):
        if self.training:
            B, _, H, W = x.shape
            masks = np.stack([scribble(H, W, self._rng) if self._rng.random() < self.p else np.zeros((H, W), np.float32) for _ in range(B)])
            x = (x - torch.from_numpy(masks).to(x.device)[:, None]).clamp(min=0)          # ink darkens the page
        f = self.fpn(self.enc(x))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


CONFIGS7 = {"softmax-bg": SoftmaxBg, "embed-nms": EmbedNms, "duo-vote": DuoVote, "cls-prior-box": ClsPriorBox,
            "gfnet-ctx": GfnetCtx, "scribble-aug": ScribbleAug}


def build7(name, classes, **override):
    return CONFIGS7[name](len(classes), **override)
