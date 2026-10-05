"""Round 7 of the 2M-parameter search: six mechanisms rounds 1-6 never used.

By now the search has varied the trunk (no effect), the readout (the large effect), the box parameterisation, the
assignment, the input view and the context mechanism. What no design has tried: refining a box more than once
against a rising quality bar, learning from the model's own averaged weights, spending the 266-channel output
budget somewhere else, changing the TRAINING DATA the sampler produces, predicting at more than one stride, and
fusing several scales at inference.

  cascade-iou   three box refinements in one pass, each supervised only on the cells whose current box already
                clears a rising IoU bar (0.5 -> 0.6 -> 0.7), as Cascade R-CNN does across heads. Aimed at
                mAP50-95: one regression cannot be accurate at every quality level.
  ema-consist   the model teaches itself: an exponential moving average of its own weights reads the clean crop,
                the trainable copy reads a photometrically wrecked one (noise, blur, gamma, dropout patches), and
                the student's heatmap is pulled towards the teacher's confident peaks on top of the labelled loss.
                A regulariser aimed at the out-of-domain gap (handwritten scans), not at the synthetic score.
  hash-cls      the class heatmap replaced by ONE objectness map plus 24 bit maps: every class has a fixed 24-bit
                code and a detection's class is the nearest code. The output stops growing with the taxonomy (25
                channels instead of 266) -- worth only ~16K parameters here, since the cost is in the head's 3x3
                convs rather than its last 1x1, so this is a test of the code representation, not of a budget move.
  copy-paste    glyph copy-paste augmentation between pages of the batch: real glyphs, with their labels, pasted
                into new contexts. The first design to change what the SAMPLER produces rather than what the
                network does with it.
  deep-super    anchor-free heads at strides 4, 8 and 16 with objects assigned to a level by size, weights shared
                across levels. Every earlier design read everything -- a 3 px dot and a 600 px bracket -- off one
                stride-4 map.
  scale-fuse    trained at one scale, decoded as the average of its own maps at 0.75x, 1x and 1.33x (shared
                weights, sizes and offsets corrected back to the 1x grid). Inference-time scale fusion: does
                scale sensitivity cap recall on real scans?
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR
from . import models as M
from .arch2 import Base, _box_l1, _dets, _peaks
from .arch4 import FcosDense, _Flat
from .arch5 import _flat_loss
from .models import ConvBNAct, Encoder, FPN, attr_loss, focal, gather_at, head


# ----------------------------------------------------------------------------------------------------------------
# 1. cascade-iou

def _ltrb_iou(a, b):
    """IoU of two [n, 4] distance quadruples (l, t, r, b) measured from the same cell."""
    iw = torch.minimum(a[:, 0], b[:, 0]) + torch.minimum(a[:, 2], b[:, 2])
    ih = torch.minimum(a[:, 1], b[:, 1]) + torch.minimum(a[:, 3], b[:, 3])
    inter = iw.clamp(min=0) * ih.clamp(min=0)
    aa = (a[:, 0] + a[:, 2]) * (a[:, 1] + a[:, 3]); bb = (b[:, 0] + b[:, 2]) * (b[:, 1] + b[:, 3])
    return inter / (aa + bb - inter + 1e-6)


class CascadeIou(FcosDense):
    THRESH = (0.0, 0.5, 0.65)          # a stage trains only where the previous box already reaches this IoU

    def __init__(self, n_cls, **kw):
        super().__init__(n_cls, **kw)
        fpn = self.ltrb[0][0].in_channels; hd = self.ltrb[0][0].out_channels
        self.ref = nn.ModuleList([head(fpn, hd // 2, 4) for _ in range(2)])
        for r in self.ref:
            nn.init.zeros_(r[-1].weight); nn.init.zeros_(r[-1].bias)

    def forward(self, x):
        f = self.fpn(self.enc(x))
        d0 = self.ltrb(f)
        d1 = d0 + self.ref[0](f)
        d2 = d1 + self.ref[1](f)
        return {"cls": self.cls(f), "ltrb": d2, "stages": [d0, d1, d2], "ctr": self.ctr(f), "attr": self.attr(f)}

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["cls"].shape
        cls_t, box_t, ctr_t, pos, attrs_t = self.targets(t, H, W)
        hm = torch.zeros(B, self.n_cls, H, W, device=out["cls"].device)
        hm.scatter_(1, cls_t.clamp(max=self.n_cls - 1)[:, None], pos[:, None].float())
        losses = {"cls": focal(out["cls"], hm)}
        gl = box_t.permute(0, 2, 3, 1)[pos]
        for k, d in enumerate(out["stages"]):
            pl = d.float().permute(0, 2, 3, 1)[pos]
            if not len(pl):
                continue
            pw = torch.exp(pl.clamp(-4, 6))
            keep = _ltrb_iou(pw.detach(), gl) >= self.THRESH[k] if k else torch.ones(len(pw), dtype=torch.bool, device=pw.device)
            if not keep.any():
                continue
            iou = _ltrb_iou(pw[keep], gl[keep])
            losses[f"iou{k}"] = (1.0 if k == len(out["stages"]) - 1 else 0.5) * -(torch.log(iou.clamp(min=1e-6))).mean()
        if pos.any():
            losses["ctr"] = F.binary_cross_entropy_with_logits(out["ctr"].float()[:, 0][pos], ctr_t[pos])
        ind = pos.reshape(B, -1).float().argsort(dim=1, descending=True)
        npos = pos.reshape(B, -1).sum(1); Kk = int(npos.max().clamp(min=1)); ind = ind[:, :Kk]
        tt = {"ind": ind, "mask": torch.arange(Kk, device=ind.device)[None] < npos[:, None],
              "cls": cls_t.reshape(B, -1).gather(1, ind).clamp(max=self.n_cls - 1),
              "attrs": attrs_t.reshape(B, -1, 5).gather(1, ind[..., None].expand(-1, -1, 5))}
        al, parts = attr_loss(out["attr"], tt, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts


# ----------------------------------------------------------------------------------------------------------------
# 2. ema-consist

def wreck(x, rng=None):
    """Photometric only -- the geometry (and so every label) is untouched: gamma, contrast, gaussian noise, blur,
    and a few dropped patches."""
    B = x.shape[0]
    g = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.6, 1.6)
    y = x.clamp(0, 1) ** g
    c = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
    y = ((y - 0.5) * c + 0.5)
    y = y + torch.randn_like(y) * torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.0, 0.08)
    if torch.rand(()) < 0.5:
        k = torch.tensor([[1.0, 2, 1], [2, 4, 2], [1, 2, 1]], device=x.device)[None, None] / 16
        y = torch.where(torch.rand(B, 1, 1, 1, device=x.device) < 0.5, F.conv2d(y, k, padding=1), y)
    m = (torch.rand(B, 1, x.shape[2] // 64, x.shape[3] // 64, device=x.device) > 0.03).float()
    y = y * F.interpolate(m, size=x.shape[-2:], mode="nearest") + (1 - F.interpolate(m, size=x.shape[-2:], mode="nearest"))
    return y.clamp(0, 1)


class EmaConsist(Base):
    DECAY = 0.999
    W_CONS = 1.0
    CONF = 0.3

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self._ema = None                       # plain tensors (not Parameters): the teacher is not a second model

    def _core(self, x, params=None):
        if params is None:
            f = self.fpn(self.enc(x))
            return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}
        from torch.func import functional_call
        return functional_call(self, params, (x,), {"_teacher": True})

    def forward(self, x, _teacher=False):
        if _teacher or not self.training:
            f = self.fpn(self.enc(x))
            return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}
        with torch.no_grad():
            if self._ema is None:
                self._ema = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                for k, v in self.state_dict().items():
                    if v.dtype.is_floating_point:
                        self._ema[k].mul_(self.DECAY).add_(v.detach(), alpha=1 - self.DECAY)
                    else:
                        self._ema[k].copy_(v)
            t_out = self._core(x, self._ema)
        out = self._core(wreck(x))
        out["teacher_hm"] = t_out["hm"].detach()
        return out

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = _flat_loss(self, out, t, attr_cls_mask)
        with torch.no_grad():
            tp = torch.sigmoid(out["teacher_hm"].float())
            m = (tp > self.CONF).float()
        if m.sum() > 0:
            sp = out["hm"].float()
            cons = (F.binary_cross_entropy_with_logits(sp, tp, reduction="none") * m).sum() / m.sum()
        else:
            cons = out["hm"].sum() * 0
        parts["cons"] = float(cons); aparts["teach_px"] = float(m.sum())
        return total + self.W_CONS * cons, parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), {k: out[k] for k in ("hm", "reg", "attr")}, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 3. hash-cls

def class_codes(n_cls, bits, seed=0):
    """Balanced binary codes, greedily chosen for a large minimum Hamming distance."""
    g = torch.Generator().manual_seed(seed)
    pool = (torch.rand(n_cls * 40, bits, generator=g) < 0.5).float()
    codes = pool[:1]
    for c in pool[1:]:
        if len(codes) >= n_cls:
            break
        if (codes - c[None]).abs().sum(1).min() >= max(4, bits // 5):
            codes = torch.cat([codes, c[None]])
    while len(codes) < n_cls:                                   # top up if the greedy pass fell short
        codes = torch.cat([codes, (torch.rand(1, bits, generator=g) < 0.5).float()])
    return codes[:n_cls]


class HashCls(Base):
    BITS = 24

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 264), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.obj = head(fpn, hd, 1, prior=0.01)
        self.bits = head(fpn, hd, self.BITS)
        self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.register_buffer("codes", class_codes(n_cls, self.BITS))        # [n_cls, BITS] in {0, 1}

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"obj": self.obj(f), "bits": self.bits(f), "reg": self.reg(f), "attr": self.attr(f)}

    def loss(self, out, t, attr_cls_mask):
        losses = {"obj": focal(out["obj"], t["hm"].amax(1, keepdim=True))}
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        b = gather_at(out["bits"], t["ind"])                                  # [B, K, BITS]
        m = t["mask"]
        if m.any():
            tgt = self.codes[t["cls"][m]]
            losses["bits"] = F.binary_cross_entropy_with_logits(b[m].float(), tgt) * 4.0
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        B, _, H, W = out["obj"].shape
        obj = torch.sigmoid(out["obj"].float())
        res = []
        for b in range(B):
            s, idx = _peaks(obj[b], K, thresh)
            if len(idx) == 0:
                z = torch.zeros(0, device=obj.device)
                res.append(_dets(torch.zeros(0, 4, device=obj.device), z, z.long(), torch.zeros(0, N_ATTR, device=obj.device), z.long(), z.long(), 0)); continue
            cell = idx % (H * W); cy = cell // W; cx = cell % W
            p = torch.sigmoid(out["bits"][b].float().reshape(self.BITS, -1)[:, cell].t())        # [n, BITS]
            # log-likelihood of every class code, i.e. nearest code in a soft Hamming sense
            ll = p.clamp(1e-4, 1 - 1e-4).log() @ self.codes.t() + (1 - p).clamp(1e-4, 1 - 1e-4).log() @ (1 - self.codes).t()
            pc, cls = torch.softmax(ll, -1).max(-1)
            reg = out["reg"][b].float().reshape(4, -1)[:, cell]
            w = torch.exp(reg[0]) * self.stride; h = torch.exp(reg[1]) * self.stride
            x = (cx.float() + reg[2]) * self.stride; y = (cy.float() + reg[3]) * self.stride
            boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, cell].t()
            res.append(_dets(boxes, s * pc, cls, attrs, cy, cx, nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 4. copy-paste

class CopyPaste(Base):
    """The flat model; the novelty is ``augment_batch`` (copisteria.detector.small.train calls it before the targets are rendered):
    glyphs are cut from one page of the batch, with their labels, and pasted onto another."""

    N_PASTE = (8, 40)
    P = 0.8

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    @staticmethod
    def augment_batch(batch):
        import random
        img = batch["img"]; boxes = batch["boxes"]
        B, _, H, W = img.shape
        if B < 2:
            return batch
        for b in range(B):
            if random.random() > CopyPaste.P:
                continue
            src = random.randrange(B - 1)
            src = src + 1 if src >= b else src
            sb = boxes[src]
            if len(sb) == 0:
                continue
            n = random.randint(*CopyPaste.N_PASTE)
            pick = torch.randint(0, len(sb), (min(n, len(sb)),))
            new = []
            for i in pick.tolist():
                x0, y0, x1, y1 = [int(v) for v in sb[i, 1:5].tolist()]
                w, h = x1 - x0, y1 - y0
                if w < 2 or h < 2 or w > W // 3 or h > H // 3:
                    continue
                dx = random.randint(0, W - w - 1); dy = random.randint(0, H - h - 1)
                patch = img[src, :, y0:y0 + h, x0:x0 + w]
                if patch.shape[-1] != w or patch.shape[-2] != h:
                    continue
                # ink is dark: keep the darker of the two, so paper does not erase what is already there
                img[b, :, dy:dy + h, dx:dx + w] = torch.minimum(img[b, :, dy:dy + h, dx:dx + w], patch)
                row = sb[i].clone(); row[1] = dx; row[2] = dy; row[3] = dx + w; row[4] = dy + h
                new.append(row)
            if new:
                boxes[b] = torch.cat([boxes[b], torch.stack(new)])
        return batch

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 5. deep-super

class DeepSuper(Base):
    LIMITS = (16.0, 48.0)              # object size in pixels: < 16 -> stride 4, < 48 -> stride 8, else stride 16

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths)
        self.lat = nn.ModuleList([nn.Conv2d(c, fpn, 1) for c in self.enc.out_channels])
        self.smooth = nn.ModuleList([ConvBNAct(fpn, fpn, 3) for _ in range(3)])
        self.hm = head(fpn, hd, n_cls, prior=0.01)                # shared across the three levels
        self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.strides = (4, 8, 16)

    def forward(self, x):
        p2, p3, p4, p5 = self.enc(x)
        y = self.lat[3](p5)
        outs = {}
        feats = []
        for k, f in ((2, p4), (1, p3), (0, p2)):
            y = F.interpolate(y, size=f.shape[-2:], mode="nearest") + self.lat[k](f)
            y = self.smooth[k](y)
            feats.append((k, y))
        feats = {k: v for k, v in feats}                          # 0 -> stride 4, 1 -> 8, 2 -> 16
        for li in range(3):
            f = feats[li]
            outs[li] = {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}
        return {"levels": outs, **outs[0]}

    def _level_targets(self, t, li, H, W):
        """The batch's objects that belong to level ``li``, as boxes in that level's pixels."""
        s = self.strides[li]
        out = []
        B, K = t["ind"].shape
        W4 = self._W
        for b in range(B):
            m = t["mask"][b]
            if not m.any():
                out.append(torch.zeros(0, 10, device=t["ind"].device)); continue
            ind = t["ind"][b][m]
            cx = ((ind % W4).float() + t["off"][b][m][:, 0]) * self.stride
            cy = ((ind // W4).float() + t["off"][b][m][:, 1]) * self.stride
            w = torch.exp(t["wh"][b][m][:, 0]) * self.stride; h = torch.exp(t["wh"][b][m][:, 1]) * self.stride
            size = torch.sqrt(w * h)
            lo = 0.0 if li == 0 else self.LIMITS[li - 1]
            hi = self.LIMITS[li] if li < 2 else float("inf")
            sel = (size >= lo) & (size < hi)
            bx = torch.stack([t["cls"][b][m].float(), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            out.append(torch.cat([bx, t["attrs"][b][m].float()], 1)[sel])
        return out

    def loss(self, out, t, attr_cls_mask):
        from .data import render_targets
        losses = {}; parts = {}
        H4, W4 = out["levels"][0]["hm"].shape[-2:]
        self._W = W4
        for li in range(3):
            o = out["levels"][li]
            Hs, Ws = o["hm"].shape[-2:]
            bx = self._level_targets(t, li, Hs, Ws)
            tl = render_targets(bx, Hs * self.strides[li], Ws * self.strides[li], self.strides[li], self.n_cls, o["hm"].device)
            losses[f"hm{li}"] = focal(o["hm"], tl["hm"])
            reg = gather_at(o["reg"], tl["ind"]); wh, off = _box_l1(reg, tl)
            losses[f"box{li}"] = wh + off
            al, p = attr_loss(o["attr"], tl, attr_cls_mask)
            losses[f"attr{li}"] = al
            if li == 0:
                parts = p
            parts[f"n{li}"] = int(tl["mask"].sum())
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        per = None
        for li in range(3):
            o = out["levels"][li]
            m = _Flat(self); m.stride = self.strides[li]
            d = M.decode(m, o, K, thresh, 0.0)
            per = d if per is None else [{k: torch.cat([a[k], b[k]]) for k in a} for a, b in zip(per, d)]
        from torchvision.ops import batched_nms
        res = []
        for d in per:
            if len(d["boxes"]):
                keep = batched_nms(d["boxes"], d["scores"], d["cls"], nms_iou)
                d = {k: v[keep] for k, v in d.items()}
            res.append(d)
        return res


# ----------------------------------------------------------------------------------------------------------------
# 6. scale-fuse

class ScaleFuse(Base):
    SCALES = (0.75, 1.0, 1.33)

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def _once(self, x):
        f = self.fpn(self.enc(x))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    def forward(self, x):
        if self.training:
            return self._once(x)
        B, _, H, W = x.shape
        base = self._once(x)
        Hs, Ws = base["hm"].shape[-2:]
        hm = torch.sigmoid(base["hm"].float()); reg = base["reg"].float(); attr = base["attr"].float(); n = 1
        for s in self.SCALES:
            if abs(s - 1.0) < 1e-6:
                continue
            h2 = int(round(H * s / 32)) * 32; w2 = int(round(W * s / 32)) * 32
            o = self._once(F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False))
            hm = hm + F.interpolate(torch.sigmoid(o["hm"].float()), size=(Hs, Ws), mode="bilinear", align_corners=False)
            r = F.interpolate(o["reg"].float(), size=(Hs, Ws), mode="bilinear", align_corners=False)
            r = torch.cat([r[:, :2] - math.log(s), r[:, 2:]], 1)      # log sizes measured in that scale's cells
            reg = reg + r
            attr = attr + F.interpolate(o["attr"].float(), size=(Hs, Ws), mode="bilinear", align_corners=False)
            n += 1
        hm = (hm / n).clamp(1e-6, 1 - 1e-6)
        return {"hm": torch.log(hm / (1 - hm)), "reg": reg / n, "attr": attr / n}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


CONFIGS8 = {"cascade-iou": CascadeIou, "ema-consist": EmaConsist, "hash-cls": HashCls,
            "copy-paste": CopyPaste, "deep-super": DeepSuper, "scale-fuse": ScaleFuse}


def build8(name, classes, **override):
    return CONFIGS8[name](len(classes), **override)
