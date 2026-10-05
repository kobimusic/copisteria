"""The precision ladder: what is the CHEAPEST mechanism that breaks the one-stage precision ceiling?

Five one-stage designs have now been scored on the real handwritten scans and they land within five points of each
other -- cnet-flat 0.65, reppoints 0.66, fcos-dense 0.69, gfl-dense 0.64, simota 0.64 -- regardless of how positives
are assigned, how the box is parameterised, or whether the score is calibrated. Two designs broke that ceiling and
both make a SECOND decision per candidate against an explicit background class: peak-relational (a transformer over
the candidates, 0.91) and fovea (a CNN over a pixel crop, 0.88 on the old corpus). Neither is cheap.

These six take the mechanism apart. The first three are a ladder of decreasing cost around the same idea, so that
whichever rung still works names the thing that matters:

  bg-dense      no second stage at all: the flat model plus one extra dense channel, a verifier trained only at
                candidate locations -- GT centres as positives, the model's own strongest false peaks as negatives,
                mined afresh every step. The score is the heatmap times the verifier. If this works, "a second
                decision on mined negatives" is the whole mechanism and it costs 40K parameters.
  bg-verify     one rung up: the peaks are gathered and a two-layer MLP over the RoI features decides class or
                background -- a real second stage, but with no attention between candidates. Isolates the second
                decision from the relational reasoning peak-relational also does.
  ctx-crop      the same verifier reading a FIVE times wider window from the stride-16 map instead of a tight RoI:
                is it the decision that matters, or how much context the decider sees?
  obj-recall    peak-relational's weakness is recall (0.87 against a heatmap's 0.95). Its proposal stage is trained
                with focal loss, which suppresses weak peaks; here objectness is trained with sampled negatives and
                4000 candidates are carried to the verifier, trading proposal precision (which the verifier can fix)
                for proposal recall (which it cannot).
  self-nms      a learnt duplicate test instead of IoU: a head predicts, for each cell, whether a stronger peak of
                the same object lies nearby, and decoding drops those instead of running NMS. Aimed at the
                false-positive half of the gap rather than at classification.
  mix-teacher   compile the expensive model into a cheap one: a frozen peak-relational scores each crop and its
                detections become a soft heatmap target for a plain one-stage student, alongside the labels. If a
                one-stage student inherits the precision, deployment never needs the second stage.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR
from . import models as M
from .arch2 import Base, PeakRelational, _box_l1, _dets, _peaks, _sincos_pos
from .arch4 import _Flat
from .arch5 import _flat_loss
from .models import ConvBNAct, Encoder, FPN, attr_loss, focal, gather_at, head


def _mined_negatives(hm_logits, t, n_neg):
    """The model's own strongest peaks that are not GT centres -- [B, n_neg] cell indices and a validity mask."""
    B, C, H, W = hm_logits.shape
    with torch.no_grad():
        p = torch.sigmoid(hm_logits.detach().float()).amax(1)
        pooled = F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]
        peaks = (p * (pooled == p)).reshape(B, -1)
        gt = torch.zeros(B, H * W, dtype=torch.bool, device=p.device).scatter_(1, t["ind"], t["mask"])
        s, i = peaks.masked_fill(gt, 0.0).topk(n_neg, 1)
    return i, s > 0.02


# ----------------------------------------------------------------------------------------------------------------
# 1. bg-dense

class BgDense(Base):
    N_NEG = 256

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.ver = head(fpn, 32, 1, dilations=(1, 2), prior=0.5)         # "is a candidate here a real object?"

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f), "ver": self.ver(f)}

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = _flat_loss(self, out, t, attr_cls_mask)
        neg_i, neg_m = _mined_negatives(out["hm"], t, self.N_NEG)
        v = out["ver"][:, 0].reshape(out["ver"].shape[0], -1).float()
        pos_v = v.gather(1, t["ind"])[t["mask"]]
        neg_v = v.gather(1, neg_i)[neg_m]
        lv = out["ver"].new_zeros((), dtype=torch.float32)
        if len(pos_v):
            lv = lv + F.binary_cross_entropy_with_logits(pos_v, torch.ones_like(pos_v))
        if len(neg_v):
            lv = lv + F.binary_cross_entropy_with_logits(neg_v, torch.zeros_like(neg_v))
        parts["ver"] = float(lv); aparts["n_neg"] = int(neg_m.sum())
        return total + lv, parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        ver = torch.sigmoid(out["ver"].float())                          # [B, 1, H, W]
        hm = torch.sigmoid(out["hm"].float()) * ver
        hm = torch.log(hm.clamp(1e-6, 1 - 1e-6) / (1 - hm.clamp(1e-6, 1 - 1e-6)))
        return M.decode(_Flat(self), {"hm": hm, "reg": out["reg"], "attr": out["attr"]}, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 2 + 3. bg-verify and ctx-crop (a second stage without attention; the RoI is tight or five times wider)

class BgVerify(Base):
    N_NEG = 192
    K_INFER = 2000
    W_OBJ = 1.0                     # softmax-verify sets this to 0: its dense softmax already trains the proposals
    WIDE = 1.0                      # x the predicted box; ctx-crop overrides
    ROI = 3

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, d=128):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.obj = head(fpn, hd, 1, dilations=(1, 2), prior=0.01)
        self.reg = head(fpn, hd, 4)
        self.d = d
        self.proj = nn.Sequential(nn.Linear(fpn * self.ROI * self.ROI + fpn, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.cls = nn.Linear(d, n_cls + 1)
        self.attr = nn.Linear(d, N_ATTR)
        nn.init.constant_(self.cls.bias, 0.0); self.cls.bias.data[n_cls] = 2.0

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"f": f, "obj": self.obj(f), "reg": self.reg(f)}

    def _tokens(self, f, reg, b_idx, cy, cx):
        from torchvision.ops import roi_align
        W = f.shape[-1]
        r = reg[b_idx, :, cy, cx].float()
        w = torch.exp(r[:, 0].clamp(-2, 6)) * self.stride * self.WIDE
        h = torch.exp(r[:, 1].clamp(-2, 6)) * self.stride * self.WIDE
        px = (cx.float() + r[:, 2]) * self.stride; py = (cy.float() + r[:, 3]) * self.stride
        boxes = torch.stack([b_idx.float(), px - w / 2, py - h / 2, px + w / 2, py + h / 2], 1)
        roi = roi_align(f.float(), boxes.detach(), (self.ROI, self.ROI), spatial_scale=1.0 / self.stride,
                        sampling_ratio=2, aligned=True).reshape(len(b_idx), -1)
        centre = f[b_idx, :, cy, cx].float()
        return self.proj(torch.cat([roi, centre], -1)), r

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["obj"].shape
        losses = {}
        if self.W_OBJ:
            losses["obj"] = self.W_OBJ * focal(out["obj"], t["hm"].amax(1, keepdim=True))
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        neg_i, neg_m = _mined_negatives(out["obj"].detach().expand(-1, 2, -1, -1), t, self.N_NEG)
        ind = torch.cat([t["ind"], neg_i], 1); mask = torch.cat([t["mask"], neg_m], 1)
        b_idx = torch.arange(B, device=ind.device)[:, None].expand_as(ind).reshape(-1)
        cy, cx = (ind // W).reshape(-1), (ind % W).reshape(-1)
        tok, _ = self._tokens(out["f"], out["reg"], b_idx, cy, cx)
        logit = self.cls(tok); attr = self.attr(tok)
        K = t["ind"].shape[1]
        target = torch.cat([t["cls"], torch.full_like(neg_i, self.n_cls)], 1).reshape(-1)
        m = mask.reshape(-1)
        losses["cls"] = F.cross_entropy(logit[m].float(), target[m])
        a = attr.reshape(B, -1, N_ATTR)[:, :K]
        tt = {"ind": torch.arange(K, device=a.device)[None].expand(B, K), "mask": t["mask"], "cls": t["cls"], "attrs": t["attrs"]}
        al, parts = attr_loss(a.permute(0, 2, 1)[..., None], tt, attr_cls_mask)
        losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        B, _, H, W = out["obj"].shape
        obj = torch.sigmoid(out["obj"].float())
        res = []
        for b in range(B):
            s, idx = _peaks(obj[b], self.K_INFER, thresh * 0.5)
            if len(idx) == 0:
                z = torch.zeros(0, device=obj.device)
                res.append(_dets(torch.zeros(0, 4, device=obj.device), z, z.long(), torch.zeros(0, N_ATTR, device=obj.device), z.long(), z.long(), 0)); continue
            cy, cx = idx // W, idx % W
            tok, r = self._tokens(out["f"], out["reg"], torch.full_like(cy, b), cy, cx)
            p = torch.softmax(self.cls(tok).float(), -1)
            pc, cls = p[:, :self.n_cls].max(-1)
            score = s * pc
            w = torch.exp(r[:, 0]) * self.stride; h = torch.exp(r[:, 1]) * self.stride
            x = (cx.float() + r[:, 2]) * self.stride; y = (cy.float() + r[:, 3]) * self.stride
            boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
            keep = score > thresh
            res.append(_dets(boxes[keep], score[keep], cls[keep], self.attr(tok).float()[keep], cy[keep], cx[keep], nms_iou))
        return res


class CtxCrop(BgVerify):
    WIDE = 5.0
    ROI = 5

    def __init__(self, n_cls, **kw):
        super().__init__(n_cls, **kw)


# ----------------------------------------------------------------------------------------------------------------
# 4. obj-recall

class ObjRecall(PeakRelational):
    """peak-relational with a recall-first proposal stage: objectness trained as plain BCE on sampled negatives
    (focal drives weak peaks to zero) and four times as many candidates carried to the verifier."""

    def __init__(self, n_cls, **kw):
        kw.setdefault("k_infer", 4000)
        super().__init__(n_cls, **kw)

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = super().loss(out, t, attr_cls_mask)
        # replace the focal objectness term with BCE on the centres + a sample of negatives
        B, _, H, W = out["obj"].shape
        v = out["obj"][:, 0].reshape(B, -1).float()
        pos = v.gather(1, t["ind"])[t["mask"]]
        neg_i = torch.randint(0, H * W, (B, 2048), device=v.device)
        neg = v.gather(1, neg_i).reshape(-1)
        l = out["obj"].new_zeros((), dtype=torch.float32)
        if len(pos):
            l = l + F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) * 2.0
        l = l + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        total = total + l                                      # the focal term stays; BCE on sampled negatives is added
        parts["obj_bce"] = float(l)
        return total, parts, aparts


# ----------------------------------------------------------------------------------------------------------------
# 5. self-nms

class SelfNms(Base):
    R = 5                       # cells: a stronger peak within this radius makes a candidate a duplicate

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 248), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.dup = head(fpn, 32, 1, dilations=(1, 2), prior=0.5)

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f), "dup": self.dup(f)}

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = _flat_loss(self, out, t, attr_cls_mask)
        B, _, H, W = out["dup"].shape
        with torch.no_grad():
            # a cell is a duplicate if a GT centre lies within R cells but is not its own cell
            gt = torch.zeros(B, H * W, device=out["dup"].device).scatter_(1, t["ind"], t["mask"].float()).reshape(B, 1, H, W)
            near = (F.max_pool2d(gt, 2 * self.R + 1, 1, self.R) > 0).float()
            dup_t = (near - gt).clamp(min=0)                    # near a centre but not the centre
            w = near                                            # train only in the neighbourhood of real objects
        d = out["dup"].float()
        lv = (F.binary_cross_entropy_with_logits(d, dup_t, reduction="none") * w).sum() / w.sum().clamp(min=1)
        parts["dup"] = float(lv)
        return total + lv, parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.0):
        dup = torch.sigmoid(out["dup"].float())
        hm = torch.sigmoid(out["hm"].float()) * (1 - dup)        # a learnt duplicate test instead of IoU NMS
        hm = torch.log(hm.clamp(1e-6, 1 - 1e-6) / (1 - hm.clamp(1e-6, 1 - 1e-6)))
        return M.decode(_Flat(self), {"hm": hm, "reg": out["reg"], "attr": out["attr"]}, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 6. mix-teacher

_TEACHER = {}


def _peak_teacher(path, classes, device):
    key = (path, str(device))
    if key not in _TEACHER:
        ck = torch.load(path, map_location=device, weights_only=False)
        m = M.build(ck["arch"], classes).to(device).eval()
        m.load_state_dict(ck["ema"])
        for p in m.parameters():
            p.requires_grad_(False)
        _TEACHER[key] = m
    return _TEACHER[key]


class MixTeacher(Base):
    TEACHER = "/root/omr/runs/ref/peak-relational/best.pt"
    W_SOFT = 1.0

    def __init__(self, n_cls, classes=None, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self._classes = classes

    def forward(self, x):
        f = self.fpn(self.enc(x))
        out = {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}
        if self.training and self._classes is not None:
            import os
            if os.path.exists(self.TEACHER):
                t = _peak_teacher(self.TEACHER, self._classes, x.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    to = t(x)
                    dets = M.decode(t, to, thresh=0.15)
                out["soft"] = self._render(dets, out["hm"].shape)
        return out

    def _render(self, dets, shape):
        """The teacher's detections as a soft target: its score at each detection's centre cell."""
        B, C, H, W = shape
        soft = torch.zeros(B, C, H, W, device=dets[0]["boxes"].device if dets else "cpu")
        for b, d in enumerate(dets):
            if not len(d["boxes"]):
                continue
            cx = ((d["boxes"][:, 0] + d["boxes"][:, 2]) / 2 / self.stride).long().clamp(0, W - 1)
            cy = ((d["boxes"][:, 1] + d["boxes"][:, 3]) / 2 / self.stride).long().clamp(0, H - 1)
            soft[b, d["cls"], cy, cx] = torch.maximum(soft[b, d["cls"], cy, cx], d["scores"])
        return soft

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = _flat_loss(self, out, t, attr_cls_mask)
        if "soft" in out:
            m = (out["soft"] > 0).float()
            if m.sum() > 0:
                soft = (F.binary_cross_entropy_with_logits(out["hm"].float(), out["soft"], reduction="none") * m).sum() / m.sum()
                parts["soft"] = float(soft); aparts["teach_px"] = float(m.sum())
                total = total + self.W_SOFT * soft
        return total, parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), {k: out[k] for k in ("hm", "reg", "attr")}, K, thresh, nms_iou)


class ObjRecallWide(ObjRecall):
    """obj-recall with a wider trunk (3.6M): the 2.0M cap was a rule of the search, not of the deployed model, and
    the winning design now trains on five times the pages it was chosen on."""

    def __init__(self, n_cls, **kw):
        kw.setdefault("widths", (32, 72, 144, 256, 384))
        kw.setdefault("depths", (2, 3, 4, 3))
        kw.setdefault("fpn", 96)
        kw.setdefault("hd", 96)
        super().__init__(n_cls, **kw)


class ObjRecall30M(ObjRecall):
    """obj-recall at the deployed detector's scale (~30M, v6 is 26M): the same design -- recall-first proposals, a
    gathered verifier deciding each candidate against an explicit background class, a transformer between the
    candidates -- with every width roughly doubled again over the 5.3M variant and a three-layer, 256-wide decoder."""

    def __init__(self, n_cls, **kw):
        kw.setdefault("widths", (64, 128, 256, 512, 768))
        kw.setdefault("depths", (2, 4, 8, 4))
        kw.setdefault("fpn", 192)
        kw.setdefault("hd", 192)
        kw.setdefault("d", 256)
        kw.setdefault("layers", 3)
        super().__init__(n_cls, **kw)


CONFIGS9 = {"obj-recall-30m": ObjRecall30M, "bg-dense": BgDense, "bg-verify": BgVerify, "ctx-crop": CtxCrop, "obj-recall": ObjRecall,
            "obj-recall-wide": ObjRecallWide, "self-nms": SelfNms, "mix-teacher": MixTeacher}


def build9(name, classes, **override):
    if name == "mix-teacher":
        override.setdefault("classes", classes)
    return CONFIGS9[name](len(classes), **override)
