"""Round 3 of the 2M-parameter search: six designs aimed at what rounds 1-2 left open.

Round 2's reading was that the READOUT limits a 2M model, not the trunk (feature hints from the 26M v6, a split
detail/context trunk and a recurrent second pass all landed within 3 points of the plain centre-heatmap model),
and that the one design which changed how objects are decided -- a relational second stage -- was the one that
closed the precision gap. Two numbers are still far from the reference: mAP50-95 (0.33 against v6's 0.81, i.e.
boxes are found but loosely placed) and recall on small ornaments. Round 3 therefore varies the readout, the
output resolution and the box representation, and adds the one kind of context nothing has used yet -- direction.

  scan-state    row-wise bidirectional GRU across the page at stride 16, fused back into the trunk: a staff is
                read left to right, and clef / key / accidental state propagates that way. Round 1's axial
                attention was all-pairs and orderless; a scan carries state and costs a fraction of the memory.
  fovea         two looks, the second at the PIXELS: a class-agnostic objectness map proposes centres, then a
                tiny expert re-reads a 32x32 patch of the input image around each candidate, resampled to the
                candidate's own size (so every glyph reaches the expert at one scale) and classifies it there.
                Round 2's peak-relational re-read the stride-4 FEATURES; this asks whether the trunk's
                downsampling is what caps small-glyph accuracy.
  dyn-proposals Sparse-R-CNN-lite: 400 learned proposal boxes and their feature vectors, refined twice by dynamic
                instance interaction (each proposal generates its own 1x1 filters over its RoI), self-attention
                among proposals, Hungarian matching at PAGE level. Unlike col-slots the proposals are learned
                constants refined iteratively, and unlike peak-relational they are not read off a heatmap.
  ink-proto     YOLACT-style instance masks: 16 prototype maps at stride 4 + per-object coefficients at the
                centre; the mask is supervised by the INK itself (dark pixels inside the labelled box -- free
                supervision the corpus already carries) and the box at inference is the mask's extent. Aimed
                squarely at mAP50-95: a box read off a mask beats a regressed width and height.
  hires-out     the same centre readout at stride 2 (PixelShuffle from the stride-4 map): an augmentation dot is
                3 px wide and had one cell to itself; now it has four. Tests output resolution alone.
  fcos-dense    every cell inside a glyph predicts the four distances to its box sides plus a centerness weight
                (FCOS), instead of one centre cell predicting width and height. Dense box supervision, a
                different box parameterisation, and no notion of a centre at all.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR
from . import models as M
from .arch2 import Base, _box_l1, _dets, _peaks, _sincos_pos
from .models import Basic, ConvBNAct, DW, DWDown, Encoder, FPN, attr_loss, focal, gather_at, head


# ----------------------------------------------------------------------------------------------------------------
# helpers

def _giou(a, b, eps=1e-7):
    """Generalised IoU of two [n, 4] xyxy box sets, elementwise."""
    ax0, ay0, ax1, ay1 = a.unbind(-1); bx0, by0, bx1, by1 = b.unbind(-1)
    aw = (ax1 - ax0).clamp(min=0); ah = (ay1 - ay0).clamp(min=0)
    bw = (bx1 - bx0).clamp(min=0); bh = (by1 - by0).clamp(min=0)
    inter = (torch.minimum(ax1, bx1) - torch.maximum(ax0, bx0)).clamp(min=0) * (torch.minimum(ay1, by1) - torch.maximum(ay0, by0)).clamp(min=0)
    union = aw * ah + bw * bh - inter
    iou = inter / (union + eps)
    cw = (torch.maximum(ax1, bx1) - torch.minimum(ax0, bx0)).clamp(min=0)
    ch = (torch.maximum(ay1, by1) - torch.minimum(ay0, by0)).clamp(min=0)
    c = cw * ch
    return iou - (c - union) / (c + eps)


def _patches(maps, cy, cx, r):
    """[C, H, W] map, centres cy / cx [K] (cells) -> [K, C, 2r+1, 2r+1] patches, zero outside."""
    C, H, W = maps.shape
    d = torch.arange(-r, r + 1, device=maps.device)
    gy = cy[:, None, None] + d[None, :, None]; gx = cx[:, None, None] + d[None, None, :]
    ok = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
    p = maps[:, gy.clamp(0, H - 1), gx.clamp(0, W - 1)]                     # [C, K, win, win]
    return (p * ok[None]).permute(1, 0, 2, 3)


def _boxes_from_targets(t, b, stride, W):
    """One sample's GT boxes in pixels, [n, 4] xyxy, plus cls / attrs."""
    m = t["mask"][b]
    ind = t["ind"][b][m]
    ci, cj = ind // W, ind % W
    cx = (cj.float() + t["off"][b][m][:, 0]) * stride; cy = (ci.float() + t["off"][b][m][:, 1]) * stride
    w = torch.exp(t["wh"][b][m][:, 0]) * stride; h = torch.exp(t["wh"][b][m][:, 1]) * stride
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), t["cls"][b][m], t["attrs"][b][m]


class _Flat:
    """Duck-typed view for M.decode's flat path."""
    mode = "flat"

    def __init__(self, m):
        self.stride = m.stride; self.n_cls = m.n_cls


# ----------------------------------------------------------------------------------------------------------------
# 1. scan-state

class RowScan(nn.Module):
    """Bidirectional GRU along the rows of a feature map (each row of cells is one sequence, left to right and
    back), projected back into the map. Zero-initialised output: the trunk starts undisturbed."""

    def __init__(self, c, h=96):
        super().__init__()
        self.gru = nn.GRU(c, h, batch_first=True, bidirectional=True)
        self.out = nn.Conv2d(2 * h, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        s = x.permute(0, 2, 3, 1).reshape(B * H, W, C).float()
        y, _ = self.gru(s)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2).to(x.dtype)
        return x + self.out(y)


class ScanState(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 224), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, gru=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths)
        self.scan4 = RowScan(widths[3], gru)                                  # stride 16
        self.scan5 = RowScan(widths[4], gru)                                  # stride 32
        self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        p2, p3, p4, p5 = self.enc(x)
        f = self.fpn((p2, p3, self.scan4(p4), self.scan5(p5)))
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
# 2. fovea

class Expert(nn.Module):
    """The second look: a 32x32 grayscale patch of the page -> class + attributes + a box correction."""

    def __init__(self, n_cls, c=(16, 32, 64, 96)):
        super().__init__()
        c1, c2, c3, c4 = c
        self.f = nn.Sequential(ConvBNAct(1, c1, 3, s=1), ConvBNAct(c1, c2, 3, s=2), Basic(c2),
                               ConvBNAct(c2, c3, 3, s=2), Basic(c3), ConvBNAct(c3, c4, 3, s=2), Basic(c4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(c4, n_cls + 1)
        self.attr = nn.Linear(c4, N_ATTR)
        self.box = nn.Linear(c4, 4)
        nn.init.zeros_(self.box.weight); nn.init.zeros_(self.box.bias)
        nn.init.constant_(self.cls.bias, 0.0); self.cls.bias.data[n_cls] = 2.0

    def forward(self, patch):
        v = self.pool(self.f(patch)).flatten(1)
        return self.cls(v), self.attr(v), self.box(v)


class Fovea(Base):
    CROP = 32
    WIN = 2.5                    # the patch spans 2.5x the candidate's longer side

    def __init__(self, n_cls, widths=(24, 48, 96, 160, 224), depths=(1, 2, 2, 1), fpn=64, hd=64, n_neg=160, k_infer=1500):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.obj = head(fpn, hd, 1, dilations=(1, 2), prior=0.01)
        self.reg = head(fpn, hd, 4)
        self.expert = Expert(n_cls)
        self.n_neg, self.k_infer = n_neg, k_infer

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"x": x, "obj": self.obj(f), "reg": self.reg(f)}

    def _crops(self, x, reg, b_idx, cy, cx):
        """roi_align the INPUT image at each candidate (flat index lists), with a window proportional to the
        candidate's own predicted size -- every glyph reaches the expert at the same apparent scale."""
        from torchvision.ops import roi_align
        r = reg[b_idx, :, cy, cx].float()                                     # [n, 4]: log w, log h, x off, y off
        w = torch.exp(r[:, 0].clamp(-2, 6)) * self.stride; h = torch.exp(r[:, 1].clamp(-2, 6)) * self.stride
        side = (torch.maximum(w, h) * self.WIN).clamp(min=8.0)
        px = (cx.float() + r[:, 2]) * self.stride; py = (cy.float() + r[:, 3]) * self.stride
        boxes = torch.stack([b_idx.float(), px - side / 2, py - side / 2, px + side / 2, py + side / 2], 1)
        return roi_align(x.float(), boxes.detach(), (self.CROP, self.CROP), spatial_scale=1.0, sampling_ratio=2, aligned=True), r, side

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["obj"].shape
        losses = {"obj": focal(out["obj"], t["hm"].amax(1, keepdim=True))}
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        with torch.no_grad():
            obj = torch.sigmoid(out["obj"].detach().float())[:, 0]
            pooled = F.max_pool2d(obj[:, None], 3, 1, 1)[:, 0]
            peaks = (obj * (pooled == obj)).reshape(B, -1)
            gt_cells = torch.zeros(B, H * W, dtype=torch.bool, device=obj.device).scatter_(1, t["ind"], t["mask"])
            neg_s, neg_i = peaks.masked_fill(gt_cells, 0.0).topk(self.n_neg, 1)
            neg_m = neg_s > 0.02
        ind = torch.cat([t["ind"], neg_i], 1); mask = torch.cat([t["mask"], neg_m], 1)
        b_idx = torch.arange(B, device=ind.device)[:, None].expand_as(ind).reshape(-1)
        cy, cx = (ind // W).reshape(-1), (ind % W).reshape(-1)
        patch, r, side = self._crops(out["x"], out["reg"], b_idx, cy, cx)
        cls_logit, attr, box = self.expert(patch)
        K = t["ind"].shape[1]
        target = torch.cat([t["cls"], torch.full_like(neg_i, self.n_cls)], 1).reshape(-1)
        m = mask.reshape(-1)
        losses["cls"] = F.cross_entropy(cls_logit[m].float(), target[m])
        mg = t["mask"].reshape(-1)
        if mg.any():
            dense = r.reshape(B, -1, 4)[:, :K].reshape(-1, 4).detach()
            res = torch.cat([t["off"].reshape(-1, 2) - dense[:, 2:], t["wh"].reshape(-1, 2) - dense[:, :2]], -1)
            losses["ref"] = F.l1_loss(box.reshape(B, -1, 4)[:, :K].reshape(-1, 4)[mg].float(), res[mg])
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
            s, idx = _peaks(obj[b], self.k_infer, thresh * 0.5)
            if len(idx) == 0:
                z = torch.zeros(0, device=obj.device)
                res.append(_dets(torch.zeros(0, 4, device=obj.device), z, z.long(), torch.zeros(0, N_ATTR, device=obj.device), z.long(), z.long(), 0)); continue
            cy, cx = idx // W, idx % W
            bi = torch.full_like(cy, b)
            patch, r, side = self._crops(out["x"], out["reg"], bi, cy, cx)
            cls_logit, attr, box = self.expert(patch)
            p = torch.softmax(cls_logit.float(), -1)
            pc, cls = p[:, :self.n_cls].max(-1)
            score = s * pc
            rr = r.float() + box.float()[:, [2, 3, 0, 1]]
            w = torch.exp(rr[:, 0]) * self.stride; h = torch.exp(rr[:, 1]) * self.stride
            x = (cx.float() + rr[:, 2]) * self.stride; y = (cy.float() + rr[:, 3]) * self.stride
            boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
            keep = score > thresh
            res.append(_dets(boxes[keep], score[keep], cls[keep], attr.float()[keep], cy[keep], cx[keep], nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 3. dyn-proposals

class DynInteract(nn.Module):
    """Sparse R-CNN's instance interaction: the proposal's feature vector generates the 1x1 filters that read its
    own RoI, so every proposal filters its region differently."""

    def __init__(self, d=64, k=16, roi=7):
        super().__init__()
        self.d, self.k = d, k
        self.gen = nn.Linear(d, 2 * d * k)
        self.n1 = nn.LayerNorm(k); self.n2 = nn.LayerNorm(d)
        self.out = nn.Linear(d * roi * roi, d); self.n3 = nn.LayerNorm(d)

    def forward(self, feat, roi):
        """feat [N, d]; roi [N, d, R, R] -> [N, d]."""
        N, d, R, _ = roi.shape
        p = self.gen(feat).reshape(N, 2, d, self.k)
        x = roi.reshape(N, d, R * R).permute(0, 2, 1)                       # [N, RR, d]
        x = F.relu(self.n1(torch.bmm(x, p[:, 0])))                          # [N, RR, k]
        x = F.relu(self.n2(torch.bmm(x, p[:, 1].transpose(1, 2))))          # [N, RR, d]
        return self.n3(self.out(x.reshape(N, -1)))


class DynProposals(Base):
    def __init__(self, n_cls, n_prop=400, d=64, roi=7, stages=2, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.proj = nn.Conv2d(fpn, d, 1)
        self.n_prop, self.d, self.roi, self.stages = n_prop, d, roi, stages
        self.prop_box = nn.Parameter(torch.tensor([[0.5, 0.5, 0.06, 0.06]]).repeat(n_prop, 1) + torch.randn(n_prop, 4) * 0.08)
        self.prop_feat = nn.Parameter(torch.randn(n_prop, d) * 0.02)
        self.inter = DynInteract(d, roi=roi)
        self.self_attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.n_sa = nn.LayerNorm(d); self.n_ffn = nn.LayerNorm(d)
        self.cls = nn.Linear(d, n_cls + 1); self.box = nn.Linear(d, 4); self.attr = nn.Linear(d, N_ATTR)
        nn.init.zeros_(self.box.weight); nn.init.zeros_(self.box.bias)
        nn.init.constant_(self.cls.bias, 0.0); self.cls.bias.data[n_cls] = 2.0

    def forward(self, x):
        from torchvision.ops import roi_align
        f = self.proj(self.fpn(self.enc(x))).float()
        B, d, H, W = f.shape
        Hp, Wp = H * self.stride, W * self.stride
        boxes = self.prop_box[None].expand(B, -1, -1).clone()
        cxcy = boxes[..., :2] * torch.tensor([Wp, Hp], device=x.device)
        wh = boxes[..., 2:].abs().clamp(min=0.004) * torch.tensor([Wp, Hp], device=x.device)
        cur = torch.cat([cxcy - wh / 2, cxcy + wh / 2], -1)
        feat = self.prop_feat[None].expand(B, -1, -1)
        outs = []
        bidx = torch.arange(B, device=x.device)[:, None].expand(B, self.n_prop).reshape(-1, 1).float()
        for s in range(self.stages):
            rois = torch.cat([bidx, cur.reshape(-1, 4)], 1)
            r = roi_align(f, rois.detach(), (self.roi, self.roi), spatial_scale=1.0 / self.stride, sampling_ratio=2, aligned=True)
            feat = self.inter(feat.reshape(-1, self.d), r).reshape(B, self.n_prop, self.d)
            feat = self.n_sa(feat + self.self_attn(feat, feat, feat, need_weights=False)[0])
            feat = self.n_ffn(feat + self.ffn(feat))
            delta = self.box(feat)
            cw = (cur[..., 2] - cur[..., 0]).clamp(min=1.0); ch = (cur[..., 3] - cur[..., 1]).clamp(min=1.0)
            ccx = (cur[..., 0] + cur[..., 2]) / 2 + delta[..., 0] * cw
            ccy = (cur[..., 1] + cur[..., 3]) / 2 + delta[..., 1] * ch
            nw = cw * torch.exp(delta[..., 2].clamp(-2, 2)); nh = ch * torch.exp(delta[..., 3].clamp(-2, 2))
            cur = torch.stack([ccx - nw / 2, ccy - nh / 2, ccx + nw / 2, ccy + nh / 2], -1)
            outs.append({"cls": self.cls(feat), "boxes": cur, "attr": self.attr(feat)})
        return {"stages": outs, "Wcells": W, **outs[-1]}

    def loss(self, out, t, attr_cls_mask):
        from scipy.optimize import linear_sum_assignment
        B = out["boxes"].shape[0]
        dev = out["boxes"].device
        Wc = out["Wcells"]
        losses = {}; parts = {}
        for si, o in enumerate(out["stages"]):
            w = 1.0 if si == len(out["stages"]) - 1 else 0.5
            cls_all, box_all, attr_all = o["cls"].float(), o["boxes"].float(), o["attr"]
            lp = torch.log_softmax(cls_all, -1)
            tgt_cls = torch.full(cls_all.shape[:2], self.n_cls, dtype=torch.long, device=dev)
            pairs = []
            for b in range(B):
                gb, gc, ga = _boxes_from_targets(t, b, self.stride, Wc)
                if len(gb) == 0:
                    continue
                with torch.no_grad():
                    cost = (-lp[b][:, gc] - _giou(box_all[b][:, None, :].expand(-1, len(gb), -1).reshape(-1, 4),
                                                  gb[None].expand(self.n_prop, -1, -1).reshape(-1, 4)).reshape(self.n_prop, -1)
                            + 0.02 * torch.cdist(box_all[b], gb, p=1))
                    pi, gi = linear_sum_assignment(cost.cpu().numpy())
                pi = torch.as_tensor(pi, device=dev); gi = torch.as_tensor(gi, device=dev)
                tgt_cls[b, pi] = gc[gi]
                pairs.append((b, pi, gi, gb, gc, ga))
            wgt = torch.ones(self.n_cls + 1, device=dev); wgt[self.n_cls] = 0.1
            losses[f"cls{si}"] = w * F.cross_entropy(cls_all.reshape(-1, self.n_cls + 1), tgt_cls.reshape(-1), weight=wgt)
            if pairs:
                pb = torch.cat([box_all[b][pi] for b, pi, gi, gb, gc, ga in pairs])
                tb = torch.cat([gb[gi] for b, pi, gi, gb, gc, ga in pairs])
                sc = torch.cat([(gb[gi][:, 2:] - gb[gi][:, :2]).clamp(min=1.0).repeat(1, 2) for b, pi, gi, gb, gc, ga in pairs])
                losses[f"box{si}"] = w * (1 - _giou(pb, tb)).mean()
                losses[f"l1{si}"] = w * 0.5 * (F.l1_loss(pb, tb, reduction="none") / sc).mean()
                if si == len(out["stages"]) - 1:
                    pa = torch.cat([attr_all[b][pi] for b, pi, gi, gb, gc, ga in pairs])
                    gc_ = torch.cat([gc[gi] for b, pi, gi, gb, gc, ga in pairs])
                    ga_ = torch.cat([ga[gi] for b, pi, gi, gb, gc, ga in pairs])
                    n = len(pa)
                    tt = {"ind": torch.arange(n, device=dev)[None], "mask": torch.ones(1, n, dtype=torch.bool, device=dev),
                          "cls": gc_[None], "attrs": ga_[None]}
                    al, parts = attr_loss(pa.t()[None, :, :, None], tt, attr_cls_mask)
                    losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.0):
        B = out["boxes"].shape[0]
        res = []
        for b in range(B):
            p = torch.softmax(out["cls"][b].float(), -1)
            pc, cls = p[:, :self.n_cls].max(-1)
            keep = pc > thresh
            cell = torch.zeros(int(keep.sum()), dtype=torch.long, device=p.device)
            res.append(_dets(out["boxes"][b].float()[keep], pc[keep], cls[keep], out["attr"][b].float()[keep], cell, cell, nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 4. ink-proto

class InkProto(Base):
    P = 16                       # prototype maps
    WIN = 12                     # mask window radius in cells (25 x 25 cells = 100 px at stride 4)
    N_MASK = 48                  # instances supervised per sample

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.proto = nn.Sequential(ConvBNAct(fpn, 48, 3), ConvBNAct(48, 48, 3), nn.Conv2d(48, self.P, 1))
        self.coef = head(fpn, 48, self.P)

    def forward(self, x):
        f = self.fpn(self.enc(x))
        out = {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f), "proto": self.proto(f), "coef": self.coef(f)}
        if self.training:
            # the ink itself is the mask target: darkness pooled to the readout's stride
            out["ink"] = F.max_pool2d(1.0 - x.float(), self.stride)
        return out

    def loss(self, out, t, attr_cls_mask):
        losses = {"hm": focal(out["hm"], t["hm"])}
        reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        B, _, H, W = out["hm"].shape
        r = self.WIN
        d = torch.arange(-r, r + 1, device=out["hm"].device)
        mloss = out["hm"].new_zeros((), dtype=torch.float32); nm = 0
        for b in range(B):
            m = t["mask"][b]
            n = int(m.sum())
            if n == 0:
                continue
            sel = torch.randperm(n, device=m.device)[:self.N_MASK]
            ind = t["ind"][b][m][sel]
            ci, cj = ind // W, ind % W
            proto = _patches(out["proto"][b].float(), ci, cj, r)                       # [k, P, win, win]
            coef = gather_at(out["coef"][b:b + 1], ind[None])[0]                        # [k, P]
            mask = (proto * coef[:, :, None, None]).sum(1)                              # [k, win, win] logits
            ink = _patches(out["ink"][b], ci, cj, r)[:, 0]                              # [k, win, win]
            # the labelled box, in the window's frame
            off = t["off"][b][m][sel]; wh = torch.exp(t["wh"][b][m][sel])
            gy = d[None, :, None].float(); gx = d[None, None, :].float()
            inside = ((gx - off[:, None, None, 0]).abs() <= wh[:, None, None, 0] / 2) & ((gy - off[:, None, None, 1]).abs() <= wh[:, None, None, 1] / 2)
            target = ((ink > 0.35) & inside).float()
            mloss = mloss + F.binary_cross_entropy_with_logits(mask.float(), target, reduction="sum") / target[0].numel()
            nm += len(sel)
        losses["mask"] = 0.5 * mloss / max(1, nm) * 16
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7, mask_min=8):
        dets = M.decode(_Flat(self), {k: out[k] for k in ("hm", "reg", "attr")}, K, thresh, nms_iou)
        r = self.WIN
        d = torch.arange(-r, r + 1, device=out["hm"].device).float()
        for b, det in enumerate(dets):
            if len(det["boxes"]) == 0:
                continue
            ci, cj = det["cell"][:, 0], det["cell"][:, 1]
            proto = _patches(out["proto"][b].float(), ci, cj, r)
            W = out["hm"].shape[-1]
            coef = gather_at(out["coef"][b:b + 1], (ci * W + cj)[None])[0]
            mask = torch.sigmoid((proto * coef[:, :, None, None]).sum(1)) > 0.5                  # [n, win, win]
            # tight extent of the mask, in pixels, around the cell centre
            any_y = mask.any(2); any_x = mask.any(1)
            has = mask.flatten(1).sum(1) >= mask_min
            idx = torch.arange(mask.shape[1], device=mask.device).float()
            y0 = torch.where(any_y, idx[None], torch.full_like(idx[None], 1e4)).min(1).values
            y1 = torch.where(any_y, idx[None], torch.full_like(idx[None], -1e4)).max(1).values
            x0 = torch.where(any_x, idx[None], torch.full_like(idx[None], 1e4)).min(1).values
            x1 = torch.where(any_x, idx[None], torch.full_like(idx[None], -1e4)).max(1).values
            cxp = (cj.float() + 0.5) * self.stride; cyp = (ci.float() + 0.5) * self.stride
            mb = torch.stack([cxp + (x0 - r) * self.stride, cyp + (y0 - r) * self.stride,
                              cxp + (x1 + 1 - r) * self.stride, cyp + (y1 + 1 - r) * self.stride], 1)
            # a mask that touches the window edge is clipped: keep the regressed box there
            edge = (x0 <= 0) | (y0 <= 0) | (x1 >= 2 * r) | (y1 >= 2 * r)
            use = has & ~edge
            det["boxes"] = torch.where(use[:, None], mb, det["boxes"])
        return dets


# ----------------------------------------------------------------------------------------------------------------
# 5. hires-out

class HiresOut(Base):
    stride = 2

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 240), depths=(1, 2, 3, 2), fpn=64, hd=48, attr_hd=48, up=40):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.up = nn.Sequential(nn.Conv2d(fpn, up * 4, 1), nn.PixelShuffle(2), nn.BatchNorm2d(up), nn.SiLU(inplace=True), ConvBNAct(up, up, 3))
        self.hm = head(up, hd, n_cls, prior=0.01); self.reg = head(up, hd, 4)
        self.attr = head(up, attr_hd, N_ATTR, dilations=(1, 2, 4, 8, 16))

    def forward(self, x):
        f = self.up(self.fpn(self.enc(x)))
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
# 6. fcos-dense

class FcosDense(Base):
    RADIUS = 1.5                 # centre sampling: cells within this many cells of the centre are positives

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.cls = head(fpn, hd, n_cls, prior=0.01)
        self.ltrb = head(fpn, hd, 4)
        self.ctr = head(fpn, hd, 1)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"cls": self.cls(f), "ltrb": self.ltrb(f), "ctr": self.ctr(f), "attr": self.attr(f)}

    def targets(self, t, H, W):
        """Per-cell assignment: the smallest box whose centre region contains the cell. Returns cls [B,H,W] (n_cls
        = background), ltrb [B,4,H,W] (log distances in cells), ctr [B,H,W], pos [B,H,W]."""
        B, K = t["ind"].shape
        dev = t["ind"].device
        cj0, ci0 = t["ind"] % W, t["ind"] // W
        cx = cj0.float() + t["off"][..., 0]; cy = ci0.float() + t["off"][..., 1]
        w = torch.exp(t["wh"][..., 0]); h = torch.exp(t["wh"][..., 1])
        area = (w * h).masked_fill(~t["mask"], float("inf"))
        r = int(max(4, self.RADIUS * 2 + 2))
        d = torch.arange(-r, r + 1, device=dev)
        win = len(d)
        gy = (ci0[:, :, None, None] + d[None, None, :, None]).expand(B, K, win, win)
        gx = (cj0[:, :, None, None] + d[None, None, None, :]).expand(B, K, win, win)
        px = gx.float() + 0.5; py = gy.float() + 0.5
        l = px - (cx[..., None, None] - w[..., None, None] / 2); tt_ = py - (cy[..., None, None] - h[..., None, None] / 2)
        rr = (cx[..., None, None] + w[..., None, None] / 2) - px; bb = (cy[..., None, None] + h[..., None, None] / 2) - py
        inside = (l > 0) & (tt_ > 0) & (rr > 0) & (bb > 0)
        near = ((px - cx[..., None, None]).abs() <= self.RADIUS) & ((py - cy[..., None, None]).abs() <= self.RADIUS)
        ok = inside & near & (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W) & t["mask"][..., None, None]
        bi = torch.arange(B, device=dev)[:, None, None, None].expand_as(gy)
        flat = (bi * H + gy.clamp(0, H - 1)) * W + gx.clamp(0, W - 1)
        prio = torch.full((B * H * W,), float("inf"), device=dev)
        a4 = area[..., None, None].expand_as(gy.float())
        prio.scatter_reduce_(0, flat[ok], a4[ok], reduce="amin")
        own = ok & (a4 == prio[flat])
        cls = torch.full((B * H * W,), self.n_cls, dtype=torch.long, device=dev)
        cls[flat[own]] = t["cls"][..., None, None].expand_as(gy)[own]
        box = torch.zeros(B * H * W, 4, device=dev)
        for k, v in enumerate((l, tt_, rr, bb)):
            box[flat[own], k] = v[own].clamp(min=0.05)
        attrs = torch.zeros(B * H * W, 5, dtype=torch.long, device=dev)
        attrs[flat[own]] = t["attrs"][:, :, None, None, :].expand(*gy.shape, 5)[own]
        cls = cls.reshape(B, H, W); box = box.reshape(B, H, W, 4).permute(0, 3, 1, 2)
        pos = cls.ne(self.n_cls)
        lr = box[:, [0, 2]]; tb = box[:, [1, 3]]
        ctr = torch.sqrt((lr.min(1).values / lr.max(1).values.clamp(min=1e-3)) * (tb.min(1).values / tb.max(1).values.clamp(min=1e-3))).clamp(0, 1)
        return cls, box, ctr * pos.float(), pos, attrs.reshape(B, H, W, 5)

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["cls"].shape
        cls_t, box_t, ctr_t, pos, attrs_t = self.targets(t, H, W)
        hm = torch.zeros(B, self.n_cls, H, W, device=out["cls"].device)
        hm.scatter_(1, cls_t.clamp(max=self.n_cls - 1)[:, None], pos[:, None].float())
        losses = {"cls": focal(out["cls"], hm)}
        n = pos.sum().clamp(min=1)
        pl = out["ltrb"].float().permute(0, 2, 3, 1)[pos]
        gl = box_t.permute(0, 2, 3, 1)[pos]
        if len(pl):
            pw = torch.exp(pl.clamp(-4, 6))
            # IoU loss on the ltrb parameterisation (FCOS's own)
            iw = torch.minimum(pw[:, 0], gl[:, 0]) + torch.minimum(pw[:, 2], gl[:, 2])
            ih = torch.minimum(pw[:, 1], gl[:, 1]) + torch.minimum(pw[:, 3], gl[:, 3])
            inter = iw.clamp(min=0) * ih.clamp(min=0)
            ap = (pw[:, 0] + pw[:, 2]) * (pw[:, 1] + pw[:, 3]); ag = (gl[:, 0] + gl[:, 2]) * (gl[:, 1] + gl[:, 3])
            losses["iou"] = -(torch.log((inter + 1.0) / (ap + ag - inter + 1.0))).mean()
            w_ctr = ctr_t[pos]
            losses["ctr"] = F.binary_cross_entropy_with_logits(out["ctr"].float()[:, 0][pos], w_ctr)
        # attributes at the positive cells (the object's own attribute ids)
        ind = pos.reshape(B, -1).float().argsort(dim=1, descending=True)
        npos = pos.reshape(B, -1).sum(1)
        Kk = int(npos.max().clamp(min=1))
        ind = ind[:, :Kk]
        tt = {"ind": ind, "mask": torch.arange(Kk, device=ind.device)[None] < npos[:, None],
              "cls": cls_t.reshape(B, -1).gather(1, ind).clamp(max=self.n_cls - 1),
              "attrs": attrs_t.reshape(B, -1, 5).gather(1, ind[..., None].expand(-1, -1, 5))}
        al, parts = attr_loss(out["attr"], tt, attr_cls_mask)
        losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.6):
        B, C, H, W = out["cls"].shape
        p = torch.sigmoid(out["cls"].float()) * torch.sigmoid(out["ctr"].float())
        res = []
        for b in range(B):
            scores, idx = p[b].reshape(-1).topk(min(K, C * H * W))
            keep = scores > thresh
            scores, idx = scores[keep], idx[keep]
            ch = idx // (H * W); cell = idx % (H * W); cy = cell // W; cx = cell % W
            d = torch.exp(out["ltrb"][b].float().reshape(4, -1)[:, cell].clamp(-4, 6)) * self.stride
            px = (cx.float() + 0.5) * self.stride; py = (cy.float() + 0.5) * self.stride
            boxes = torch.stack([px - d[0], py - d[1], px + d[2], py + d[3]], 1)
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, cell].t()
            res.append(_dets(boxes, scores, ch, attrs, cy, cx, nms_iou))
        return res


CONFIGS4 = {
    "scan-state": ScanState,
    "fovea": Fovea,
    "dyn-proposals": DynProposals,
    "ink-proto": InkProto,
    "hires-out": HiresOut,
    "fcos-dense": FcosDense,
}


def build4(name, classes, **override):
    return CONFIGS4[name](len(classes), **override)
