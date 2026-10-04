"""Round 5 of the 2M-parameter search: six further axes untouched by rounds 1-4.

  corners        CornerNet: an object is a top-left and a bottom-right corner, each a per-class keypoint heatmap
                 with a sub-cell offset, paired by a learnt 1-d embedding (pull together the two corners of one
                 object, push apart corners of different objects). No centre, no width / height, no anchors: the
                 box is exactly where the two corners land.
  retina-anchors RetinaNet: nine anchor boxes per cell over the four FPN levels (three at stride 4), positives by
                 IoU >= 0.5, the box as a delta from its anchor. The classic anchor-based readout, never tried here
                 (y8p2 was anchor-free); a glyph the size and shape of an anchor is a matching problem, not a
                 regression from one cell.
  page-moe       conditional computation: three expert readout heads, one chosen per page by a softmax gate on the
                 page's pooled trunk features (engraved / handwritten / dense / sparse are the gate's to discover).
                 Every earlier design applied the same head to every page.
  nat-head       neighbourhood attention as the READOUT: each stride-4 cell attends over its 7x7 neighbourhood
                 before classification -- attention where the decision is made, not in the trunk (swin-trunk) and
                 not between candidates (peak-relational).
  o2o-dense      one-to-one dense assignment (POTO / DeFCN): each object trains exactly one cell, chosen by
                 Hungarian matching on the model's own cost, and there is NO NMS at decode -- the network is trained
                 to suppress its own duplicates. Precision by construction rather than by post-processing.
  recon-aux      the flat model with an auxiliary decoder that reconstructs the PAGE from the stride-4 features
                 (an autoencoding side task): every glyph pixel is supervision, not just the labelled boxes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR
from . import models as M
from .arch2 import Base, _box_l1, _dets, _peaks
from .arch4 import FcosDense, _Flat, _giou
from .arch5 import _flat_loss
from .models import Basic, ConvBNAct, DW, DWDown, Encoder, FPN, attr_loss, focal, gather_at, head


# ----------------------------------------------------------------------------------------------------------------
# 1. corners

class Corners(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 240), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, topk=40):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.tl = head(fpn, hd, n_cls, prior=0.01); self.br = head(fpn, hd, n_cls, prior=0.01)
        self.geo = head(fpn, hd, 6)                                   # tl off (2), br off (2), tl emb, br emb
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.topk = topk

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"tl": self.tl(f), "br": self.br(f), "geo": self.geo(f), "attr": self.attr(f)}

    def _corner_targets(self, t, H, W):
        """GT corners (cells): tl / br cell indices [B, K], sub-cell offsets [B, K, 2] each, and heatmaps."""
        B, K = t["ind"].shape
        dev = t["ind"].device
        cx = (t["ind"] % W).float() + t["off"][..., 0]; cy = (t["ind"] // W).float() + t["off"][..., 1]
        w = torch.exp(t["wh"][..., 0]); h = torch.exp(t["wh"][..., 1])
        x0 = (cx - w / 2).clamp(0, W - 1e-3); y0 = (cy - h / 2).clamp(0, H - 1e-3)
        x1 = (cx + w / 2).clamp(0, W - 1e-3); y1 = (cy + h / 2).clamp(0, H - 1e-3)
        tl_i = y0.floor().long() * W + x0.floor().long(); br_i = y1.floor().long() * W + x1.floor().long()
        tl_o = torch.stack([x0 - x0.floor(), y0 - y0.floor()], -1); br_o = torch.stack([x1 - x1.floor(), y1 - y1.floor()], -1)
        # heatmaps: the same renderer as the centre targets, on boxes translated so their "centre" is the corner
        def hm_for(px, py):
            bx = []
            for b in range(B):
                m = t["mask"][b]
                n = int(m.sum())
                if n == 0:
                    bx.append(torch.zeros(0, 10, device=dev)); continue
                ww = w[b][m] * self.stride; hh = h[b][m] * self.stride
                c = torch.stack([t["cls"][b][m].float(), px[b][m] * self.stride - ww / 2, py[b][m] * self.stride - hh / 2,
                                 px[b][m] * self.stride + ww / 2, py[b][m] * self.stride + hh / 2], 1)
                bx.append(torch.cat([c, t["attrs"][b][m].float()], 1))
            from .data import render_targets
            return render_targets(bx, H * self.stride, W * self.stride, self.stride, self.n_cls, dev)["hm"]
        return tl_i, br_i, tl_o, br_o, hm_for(x0, y0), hm_for(x1, y1)

    def loss(self, out, t, attr_cls_mask):
        B, C, H, W = out["tl"].shape
        tl_i, br_i, tl_o, br_o, tl_hm, br_hm = self._corner_targets(t, H, W)
        losses = {"tl": focal(out["tl"], tl_hm), "br": focal(out["br"], br_hm)}
        g_tl = gather_at(out["geo"], tl_i); g_br = gather_at(out["geo"], br_i)            # [B, K, 6]
        m = t["mask"]; n = m.sum().clamp(min=1)
        losses["off"] = ((F.l1_loss(g_tl[..., 0:2], tl_o, reduction="none") + F.l1_loss(g_br[..., 2:4], br_o, reduction="none")).sum(-1) * m).sum() / n
        e_tl = g_tl[..., 4]; e_br = g_br[..., 5]
        mean = (e_tl + e_br) / 2
        pull = (((e_tl - mean) ** 2 + (e_br - mean) ** 2) * m).sum() / n
        push = 0.0
        for b in range(B):
            mb = mean[b][m[b]]
            if len(mb) > 1:
                d = (mb[:, None] - mb[None, :]).abs()
                push = push + (F.relu(1.0 - d).triu(1)).sum() / max(1, len(mb) * (len(mb) - 1) / 2)
        losses["pull"] = 0.1 * pull; losses["push"] = 0.1 * push / B
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        B, C, H, W = out["tl"].shape
        res = []
        for b in range(B):
            def top(hm):
                p = torch.sigmoid(hm.float()); pooled = F.max_pool2d(p[None], 3, 1, 1)[0]; p = p * (pooled == p)
                s, i = p.reshape(C, -1).topk(self.topk, 1)                                   # [C, k]
                return s, i
            s1, i1 = top(out["tl"][b]); s2, i2 = top(out["br"][b])
            geo = out["geo"][b].float().reshape(6, -1)
            x1 = (i1 % W).float() + geo[0][i1]; y1 = (i1 // W).float() + geo[1][i1]
            x2 = (i2 % W).float() + geo[2][i2]; y2 = (i2 // W).float() + geo[3][i2]
            e1 = geo[4][i1]; e2 = geo[5][i2]
            k = self.topk
            valid = (x2[:, None, :] > x1[:, :, None]) & (y2[:, None, :] > y1[:, :, None]) & ((e1[:, :, None] - e2[:, None, :]).abs() < 0.5)
            score = (s1[:, :, None] + s2[:, None, :]) / 2
            score = score.masked_fill(~valid, 0.0)
            sc = score.reshape(-1); keep = sc > thresh
            idx = keep.nonzero()[:, 0]
            if len(idx) > K:
                idx = idx[sc[idx].topk(K).indices]
            c = idx // (k * k); a = (idx % (k * k)) // k; d = idx % k
            boxes = torch.stack([x1[c, a], y1[c, a], x2[c, d], y2[c, d]], 1) * self.stride
            cxm = ((boxes[:, 0] + boxes[:, 2]) / 2 / self.stride).long().clamp(0, W - 1); cym = ((boxes[:, 1] + boxes[:, 3]) / 2 / self.stride).long().clamp(0, H - 1)
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, cym * W + cxm].t()
            res.append(_dets(boxes, sc[idx], c, attrs, cym, cxm, nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 2. retina-anchors

def _neg_focal_rows(lg):
    """Focal loss of rows treated as all-negative: sum over classes of 0.75 * p^2 * -log(1 - p)."""
    p = torch.sigmoid(lg.float())
    return (F.binary_cross_entropy_with_logits(lg.float(), torch.zeros_like(lg.float()), reduction="none") * p ** 2 * 0.75).sum()


def _neg_focal(lg, chunk=1 << 15):
    """The same over every anchor, in chunks under activation checkpointing."""
    from torch.utils.checkpoint import checkpoint
    total = lg.new_zeros((), dtype=torch.float32)
    for i in range(0, lg.shape[0], chunk):
        total = total + checkpoint(_neg_focal_rows, lg[i:i + chunk], use_reentrant=False)
    return total


class RetinaAnchors(Base):
    SCALES = (1.0, 1.6, 2.5)         # x the level's stride
    RATIOS = (0.5, 1.0, 2.0)

    def __init__(self, n_cls, widths=(24, 48, 96, 160, 224), depths=(1, 2, 2, 1), d=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths)
        self.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in self.enc.out_channels])
        self.smooth = nn.ModuleList([ConvBNAct(d, d, 3) for _ in range(4)])
        self.na = [3, 9, 9, 9]                                        # anchors per cell at strides 4, 8, 16, 32
        self.cls = head(d, hd, 9 * n_cls, prior=0.01)                 # shared by strides 8-32 (nine anchors)
        self.box = head(d, hd, 9 * 4)
        self.cls4 = head(d, hd, 3 * n_cls, prior=0.01)                # stride 4: three anchors (aspect only)
        self.box4 = head(d, hd, 3 * 4)
        self.attr = head(d, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))  # attributes at the stride-4 level
        self.strides = (4, 8, 16, 32)

    def anchors(self, level, H, W, device):
        s = self.strides[level]
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        cx = (xs.float() + 0.5) * s; cy = (ys.float() + 0.5) * s
        specs = [(sc, r) for sc in self.SCALES for r in self.RATIOS]
        if self.na[level] == 3:
            specs = [(1.0, r) for r in self.RATIOS]
        out = []
        for sc, r in specs:
            w = s * sc * 2 * math.sqrt(r); h = s * sc * 2 / math.sqrt(r)
            out.append(torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1))
        return torch.stack(out, 2).reshape(-1, 4)                     # [H*W*na, 4] (cell-major, anchor-minor)

    def forward(self, x):
        feats = self.enc(x)
        lats = [l(f) for l, f in zip(self.lat, feats)]
        p = lats[3]
        outs = [None] * 4
        outs[3] = self.smooth[3](p)
        for k in (2, 1, 0):
            p = F.interpolate(p, size=lats[k].shape[-2:], mode="nearest") + lats[k]
            outs[k] = self.smooth[k](p)
        res = {"levels": [], "attr": self.attr(outs[0])}
        for k, f in enumerate(outs):
            na = self.na[k]
            c = (self.cls4 if k == 0 else self.cls)(f); bx = (self.box4 if k == 0 else self.box)(f)
            B, _, H, W = c.shape
            c = c.reshape(B, na, self.n_cls, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, self.n_cls)
            bx = bx.reshape(B, na, 4, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 4)
            res["levels"].append({"cls": c, "box": bx, "hw": (H, W)})
        return res

    def _decode_boxes(self, anc, delta):
        aw = anc[:, 2] - anc[:, 0]; ah = anc[:, 3] - anc[:, 1]; ax = (anc[:, 0] + anc[:, 2]) / 2; ay = (anc[:, 1] + anc[:, 3]) / 2
        d = delta.float()
        cx = ax + d[..., 0] * aw; cy = ay + d[..., 1] * ah
        w = aw * torch.exp(d[..., 2].clamp(-3, 3)); h = ah * torch.exp(d[..., 3].clamp(-3, 3))
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

    def loss(self, out, t, attr_cls_mask):
        from .arch4 import _boxes_from_targets
        from .evaluate import iou_matrix  # noqa: F401  (numpy version; the torch one is inline below)
        dev = out["attr"].device
        H4, W4 = out["levels"][0]["hw"]
        ancs = [self.anchors(k, *lv["hw"], dev) for k, lv in enumerate(out["levels"])]
        anc = torch.cat(ancs, 0)                                                                 # [A, 4]
        cls_all = torch.cat([lv["cls"] for lv in out["levels"]], 1)                              # [B, A, n_cls]
        box_all = torch.cat([lv["box"] for lv in out["levels"]], 1)                              # [B, A, 4]
        B, A, _ = cls_all.shape
        losses = {}; n_pos = 0; cls_loss = cls_all.new_zeros((), dtype=torch.float32); box_loss = 0.0
        for b in range(B):
            gb, gc, ga = _boxes_from_targets(t, b, self.stride, W4)
            pos = None
            if len(gb):
                ix0 = torch.maximum(anc[:, None, 0], gb[None, :, 0]); iy0 = torch.maximum(anc[:, None, 1], gb[None, :, 1])
                ix1 = torch.minimum(anc[:, None, 2], gb[None, :, 2]); iy1 = torch.minimum(anc[:, None, 3], gb[None, :, 3])
                inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
                aa = (anc[:, 2] - anc[:, 0]) * (anc[:, 3] - anc[:, 1]); ga_ = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
                iou = inter / (aa[:, None] + ga_[None, :] - inter + 1e-6)                        # [A, G]
                best_iou, best_g = iou.max(1)
                pos = best_iou >= 0.5
                # every GT gets at least its best anchor
                top_a = iou.argmax(0); pos[top_a] = True; best_g[top_a] = torch.arange(len(gb), device=dev)
                ign = (best_iou >= 0.4) & ~pos
                n_pos += int(pos.sum())
                if pos.any():
                    pb = self._decode_boxes(anc[pos], box_all[b][pos]); tb = gb[best_g[pos]]
                    box_loss = box_loss + (1 - _giou(pb, tb)).sum()
                keep = ~ign
            else:
                keep = torch.ones(A, dtype=torch.bool, device=dev)
            # focal over ~390K anchors x 266 classes, computed as "every anchor negative" in chunks and then
            # corrected on the few positive and ignored rows: the dense form allocated four [A, n_cls] tensors per
            # sample (415 MB each) and ran the GPU out of memory under load.
            lg = cls_all[b]
            cls_loss = cls_loss + _neg_focal(lg)
            drop = (~keep).nonzero()[:, 0]
            if len(drop):
                cls_loss = cls_loss - _neg_focal_rows(lg[drop])
            if pos is not None and pos.any():
                pidx = pos.nonzero()[:, 0]
                lp = lg[pidx].float()
                tp = torch.zeros(len(pidx), self.n_cls, device=dev)
                tp[torch.arange(len(pidx), device=dev), gc[best_g[pos]]] = 1.0
                pr = torch.sigmoid(lp)
                bce = F.binary_cross_entropy_with_logits(lp, tp, reduction="none")
                ptv = torch.where(tp > 0, pr, 1 - pr)
                cls_loss = cls_loss + (bce * (1 - ptv) ** 2 * torch.where(tp > 0, 0.25, 0.75)).sum() - _neg_focal_rows(lp)
        n = max(1, n_pos)
        losses["cls"] = cls_loss / n; losses["box"] = box_loss / n if torch.is_tensor(box_loss) else torch.zeros((), device=dev)
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        parts["npos"] = n_pos
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.5):
        dev = out["attr"].device
        H4, W4 = out["levels"][0]["hw"]
        ancs = torch.cat([self.anchors(k, *lv["hw"], dev) for k, lv in enumerate(out["levels"])], 0)
        cls_all = torch.cat([lv["cls"] for lv in out["levels"]], 1); box_all = torch.cat([lv["box"] for lv in out["levels"]], 1)
        res = []
        for b in range(cls_all.shape[0]):
            p = torch.sigmoid(cls_all[b].float())
            s, idx = p.reshape(-1).topk(min(K * 3, p.numel()))
            keep = s > thresh; s, idx = s[keep], idx[keep]
            a = idx // self.n_cls; c = idx % self.n_cls
            boxes = self._decode_boxes(ancs[a], box_all[b][a])
            cxm = ((boxes[:, 0] + boxes[:, 2]) / 2 / self.stride).long().clamp(0, W4 - 1); cym = ((boxes[:, 1] + boxes[:, 3]) / 2 / self.stride).long().clamp(0, H4 - 1)
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, cym * W4 + cxm].t()
            d = _dets(boxes, s, c, attrs, cym, cxm, nms_iou)
            if len(d["boxes"]) > K:
                top = d["scores"].topk(K).indices
                d = {k: v[top] for k, v in d.items()}
            res.append(d)
        return res


# ----------------------------------------------------------------------------------------------------------------
# 3. page-moe

class PageMoE(Base):
    def __init__(self, n_cls, n_exp=3, widths=(24, 48, 96, 176, 240), depths=(1, 2, 3, 2), fpn=64, hd=40, attr_hd=40):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.gate = nn.Sequential(nn.Linear(widths[4], 64), nn.SiLU(), nn.Linear(64, n_exp))
        self.hm = nn.ModuleList([head(fpn, hd, n_cls, prior=0.01) for _ in range(n_exp)])
        self.reg = nn.ModuleList([head(fpn, hd, 4) for _ in range(n_exp)])
        self.attr = nn.ModuleList([head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8)) for _ in range(n_exp)])
        self.n_exp = n_exp

    def forward(self, x):
        p2, p3, p4, p5 = self.enc(x)
        f = self.fpn((p2, p3, p4, p5))
        g = torch.softmax(self.gate(p5.float().mean((2, 3))), -1)                            # [B, E]
        # every expert runs (they are small); the page's mixture weights combine their maps
        hm = sum(g[:, e, None, None, None] * self.hm[e](f).float() for e in range(self.n_exp))
        reg = sum(g[:, e, None, None, None] * self.reg[e](f).float() for e in range(self.n_exp))
        attr = sum(g[:, e, None, None, None] * self.attr[e](f).float() for e in range(self.n_exp))
        return {"hm": hm.to(f.dtype), "reg": reg.to(f.dtype), "attr": attr.to(f.dtype), "gate": g}

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = _flat_loss(self, out, t, attr_cls_mask)
        # keep the gate from collapsing to one expert: a mild load-balance term on the batch mean
        gm = out["gate"].mean(0)
        lb = 0.01 * self.n_exp * (gm * gm).sum()
        parts["lb"] = float(lb); aparts["gate_max"] = float(out["gate"].max(1).values.mean())
        return total + lb, parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 4. nat-head

class NeighbourAttn(nn.Module):
    """Each cell attends over its (2r+1)^2 neighbourhood (unfolded keys / values), + MLP."""

    def __init__(self, d, r=3, heads=4, rows=32):
        super().__init__()
        self.r, self.heads, self.d = r, heads, d
        self.q = nn.Conv2d(d, d, 1); self.kv = nn.Conv2d(d, 2 * d, 1); self.proj = nn.Conv2d(d, d, 1)
        self.n1 = nn.GroupNorm(1, d); self.n2 = nn.GroupNorm(1, d)
        self.mlp = nn.Sequential(nn.Conv2d(d, 2 * d, 1), nn.GELU(), nn.Conv2d(2 * d, d, 1))
        self.pos = nn.Parameter(torch.zeros(heads, (2 * r + 1) ** 2))
        self.rows = rows

    def _attend(self, q, kvm, H, W):
        """q [B, C, H, W] queries, kvm [B, 2C, H + 2r, W] keys / values with an r-row halo -> [B, C, H, W]."""
        B, C = q.shape[:2]
        r = self.r; n = (2 * r + 1) ** 2; hd = C // self.heads
        qq = q.reshape(B, self.heads, hd, H * W)
        kv = F.unfold(kvm, 2 * r + 1, padding=(0, r)).reshape(B, 2, self.heads, hd, n, H * W)
        att = (qq[:, :, :, None, :] * kv[:, 0]).sum(2) / math.sqrt(hd) + self.pos[None, :, :, None]
        att = torch.softmax(att, 2)
        return (att[:, :, None] * kv[:, 1]).sum(3).reshape(B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        r = self.r
        h = self.n1(x)
        q = self.q(h); kvm = F.pad(self.kv(h), (0, 0, r, r))
        outs = [self._attend(q[:, :, y0:y0 + self.rows], kvm[:, :, y0:y0 + self.rows + 2 * r], min(self.rows, H - y0), W)
                for y0 in range(0, H, self.rows)]
        x = x + self.proj(torch.cat(outs, 2))
        return x + self.mlp(self.n2(x))


class NatHead(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 224), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, r=3):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.nat = NeighbourAttn(fpn, r)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        f = self.fpn(self.enc(x))
        f = checkpoint(self.nat, f, use_reentrant=False) if self.training else self.nat(f)
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 5. o2o-dense

class O2ODense(FcosDense):
    RADIUS = 2.5

    def __init__(self, n_cls, **kw):
        super().__init__(n_cls, **kw)
        self.obj = self.ctr; del self.ctr

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"cls": self.cls(f), "ltrb": self.ltrb(f), "obj": self.obj(f), "attr": self.attr(f)}

    @torch.no_grad()
    def assign(self, out, t):
        """One cell per object: Hungarian matching between the objects and the cells within RADIUS of their
        centres, cost = (1 - p_cls^0.8 * IoU^0.2) as in POTO. The 3x3 max filter is what a one-to-one model uses
        instead of NMS: a duplicate next to the matched cell is a negative."""
        B, C, H, W = out["cls"].shape
        dev = out["cls"].device
        r = int(math.ceil(self.RADIUS)); d = torch.arange(-r, r + 1, device=dev); win = len(d)
        cls_t = torch.full((B, H, W), self.n_cls, dtype=torch.long, device=dev)
        box_t = torch.zeros(B, 4, H, W, device=dev); attrs_t = torch.zeros(B, H, W, 5, dtype=torch.long, device=dev)
        pd = torch.exp(out["ltrb"].float().clamp(-4, 6)); pc = torch.sigmoid(out["cls"].float()) * torch.sigmoid(out["obj"].float())
        for b in range(B):
            m = t["mask"][b]
            if not m.any():
                continue
            ind = t["ind"][b][m]; K = len(ind); ci0, cj0 = ind // W, ind % W
            cx = cj0.float() + t["off"][b][m][:, 0]; cy = ci0.float() + t["off"][b][m][:, 1]
            w = torch.exp(t["wh"][b][m][:, 0]); h = torch.exp(t["wh"][b][m][:, 1])
            gy = (ci0[:, None, None] + d[None, :, None]).expand(K, win, win); gx = (cj0[:, None, None] + d[None, None, :]).expand(K, win, win)
            px = gx.float() + 0.5; py = gy.float() + 0.5
            ok = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W) & ((px - cx[:, None, None]).abs() <= self.RADIUS) & ((py - cy[:, None, None]).abs() <= self.RADIUS)
            gyc, gxc = gy.clamp(0, H - 1), gx.clamp(0, W - 1)
            l_, t_, r_, b_ = [pd[b, k, gyc, gxc] for k in range(4)]
            x0 = px - l_; y0 = py - t_; x1 = px + r_; y1 = py + b_
            gx0 = (cx - w / 2)[:, None, None]; gy0 = (cy - h / 2)[:, None, None]; gx1 = (cx + w / 2)[:, None, None]; gy1 = (cy + h / 2)[:, None, None]
            inter = (torch.minimum(x1, gx1) - torch.maximum(x0, gx0)).clamp(min=0) * (torch.minimum(y1, gy1) - torch.maximum(y0, gy0)).clamp(min=0)
            iou = inter / ((x1 - x0) * (y1 - y0) + (w * h)[:, None, None] - inter + 1e-6)
            gcls = t["cls"][b][m]
            p_cls = pc[b, gcls[:, None, None].expand(K, win, win), gyc, gxc].clamp(1e-6, 1)
            cost = (1 - p_cls ** 0.8 * iou.clamp(min=1e-6) ** 0.2).masked_fill(~ok, 10.0)
            # one cell may be a candidate of several objects: greedy one-to-one over the (object, cell) pairs in
            # cost order -- the exact Hungarian over thousands of cells cost 2.4 s a step
            cells = (gyc * W + gxc).reshape(K, -1); flat_cost = cost.reshape(K, -1)
            order = flat_cost.reshape(-1).argsort()
            order = order[flat_cost.reshape(-1)[order] < 9.0].cpu().numpy()
            ko = (order // flat_cost.shape[1]); co = cells.reshape(-1).cpu().numpy()[order]
            used_k = set(); used_c = set(); ki = []; cell = []
            for k_, c_ in zip(ko.tolist(), co.tolist()):
                if k_ in used_k or c_ in used_c:
                    continue
                used_k.add(k_); used_c.add(c_); ki.append(k_); cell.append(c_)
                if len(used_k) == K:
                    break
            ki = torch.as_tensor(ki, device=dev, dtype=torch.long); cell = torch.as_tensor(cell, device=dev, dtype=torch.long)
            cyy, cxx = cell // W, cell % W
            cls_t[b, cyy, cxx] = gcls[ki]
            pxx = cxx.float() + 0.5; pyy = cyy.float() + 0.5
            box_t[b, 0, cyy, cxx] = (pxx - (cx - w / 2)[ki]).clamp(min=0.05); box_t[b, 1, cyy, cxx] = (pyy - (cy - h / 2)[ki]).clamp(min=0.05)
            box_t[b, 2, cyy, cxx] = ((cx + w / 2)[ki] - pxx).clamp(min=0.05); box_t[b, 3, cyy, cxx] = ((cy + h / 2)[ki] - pyy).clamp(min=0.05)
            attrs_t[b, cyy, cxx] = t["attrs"][b][m][ki]
        pos = cls_t.ne(self.n_cls)
        return pos, cls_t, box_t, attrs_t

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["cls"].shape
        pos, cls_t, box_t, attrs_t = self.assign(out, t)
        hm = torch.zeros(B, self.n_cls, H, W, device=out["cls"].device)
        hm.scatter_(1, cls_t.clamp(max=self.n_cls - 1)[:, None], pos[:, None].float())
        losses = {"cls": focal(out["cls"], hm)}
        pl = out["ltrb"].float().permute(0, 2, 3, 1)[pos]; gl = box_t.permute(0, 2, 3, 1)[pos]
        if len(pl):
            pw = torch.exp(pl.clamp(-4, 6))
            iw = torch.minimum(pw[:, 0], gl[:, 0]) + torch.minimum(pw[:, 2], gl[:, 2]); ih = torch.minimum(pw[:, 1], gl[:, 1]) + torch.minimum(pw[:, 3], gl[:, 3])
            inter = iw.clamp(min=0) * ih.clamp(min=0)
            ap = (pw[:, 0] + pw[:, 2]) * (pw[:, 1] + pw[:, 3]); ag = (gl[:, 0] + gl[:, 2]) * (gl[:, 1] + gl[:, 3])
            losses["iou"] = -(torch.log((inter + 1.0) / (ap + ag - inter + 1.0))).mean()
        losses["obj"] = F.binary_cross_entropy_with_logits(out["obj"].float()[:, 0], pos.float()) * 4.0
        ind = pos.reshape(B, -1).float().argsort(dim=1, descending=True)
        npos = pos.reshape(B, -1).sum(1); Kk = int(npos.max().clamp(min=1)); ind = ind[:, :Kk]
        tt = {"ind": ind, "mask": torch.arange(Kk, device=ind.device)[None] < npos[:, None],
              "cls": cls_t.reshape(B, -1).gather(1, ind).clamp(max=self.n_cls - 1),
              "attrs": attrs_t.reshape(B, -1, 5).gather(1, ind[..., None].expand(-1, -1, 5))}
        al, parts = attr_loss(out["attr"], tt, attr_cls_mask); losses["attr"] = al
        parts["npos"] = int(pos.sum())
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.0):
        # NO NMS: the model was trained one-to-one; only the 3x3 max filter POTO uses
        B, C, H, W = out["cls"].shape
        p = torch.sigmoid(out["cls"].float()) * torch.sigmoid(out["obj"].float())
        pooled = F.max_pool2d(p, 3, 1, 1); p = p * (pooled == p)
        res = []
        for b in range(B):
            scores, idx = p[b].reshape(-1).topk(min(K, C * H * W))
            keep = scores > thresh; scores, idx = scores[keep], idx[keep]
            ch = idx // (H * W); cell = idx % (H * W); cy = cell // W; cx = cell % W
            d = torch.exp(out["ltrb"][b].float().reshape(4, -1)[:, cell].clamp(-4, 6)) * self.stride
            px = (cx.float() + 0.5) * self.stride; py = (cy.float() + 0.5) * self.stride
            boxes = torch.stack([px - d[0], py - d[1], px + d[2], py + d[3]], 1)
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, cell].t()
            res.append(_dets(boxes, scores, ch, attrs, cy, cx, 0.0))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 6. recon-aux

class ReconAux(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 240), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, w_rec=1.0):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.dec = nn.Sequential(ConvBNAct(fpn, 48, 3), nn.Conv2d(48, 16, 1), nn.PixelShuffle(4))        # stride 4 -> the page, 1 channel
        self.w_rec = w_rec

    def forward(self, x):
        f = self.fpn(self.enc(x))
        out = {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}
        if self.training:
            out["rec"] = self.dec(f); out["x"] = x
        return out

    def loss(self, out, t, attr_cls_mask):
        total, parts, aparts = _flat_loss(self, out, t, attr_cls_mask)
        ink = 1.0 - out["x"].float()
        # ink pixels are rare: weight them so the decoder cannot win by painting paper
        w = 1.0 + 9.0 * (ink > 0.3).float()
        rec = self.w_rec * (F.binary_cross_entropy_with_logits(out["rec"].float(), ink, reduction="none") * w).mean()
        parts["rec"] = float(rec)
        return total + rec, parts, aparts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), {k: out[k] for k in ("hm", "reg", "attr")}, K, thresh, nms_iou)


CONFIGS6 = {"corners": Corners, "retina-anchors": RetinaAnchors, "page-moe": PageMoE, "nat-head": NatHead,
            "o2o-dense": O2ODense, "recon-aux": ReconAux}


def build6(name, classes, **override):
    return CONFIGS6[name](len(classes), **override)
