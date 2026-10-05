"""Round 4 of the 2M-parameter search: six axes none of rounds 1-3 touched.

Rounds 1-3 varied the trunk (no effect), the readout (large effect: relational second stage -> precision; dense
per-cell boxes -> the best recall / tightness balance among one-stage designs) and the box representation (masks
hurt, learned proposals did not converge). Six axes remain untouched: how a box is PARAMETERISED as a
distribution and how the score is calibrated (gfl-dense), sample points as the box (reppoints), how positives are
ASSIGNED (simota), what the model is given as INPUT (staffless-in), an attention trunk (swin-trunk), and sharing
one trunk across two SCALES (siam-scale).

  gfl-dense    Generalized Focal Loss: each box side is a distribution over 16 log-spaced distances (the box is
               its expectation), and the class score is trained towards the box's IoU (quality focal), so a
               detection's confidence IS its localisation quality -- the calibration the two-stage designs lacked.
  reppoints    nine sample points per object instead of a width and height: a deformable 3x3 whose offsets are
               predicted per cell samples the features where the glyph is, and the box is the points' extent.
  simota       the FCOS readout with YOLOX's SimOTA assignment: which cells are positives is decided per object by
               the model's own current cost (classification + IoU), dynamic-k -- not by a fixed centre radius.
  staffless-in three input channels: the page, the page with its long horizontal runs removed (a fixed morphological
               opening, 25 px), and those runs alone. The staff lines are the one structure every glyph is drawn
               across; handing the model a lines-free view is a question about the input, not the network.
  swin-trunk   the stride-16 / 32 stages as windowed self-attention blocks (8x8 windows, depthwise conv between
               blocks for cross-window mixing) under the conv stem: the first non-conv trunk in the search.
  siam-scale   the flat model's trunk run twice with SHARED weights, on the page and on the page at half size,
               the half-size features upsampled and gated into the full-size ones: a doubled receptive field and
               explicit scale context at zero extra parameters (round 2's recur spent compute on a second pass
               over the OUTPUT; this spends it on a second SCALE).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR
from . import models as M
from .arch2 import Base, _box_l1, _dets, _peaks
from .arch4 import FcosDense, _Flat, _giou
from .models import Basic, ConvBNAct, DW, DWDown, Encoder, FPN, attr_loss, focal, gather_at, head


def _flat_loss(self, out, t, attr_cls_mask):
    losses = {"hm": focal(out["hm"], t["hm"])}
    reg = gather_at(out["reg"], t["ind"]); losses["wh"], losses["off"] = _box_l1(reg, t)
    al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
    return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts


# ----------------------------------------------------------------------------------------------------------------
# 1. gfl-dense

class GflDense(FcosDense):
    BINS = 16
    D_MIN, D_MAX = 0.25, 160.0          # cells: a dot's half-width .. a system-wide bracket

    def __init__(self, n_cls, **kw):
        super().__init__(n_cls, **kw)
        fpn = self.ltrb[0][0].in_channels; hd = self.ltrb[0][0].out_channels
        self.ltrb = head(fpn, hd, 4 * self.BINS)
        del self.ctr
        self.register_buffer("logv", torch.linspace(math.log(self.D_MIN), math.log(self.D_MAX), self.BINS))

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"cls": self.cls(f), "ltrb": self.ltrb(f), "attr": self.attr(f)}

    def _expect(self, logits):
        """[..., 4*BINS] -> distances in cells [..., 4] (expectation of exp over the bins)."""
        p = torch.softmax(logits.float().reshape(*logits.shape[:-1], 4, self.BINS), -1)
        return torch.exp((p * self.logv).sum(-1))

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["cls"].shape
        cls_t, box_t, ctr_t, pos, attrs_t = self.targets(t, H, W)
        losses = {}
        n = pos.sum().clamp(min=1)
        lg = out["ltrb"].permute(0, 2, 3, 1)[pos]                                  # [n, 4*BINS]
        gl = box_t.permute(0, 2, 3, 1)[pos]                                        # [n, 4] cells
        if len(lg):
            # DFL: the target distance as a continuous bin index, cross-entropy to its two neighbours
            idx = (torch.log(gl.clamp(self.D_MIN, self.D_MAX)) - self.logv[0]) / (self.logv[1] - self.logv[0])
            lo = idx.floor().long().clamp(0, self.BINS - 2); hi = lo + 1; wl = (hi.float() - idx).clamp(0, 1); wh_ = 1 - wl
            lp = torch.log_softmax(lg.float().reshape(-1, 4, self.BINS), -1)
            losses["dfl"] = 0.25 * (-(lp.gather(-1, lo[..., None])[..., 0] * wl + lp.gather(-1, hi[..., None])[..., 0] * wh_)).mean()
            pd = self._expect(lg)
            iw = torch.minimum(pd[:, 0], gl[:, 0]) + torch.minimum(pd[:, 2], gl[:, 2])
            ih = torch.minimum(pd[:, 1], gl[:, 1]) + torch.minimum(pd[:, 3], gl[:, 3])
            inter = iw.clamp(min=0) * ih.clamp(min=0)
            ap = (pd[:, 0] + pd[:, 2]) * (pd[:, 1] + pd[:, 3]); ag = (gl[:, 0] + gl[:, 2]) * (gl[:, 1] + gl[:, 3])
            iou = inter / (ap + ag - inter + 1e-6)
            losses["iou"] = (1 - iou).mean()
            q = iou.detach()
        else:
            q = None
        # quality focal: the class score is trained towards the IoU of the box predicted at that cell
        tgt = torch.zeros_like(out["cls"], dtype=torch.float32)
        if q is not None:
            tgt.permute(0, 2, 3, 1)[pos] = F.one_hot(cls_t[pos].clamp(max=self.n_cls - 1), self.n_cls).float() * q[:, None]
        from torch.utils.checkpoint import checkpoint
        def _qfl(lg_, tg_):
            p_ = torch.sigmoid(lg_.float())
            return (F.binary_cross_entropy_with_logits(lg_.float(), tg_, reduction="none") * (tg_ - p_).abs() ** 2).sum()
        tot = out["cls"].new_zeros((), dtype=torch.float32)
        for b in range(B):
            tot = tot + checkpoint(_qfl, out["cls"][b], tgt[b], use_reentrant=False)
        losses["qfl"] = tot / n
        # attributes at the positive cells
        ind = pos.reshape(B, -1).float().argsort(dim=1, descending=True)
        npos = pos.reshape(B, -1).sum(1); Kk = int(npos.max().clamp(min=1)); ind = ind[:, :Kk]
        tt = {"ind": ind, "mask": torch.arange(Kk, device=ind.device)[None] < npos[:, None],
              "cls": cls_t.reshape(B, -1).gather(1, ind).clamp(max=self.n_cls - 1),
              "attrs": attrs_t.reshape(B, -1, 5).gather(1, ind[..., None].expand(-1, -1, 5))}
        al, parts = attr_loss(out["attr"], tt, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.6):
        B, C, H, W = out["cls"].shape
        p = torch.sigmoid(out["cls"].float())
        res = []
        for b in range(B):
            scores, idx = p[b].reshape(-1).topk(min(K, C * H * W))
            keep = scores > thresh; scores, idx = scores[keep], idx[keep]
            ch = idx // (H * W); cell = idx % (H * W); cy = cell // W; cx = cell % W
            d = self._expect(out["ltrb"][b].reshape(4 * self.BINS, -1)[:, cell].t()) * self.stride       # [n, 4] px
            px = (cx.float() + 0.5) * self.stride; py = (cy.float() + 0.5) * self.stride
            boxes = torch.stack([px - d[:, 0], py - d[:, 1], px + d[:, 2], py + d[:, 3]], 1)
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, cell].t()
            res.append(_dets(boxes, scores, ch, attrs, cy, cx, nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 2. reppoints

class RepPoints(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 240), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        from torchvision.ops import DeformConv2d
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.off = nn.Sequential(ConvBNAct(fpn, hd, 3), nn.Conv2d(hd, 18, 3, padding=1))       # 9 (dy, dx) in cells
        nn.init.zeros_(self.off[-1].weight); nn.init.zeros_(self.off[-1].bias)                  # starts as a plain 3x3
        self.dcn = DeformConv2d(fpn, fpn, 3, padding=1)
        self.dcn_bn = nn.Sequential(nn.BatchNorm2d(fpn), nn.SiLU(inplace=True))
        self.hm = head(fpn, hd, n_cls, prior=0.01)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        base = torch.tensor([[dy, dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=torch.float32)
        self.register_buffer("base", base.reshape(1, 18, 1, 1))
        self.scale = nn.Parameter(torch.tensor(2.0))                                           # offsets are learnt in units of `scale` cells

    def forward(self, x):
        f = self.fpn(self.enc(x))
        off = self.off(f).float() * self.scale                                                 # [B, 18, H, W] learnt deltas (cells)
        g = self.dcn_bn(self.dcn(f.float(), off).to(f.dtype))
        return {"hm": self.hm(g), "attr": self.attr(g), "off": off}

    def points_box(self, off, ind):
        """The nine points at the cells ``ind`` -> [B, K, 4] box (cells, xyxy) from their extent."""
        o = gather_at(off, ind)                                                                # [B, K, 18]
        B, K, _ = o.shape
        W = off.shape[-1]
        cy = (ind // W).float()[..., None] + 0.5; cx = (ind % W).float()[..., None] + 0.5
        py = cy + self.base.reshape(1, 1, 9, 2)[..., 0] + o.reshape(B, K, 9, 2)[..., 0]
        px = cx + self.base.reshape(1, 1, 9, 2)[..., 1] + o.reshape(B, K, 9, 2)[..., 1]
        return torch.stack([px.min(-1).values, py.min(-1).values, px.max(-1).values, py.max(-1).values], -1)

    def loss(self, out, t, attr_cls_mask):
        losses = {"hm": focal(out["hm"], t["hm"])}
        W = out["hm"].shape[-1]
        pb = self.points_box(out["off"], t["ind"])                                             # [B, K, 4] cells
        cy = (t["ind"] // W).float() + t["off"][..., 1]; cx = (t["ind"] % W).float() + t["off"][..., 0]
        w = torch.exp(t["wh"][..., 0]); h = torch.exp(t["wh"][..., 1])
        gb = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        m = t["mask"]
        if m.any():
            losses["giou"] = (1 - _giou(pb[m], gb[m])).mean()
            losses["l1"] = 0.05 * F.l1_loss(pb[m], gb[m])
        al, parts = attr_loss(out["attr"], t, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        hm = torch.sigmoid(out["hm"].float())
        B, C, H, W = hm.shape
        res = []
        for b in range(B):
            s, idx = _peaks(hm[b], K, thresh)
            ch = idx // (H * W); cell = idx % (H * W)
            boxes = self.points_box(out["off"][b:b + 1], cell[None])[0] * self.stride
            attrs = out["attr"][b].float().reshape(N_ATTR, -1)[:, cell].t()
            res.append(_dets(boxes, s, ch, attrs, cell // W, cell % W, nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 3. simota

class SimOTA(FcosDense):
    RADIUS = 2.5                # candidate cells: within this many cells of the centre (the pool SimOTA picks from)
    TOPQ = 10                   # dynamic k = sum of the top-q IoUs (clamped to >= 1)

    def __init__(self, n_cls, **kw):
        super().__init__(n_cls, **kw)
        self.obj = self.ctr                                                        # the same head, read as objectness
        del self.ctr

    def forward(self, x):
        f = self.fpn(self.enc(x))
        return {"cls": self.cls(f), "ltrb": self.ltrb(f), "obj": self.obj(f), "attr": self.attr(f)}

    @torch.no_grad()
    def assign(self, out, t):
        """SimOTA per image: for every GT, candidates = cells within RADIUS of its centre; cost = BCE(cls) + 3 (1 - IoU)
        of the box the cell currently predicts; dynamic k from the top-q IoUs. Returns pos [B,H,W], cls_t, box_t (ltrb
        in cells), attrs_t, iou_t (the assigned cell's IoU, the objectness target)."""
        B, C, H, W = out["cls"].shape
        dev = out["cls"].device
        r = int(math.ceil(self.RADIUS)); d = torch.arange(-r, r + 1, device=dev); win = len(d)
        cls_t = torch.full((B, H, W), self.n_cls, dtype=torch.long, device=dev)
        box_t = torch.zeros(B, 4, H, W, device=dev); iou_t = torch.zeros(B, H, W, device=dev)
        attrs_t = torch.zeros(B, H, W, 5, dtype=torch.long, device=dev)
        pd = torch.exp(out["ltrb"].float().clamp(-4, 6))                                   # [B, 4, H, W] cells
        pc = torch.sigmoid(out["cls"].float()) * torch.sigmoid(out["obj"].float())
        for b in range(B):
            m = t["mask"][b]
            if not m.any():
                continue
            ind = t["ind"][b][m]; K = len(ind)
            ci0, cj0 = ind // W, ind % W
            cx = cj0.float() + t["off"][b][m][:, 0]; cy = ci0.float() + t["off"][b][m][:, 1]
            w = torch.exp(t["wh"][b][m][:, 0]); h = torch.exp(t["wh"][b][m][:, 1])
            gy = (ci0[:, None, None] + d[None, :, None]).expand(K, win, win); gx = (cj0[:, None, None] + d[None, None, :]).expand(K, win, win)
            px = gx.float() + 0.5; py = gy.float() + 0.5
            ok = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W) & ((px - cx[:, None, None]).abs() <= self.RADIUS) & ((py - cy[:, None, None]).abs() <= self.RADIUS)
            gyc, gxc = gy.clamp(0, H - 1), gx.clamp(0, W - 1)
            l_, t_, r_, b_ = [pd[b, k, gyc, gxc] for k in range(4)]                            # [K, win, win]
            x0 = px - l_; y0 = py - t_; x1 = px + r_; y1 = py + b_
            gx0 = (cx - w / 2)[:, None, None]; gy0 = (cy - h / 2)[:, None, None]; gx1 = (cx + w / 2)[:, None, None]; gy1 = (cy + h / 2)[:, None, None]
            inter = (torch.minimum(x1, gx1) - torch.maximum(x0, gx0)).clamp(min=0) * (torch.minimum(y1, gy1) - torch.maximum(y0, gy0)).clamp(min=0)
            iou = inter / ((x1 - x0) * (y1 - y0) + (w * h)[:, None, None] - inter + 1e-6)
            gcls = t["cls"][b][m]
            p_cls = pc[b, gcls[:, None, None].expand(K, win, win), gyc, gxc].clamp(1e-6, 1 - 1e-6)
            cost = (-torch.log(p_cls) + 3.0 * (1 - iou)).masked_fill(~ok, 1e5)
            flat_cost = cost.reshape(K, -1); flat_iou = (iou * ok).reshape(K, -1)
            topq = flat_iou.topk(min(self.TOPQ, flat_iou.shape[1]), 1).values.sum(1).clamp(min=1).long()
            chosen = torch.zeros_like(flat_cost, dtype=torch.bool)
            order = flat_cost.argsort(1)
            rank = torch.zeros_like(order); rank.scatter_(1, order, torch.arange(order.shape[1], device=dev)[None].expand_as(order))
            chosen = (rank < topq[:, None]) & (flat_cost < 1e4)
            # a cell claimed by several objects goes to the cheapest
            cells = (gyc * W + gxc).reshape(K, -1)
            best = torch.full((H * W,), 1e6, device=dev)
            best.scatter_reduce_(0, cells[chosen], flat_cost[chosen], reduce="amin")
            chosen &= flat_cost <= best[cells]
            kk, cc = chosen.nonzero(as_tuple=True)
            cell = cells[kk, cc]; cyy, cxx = cell // W, cell % W
            cls_t[b, cyy, cxx] = gcls[kk]
            box_t[b, 0, cyy, cxx] = (px.reshape(K, -1)[kk, cc] - gx0.reshape(K)[kk]).clamp(min=0.05)
            box_t[b, 1, cyy, cxx] = (py.reshape(K, -1)[kk, cc] - gy0.reshape(K)[kk]).clamp(min=0.05)
            box_t[b, 2, cyy, cxx] = (gx1.reshape(K)[kk] - px.reshape(K, -1)[kk, cc]).clamp(min=0.05)
            box_t[b, 3, cyy, cxx] = (gy1.reshape(K)[kk] - py.reshape(K, -1)[kk, cc]).clamp(min=0.05)
            iou_t[b, cyy, cxx] = flat_iou[kk, cc]
            attrs_t[b, cyy, cxx] = t["attrs"][b][m][kk]
        pos = cls_t.ne(self.n_cls)
        return pos, cls_t, box_t, attrs_t, iou_t

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["cls"].shape
        pos, cls_t, box_t, attrs_t, iou_t = self.assign(out, t)
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
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.6):
        return FcosDense.decode_out(self, {**out, "ctr": out["obj"]}, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 4. staffless-in

def staff_views(x, run=25):
    """x [B, 1, H, W] page in [0, 1] -> [B, 3, H, W]: the page, the page without its long horizontal ink runs, and
    those runs alone. Opening of the ink with a 1 x ``run`` structuring element (erode = min over the run, dilate =
    max back), so only strokes at least ``run`` px long survive: staff lines, beams, long barlines' cross ties."""
    ink = 1.0 - x
    pad = run // 2
    eroded = -F.max_pool2d(-ink, (1, run), stride=1, padding=(0, pad))
    lines = F.max_pool2d(eroded, (1, run), stride=1, padding=(0, pad))
    rest = (ink - lines).clamp(min=0)
    return torch.cat([x, 1.0 - rest, lines], 1)


class StafflessIn(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64, run=25):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths)
        self.enc.stem = ConvBNAct(3, widths[0], 3, s=2)
        self.fpn = FPN(self.enc.out_channels, fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        self.run = run

    def forward(self, x):
        with torch.no_grad():
            v = staff_views(x.float(), self.run).to(x.dtype)
        f = self.fpn(self.enc(v))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 5. swin-trunk

class WinAttn(nn.Module):
    """Windowed multi-head self-attention (non-overlapping ws x ws windows) with a learnt relative position bias,
    + MLP; a 3x3 depthwise conv after the block mixes across windows (instead of Swin's shifted windows)."""

    def __init__(self, d, ws=8, heads=4):
        super().__init__()
        self.ws, self.heads, self.d = ws, heads, d
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        idx = torch.arange(ws); rel = (idx[:, None] - idx[None, :]) + ws - 1
        self.register_buffer("rel", (rel[:, None, :, None] * (2 * ws - 1) + rel[None, :, None, :]).reshape(ws * ws, ws * ws))
        self.mix = nn.Conv2d(d, d, 3, padding=1, groups=d)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        ph, pw = (-H) % ws, (-W) % ws
        xp = F.pad(x, (0, pw, 0, ph))
        Hp, Wp = xp.shape[-2:]
        t = xp.permute(0, 2, 3, 1).reshape(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        h = self.n1(t)
        qkv = self.qkv(h).reshape(t.shape[0], ws * ws, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = self.bias[self.rel].permute(2, 0, 1)[None]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        t = t + self.proj(a.transpose(1, 2).reshape(t.shape[0], ws * ws, C))
        t = t + self.mlp(self.n2(t))
        y = t.reshape(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)[:, :, :H, :W].contiguous()
        return y + self.mix(y)


class SwinTrunk(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 160, 192), depths=(1, 2, 2, 2), fpn=64, hd=64, attr_hd=64, ws=8):
        super().__init__(n_cls)
        s, c2, c3, c4, c5 = widths
        self.stem = ConvBNAct(1, s, 3, s=2)
        self.p2 = nn.Sequential(ConvBNAct(s, c2, 3, s=2), *[Basic(c2) for _ in range(depths[0])])
        self.p3 = nn.Sequential(ConvBNAct(c2, c3, 3, s=2), *[Basic(c3) for _ in range(depths[1])])
        self.p4 = nn.Sequential(DWDown(c3, c4), *[WinAttn(c4, ws) for _ in range(depths[2])])
        self.p5 = nn.Sequential(DWDown(c4, c5), *[WinAttn(c5, ws) for _ in range(depths[3])])
        self.fpn = FPN((c2, c3, c4, c5), fpn)
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        p2 = self.p2(self.stem(x)); p3 = self.p3(p2); p4 = self.p4(p3); p5 = self.p5(p4)
        f = self.fpn((p2, p3, p4, p5))
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 6. siam-scale

class SiamScale(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.gate = nn.Sequential(nn.Conv2d(2 * fpn, fpn, 1), nn.Sigmoid())
        self.hm = head(fpn, hd, n_cls, prior=0.01); self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))

    def forward(self, x):
        f1 = self.fpn(self.enc(x))
        f0 = self.fpn(self.enc(F.avg_pool2d(x, 2)))
        f0 = F.interpolate(f0, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([f1, f0], 1))
        f = f1 + g * f0
        return {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}

    loss = _flat_loss

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.7):
        return M.decode(_Flat(self), out, K, thresh, nms_iou)


CONFIGS5 = {"gfl-dense": GflDense, "reppoints": RepPoints, "simota": SimOTA, "staffless-in": StafflessIn,
            "swin-trunk": SwinTrunk, "siam-scale": SiamScale}


def build5(name, classes, **override):
    return CONFIGS5[name](len(classes), **override)
