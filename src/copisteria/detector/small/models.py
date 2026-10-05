"""The centre-heatmap model family for the 2M-parameter search: one grayscale page in, dense maps at stride 4 out.

Every variant shares the trunk (a small conv encoder with residual 3x3 blocks at strides 4/8 where the small
glyphs live and depthwise-separable blocks at 16/32, an FPN back to stride 4) and the readout: a class heatmap
(peaks = object centres, no anchors, no NMS beyond a 3x3 max-pool), width/height + sub-cell offset at the centre,
and the note attributes (staff_position, stem_dir, dots, grace, voice_slot) read at the same centre from a
dilated head whose receptive field spans the staff. Variants:

  flat      one heatmap channel per class (266)                                   -- the CenterNet reading
  family    one heatmap channel per FAMILY (36) + a class classifier at the centre  -- hierarchical detection
  staff     flat + an auxiliary staff-line map, so staff_position can also be read geometrically
  axial     flat + row / column self-attention at strides 16 and 32 (global horizontal context along the staff)

``build(name)`` returns the configured model; ``count(model)`` its parameter count. Widths were chosen so every
configuration lands close to (and under) 2.0M parameters -- see CONFIGS.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR


class ConvBNAct(nn.Sequential):
    def __init__(self, i, o, k=3, s=1, d=1, g=1, act=True):
        p = d * (k - 1) // 2
        layers = [nn.Conv2d(i, o, k, s, p, dilation=d, groups=g, bias=False), nn.BatchNorm2d(o)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class Basic(nn.Module):
    """Residual block of two 3x3 convs (ResNet-18 style): full convs where the glyphs are small."""

    def __init__(self, c):
        super().__init__()
        self.a = ConvBNAct(c, c, 3); self.b = ConvBNAct(c, c, 3, act=False)

    def forward(self, x):
        return F.silu(x + self.b(self.a(x)))


class DW(nn.Module):
    """Inverted residual (MobileNetV2 style, expansion 2): the deep, wide stages at a quarter of the parameters."""

    def __init__(self, c, e=2):
        super().__init__()
        h = c * e
        self.f = nn.Sequential(ConvBNAct(c, h, 1), ConvBNAct(h, h, 3, g=h), ConvBNAct(h, c, 1, act=False))

    def forward(self, x):
        return x + self.f(x)


class DWDown(nn.Module):
    """Stride-2 depthwise 3x3 + pointwise: a cheap downsampling step for the wide stages."""

    def __init__(self, i, o):
        super().__init__()
        self.f = nn.Sequential(ConvBNAct(i, i, 3, s=2, g=i), ConvBNAct(i, o, 1))

    def forward(self, x):
        return self.f(x)


class AxialBlock(nn.Module):
    """Self-attention along rows then along columns of a feature map (projected to ``d`` channels), + MLP. A staff
    system is a horizontal strip: row attention lets every cell see the whole staff line it sits on."""

    def __init__(self, c, d=96, heads=4):
        super().__init__()
        self.inp = nn.Conv2d(c, d, 1); self.out = nn.Conv2d(d, c, 1)
        self.row = nn.MultiheadAttention(d, heads, batch_first=True)
        self.col = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.pos_r = nn.Parameter(torch.zeros(1, 1, 1, d)); self.pos_c = nn.Parameter(torch.zeros(1, 1, 1, d))
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)     # starts as identity: the trunk is not disturbed

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.inp(x).permute(0, 2, 3, 1)                                  # [B, H, W, d]
        r = self.n1(t).reshape(B * H, W, -1)
        t = t + self.row(r, r, r, need_weights=False)[0].reshape(B, H, W, -1)
        c = self.n2(t).permute(0, 2, 1, 3).reshape(B * W, H, -1)
        t = t + self.col(c, c, c, need_weights=False)[0].reshape(B, W, H, -1).permute(0, 2, 1, 3)
        t = t + self.mlp(self.n3(t))
        return x + self.out(t.permute(0, 3, 1, 2))


class Encoder(nn.Module):
    """stem (stride 2) -> P2 (4) -> P3 (8) -> P4 (16) -> P5 (32). ``widths`` = (stem, P2, P3, P4, P5), ``depths`` the
    block counts at P2..P5; P2 / P3 use Basic blocks, P4 / P5 depthwise ones."""

    def __init__(self, widths=(24, 48, 96, 160, 224), depths=(1, 2, 3, 2), axial=False):
        super().__init__()
        s, c2, c3, c4, c5 = widths
        self.stem = ConvBNAct(1, s, 3, s=2)
        self.p2 = nn.Sequential(ConvBNAct(s, c2, 3, s=2), *[Basic(c2) for _ in range(depths[0])])
        self.p3 = nn.Sequential(ConvBNAct(c2, c3, 3, s=2), *[Basic(c3) for _ in range(depths[1])])
        self.p4 = nn.Sequential(DWDown(c3, c4), *[DW(c4) for _ in range(depths[2])], *([AxialBlock(c4)] if axial else []))
        self.p5 = nn.Sequential(DWDown(c4, c5), *[DW(c5) for _ in range(depths[3])], *([AxialBlock(c5)] if axial else []))
        self.out_channels = (c2, c3, c4, c5)

    def forward(self, x):
        x = self.stem(x)
        p2 = self.p2(x); p3 = self.p3(p2); p4 = self.p4(p3); p5 = self.p5(p4)
        return p2, p3, p4, p5


class FPN(nn.Module):
    """Top-down fusion to one stride-4 map of ``d`` channels."""

    def __init__(self, chans, d=64):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in chans])
        self.smooth = nn.ModuleList([ConvBNAct(d, d, 3) for _ in chans[:-1]])

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        x = self.lat[3](p5)
        for k, f in ((2, p4), (1, p3), (0, p2)):
            x = F.interpolate(x, size=f.shape[-2:], mode="nearest") + self.lat[k](f)
            x = self.smooth[k](x)
        return x


def head(d, hd, out, dilations=(1,), prior=None):
    layers = [ConvBNAct(d, hd, 3, d=dilations[0])] + [ConvBNAct(hd, hd, 3, d=dl) for dl in dilations[1:]]
    last = nn.Conv2d(hd, out, 1)
    if prior is not None:
        nn.init.constant_(last.bias, -math.log((1 - prior) / prior))
    return nn.Sequential(*layers, last)


class CenterNet(nn.Module):
    def __init__(self, n_cls, mode="flat", fam_of_cls=None, n_fam=0, staff=False, axial=False,
                 widths=(24, 48, 96, 160, 224), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__()
        self.n_cls, self.mode, self.staff_head = n_cls, mode, staff
        self.enc = Encoder(widths, depths, axial=axial)
        self.fpn = FPN(self.enc.out_channels, fpn)
        n_hm = n_fam if mode == "family" else n_cls
        self.hm = head(fpn, hd, n_hm, prior=0.01)
        self.reg = head(fpn, hd, 4)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4, 8))
        if mode == "family":
            self.cls = head(fpn, hd, n_cls)
            self.register_buffer("fam_of_cls", torch.as_tensor(fam_of_cls, dtype=torch.long))
            self.n_fam = n_fam
        if staff:
            self.staff = head(fpn, 16, 1, prior=0.05)
        self.stride = 4

    def forward(self, x):
        f = self.fpn(self.enc(x))
        out = {"hm": self.hm(f), "reg": self.reg(f), "attr": self.attr(f)}
        if self.mode == "family":
            out["cls"] = self.cls(f)
        if self.staff_head:
            out["staff"] = self.staff(f)
        return out


# ----------------------------------------------------------------------------------------------------------------

CONFIGS = {
    # name: kwargs beyond n_cls / family tables. Widths trimmed per variant so every model sits just under 2.0M.
    "cnet-flat":   dict(mode="flat",   widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2)),
    "cnet-family": dict(mode="family", widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2)),
    "cnet-staff":  dict(mode="flat",   widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), staff=True),
    "cnet-axial":  dict(mode="flat",   widths=(24, 48, 96, 160, 224), depths=(1, 2, 2, 1), axial=True),
}


def families(classes):
    """(family names, family id per class) -- the class name's prefix before '_'."""
    fams = sorted({c.partition("_")[0] for c in classes})
    fid = {f: i for i, f in enumerate(fams)}
    return fams, [fid[c.partition("_")[0]] for c in classes]


ARCH2 = ("peak-relational", "seg-vote", "cnet-recur", "cnet-distill", "cnet-twostream", "cnet-embed")   # copisteria.detector.small.arch2
ARCH3 = ("col-slots", "staff-canon")                                                                     # copisteria.detector.small.arch3
ARCH4 = ("scan-state", "fovea", "dyn-proposals", "ink-proto", "hires-out", "fcos-dense")                  # copisteria.detector.small.arch4
ARCH5 = ("gfl-dense", "reppoints", "simota", "staffless-in", "swin-trunk", "siam-scale")                   # copisteria.detector.small.arch5
ARCH6 = ("corners", "retina-anchors", "page-moe", "nat-head", "o2o-dense", "recon-aux")                     # copisteria.detector.small.arch6
ARCH7 = ("softmax-bg", "embed-nms", "duo-vote", "cls-prior-box", "gfnet-ctx", "scribble-aug")                # copisteria.detector.small.arch7
ARCH8 = ("cascade-iou", "ema-consist", "hash-cls", "copy-paste", "deep-super", "scale-fuse")                 # copisteria.detector.small.arch8
ARCH9 = ("bg-dense", "bg-verify", "ctx-crop", "obj-recall", "obj-recall-wide", "obj-recall-30m", "self-nms", "mix-teacher")                       # copisteria.detector.small.arch9
ARCH10 = ("softmax-verify", "softmax-fcos", "frozen-verify", "tiny-verify")                                 # copisteria.detector.small.arch10
CONFIGS.update({n: None for n in ARCH2 + ARCH3 + ARCH4 + ARCH5 + ARCH6 + ARCH7 + ARCH8 + ARCH9 + ARCH10})


def build(name, classes, **override):
    if name in ARCH2:
        from .arch2 import build2
        return build2(name, classes, **override)
    if name in ARCH3:
        from .arch3 import build3
        return build3(name, classes, **override)
    if name in ARCH4:
        from .arch4 import build4
        return build4(name, classes, **override)
    if name in ARCH5:
        from .arch5 import build5
        return build5(name, classes, **override)
    if name in ARCH6:
        from .arch6 import build6
        return build6(name, classes, **override)
    if name in ARCH7:
        from .arch7 import build7
        return build7(name, classes, **override)
    if name in ARCH8:
        from .arch8 import build8
        return build8(name, classes, **override)
    if name in ARCH9:
        from .arch9 import build9
        return build9(name, classes, **override)
    if name in ARCH10:
        from .arch10 import build10
        return build10(name, classes, **override)
    cfg = {**CONFIGS[name], **override}
    fams, fam_of_cls = families(classes)
    return CenterNet(len(classes), fam_of_cls=fam_of_cls, n_fam=len(fams), **cfg)


def count(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------------------------------------------
# losses

def _focal_sample(logits, g, eps=1e-4):
    """One sample's penalty-reduced focal terms (summed), exact CenterNet form."""
    p = torch.sigmoid(logits.float()).clamp(eps, 1 - eps)
    pos = g.eq(1)
    pos_loss = (torch.log(p[pos]) * (1 - p[pos]) ** 2).sum()
    neg_loss = (torch.log1p(-p) * p * p * (1 - g) ** 4).masked_fill(pos, 0.0).sum()
    return -(pos_loss + neg_loss)


def focal(pred_logits, gt, eps=1e-4):
    """CenterNet's penalty-reduced focal loss on a sigmoid heatmap; normalised by the number of centres. Exact, but
    computed one sample at a time under activation checkpointing: the elementwise intermediates ([266, 256, 256]
    float32, five of them per sample) are recomputed in backward instead of kept -- over a batch of 16 they were
    6.8 GB, and a whole-batch version took 24 GB per run."""
    from torch.utils.checkpoint import checkpoint
    n = gt.eq(1).sum().clamp(min=1).float()
    total = pred_logits.new_zeros((), dtype=torch.float32)
    for b in range(pred_logits.shape[0]):
        total = total + checkpoint(_focal_sample, pred_logits[b], gt[b], eps, use_reentrant=False)
    return total / n


def gather_at(feat, ind):
    """feat [B, C, H, W], ind [B, K] flat cell index -> [B, K, C] in float32 (gathered in the map's own dtype, cast
    after: casting a whole dense head first cost a 1-2 GB copy per head)."""
    B, C = feat.shape[:2]
    f = feat.view(B, C, -1).permute(0, 2, 1)
    return f.gather(1, ind.unsqueeze(-1).expand(-1, -1, C)).float()


ATTR_CLASS_PREFIXES = ("note", "rest", "mRest", "multiRest")


def attr_loss(attr_logits, t, attr_cls_mask):
    """Cross-entropy per attribute at the note / rest centres. ``attr_cls_mask`` [n_cls] bool: classes that carry
    attributes. staff_position / voice_slot skip <na> targets; dots / grace / stem_dir learn <na> as an answer."""
    B, K = t["ind"].shape
    logits = gather_at(attr_logits, t["ind"])                         # [B, K, N_ATTR]
    m = t["mask"] & attr_cls_mask[t["cls"]]
    if not m.any():
        return attr_logits.sum() * 0, {}
    total = 0.0; parts = {}
    from .data import LEARN_NA
    for k, a in enumerate(ATTRS):
        o, n = ATTR_OFFSETS[a], ATTR_SIZES[a]
        lg = logits[..., o:o + n][m]
        gt = t["attrs"][..., k][m]
        mm = torch.ones_like(gt, dtype=torch.bool) if a in LEARN_NA else gt.ne(0)
        if mm.any():
            l = F.cross_entropy(lg[mm], gt[mm])
            total = total + l; parts[a] = float(l)
    return total, parts


def compute_loss(model, out, t, attr_cls_mask, fam_hm=None, staff_gt=None, w_wh=0.1, w_off=1.0, w_attr=1.0, w_staff=1.0):
    if hasattr(model, "loss"):                                             # the round-2 designs (copisteria.detector.small.arch2)
        return model.loss(out, t, attr_cls_mask)
    losses = {}
    if model.mode == "family":
        losses["hm"] = focal(out["hm"], fam_hm)
        # class at the centre, softmax restricted to the family's classes
        lg = gather_at(out["cls"], t["ind"])                               # [B, K, n_cls]
        fam_of = model.fam_of_cls                                          # [n_cls]
        m = t["mask"]
        if m.any():
            g_cls = t["cls"][m]; g_fam = fam_of[g_cls]
            lgm = lg[m]
            allowed = fam_of[None, :] == g_fam[:, None]
            lgm = lgm.masked_fill(~allowed, -1e4)
            losses["cls"] = F.cross_entropy(lgm, g_cls)
    else:
        losses["hm"] = focal(out["hm"], t["hm"])
    reg = gather_at(out["reg"], t["ind"])                                    # [B, K, 4]
    m = t["mask"].unsqueeze(-1).float()
    n = t["mask"].sum().clamp(min=1)
    losses["wh"] = w_wh * (F.l1_loss(reg[..., :2], t["wh"], reduction="none") * m).sum() / n
    losses["off"] = w_off * (F.l1_loss(reg[..., 2:], t["off"], reduction="none") * m).sum() / n
    al, parts = attr_loss(out["attr"], t, attr_cls_mask)
    losses["attr"] = w_attr * al
    if model.staff_head and staff_gt is not None:
        losses["staff"] = w_staff * F.binary_cross_entropy_with_logits(out["staff"].float(), staff_gt)
    total = sum(losses.values())
    return total, {k: float(v) for k, v in losses.items()}, parts


# ----------------------------------------------------------------------------------------------------------------
# decoding

@torch.no_grad()
def decode(model, out, K=3000, thresh=0.03, nms_iou=0.7):
    """Peaks of the heatmap -> boxes. Returns per image a dict of tensors: boxes [n, 4] (pixels), scores [n],
    cls [n], attrs [n, N_ATTR] logits, cell [n, 2] (cy, cx cell index)."""
    if hasattr(model, "decode_out"):                                       # the round-2 designs (copisteria.detector.small.arch2)
        return model.decode_out(out, K=K, thresh=thresh, nms_iou=nms_iou)
    stride = model.stride
    hm = torch.sigmoid(out["hm"].float())
    B, C, H, W = hm.shape
    pooled = F.max_pool2d(hm, 3, 1, 1)
    hm = hm * (pooled == hm)
    res = []
    for b in range(B):
        scores, idx = hm[b].view(-1).topk(min(K, C * H * W))
        keep = scores > thresh
        scores, idx = scores[keep], idx[keep]
        ch = idx // (H * W); cell = idx % (H * W); cy = cell // W; cx = cell % W
        reg = out["reg"][b].float().view(4, -1)[:, cell]                     # [4, n]
        w, h = torch.exp(reg[0]) * stride, torch.exp(reg[1]) * stride
        x = (cx.float() + reg[2]) * stride; y = (cy.float() + reg[3]) * stride
        boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
        attrs = out["attr"][b].float().view(N_ATTR, -1)[:, cell].t()
        if model.mode == "family":
            cls_logits = out["cls"][b].float().view(model.n_cls, -1)[:, cell].t()      # [n, n_cls]
            allowed = model.fam_of_cls[None, :] == ch[:, None]
            prob = torch.softmax(cls_logits.masked_fill(~allowed, -1e4), -1)
            pc, cls = prob.max(-1)
            scores = scores * pc
        else:
            cls = ch
        if nms_iou and len(boxes):
            from torchvision.ops import batched_nms
            k = batched_nms(boxes, scores, cls, nms_iou)
            boxes, scores, cls, attrs, cy, cx = boxes[k], scores[k], cls[k], attrs[k], cy[k], cx[k]
        res.append({"boxes": boxes, "scores": scores, "cls": cls, "attrs": attrs, "cell": torch.stack([cy, cx], 1)})
    return res


def attrs_argmax(attr_logits):
    """[n, N_ATTR] logits -> [n, 5] ids (one per attribute group)."""
    return torch.stack([attr_logits[:, ATTR_OFFSETS[a]:ATTR_OFFSETS[a] + ATTR_SIZES[a]].argmax(-1) for a in ATTRS], 1)


# ----------------------------------------------------------------------------------------------------------------
# geometric staff position from the staff-line map (the "staff" variant)

@torch.no_grad()
def geometric_positions(staff_logits, boxes, cls, note_mask, stride=4, min_peak=0.3):
    """For every detection with ``note_mask`` True, read staff_position from the predicted staff-line map: take the
    map's column at the note's x (3 cells wide), find its peaks, group them into runs of near-equal spacing, and
    pick the 5-line group nearest the note's centre; position = round(2 * (mid_line_y - cy) / spacing). Returns
    [n] ints (staff_position) with a large sentinel (99) where no staff group was found."""
    import numpy as np
    sm = torch.sigmoid(staff_logits.float()).cpu().numpy().reshape(staff_logits.shape[-2:])   # [Hs, Ws]
    Hs, Ws = sm.shape
    out = np.full(len(boxes), 99, dtype=np.int64)
    bx = boxes.cpu().numpy()
    for i in np.nonzero(note_mask.cpu().numpy())[0]:
        cx = (bx[i, 0] + bx[i, 2]) / 2; cy = (bx[i, 1] + bx[i, 3]) / 2
        j = int(cx // stride)
        if j < 0 or j >= Ws:
            continue
        col = sm[:, max(0, j - 1):min(Ws, j + 2)].mean(1)
        # sub-cell peak positions by parabolic interpolation
        peaks = []
        for y in range(1, Hs - 1):
            if col[y] >= min_peak and col[y] >= col[y - 1] and col[y] > col[y + 1]:
                a, b, c = col[y - 1], col[y], col[y + 1]
                den = a - 2 * b + c
                d = 0.5 * (a - c) / den if den != 0 else 0.0
                peaks.append((y + d + 0.5) * stride)
        if len(peaks) < 5:
            continue
        peaks = np.asarray(peaks)
        gaps = np.diff(peaks)
        best = None
        for s0 in range(len(peaks) - 4):
            g = gaps[s0:s0 + 4]
            med = np.median(g)
            if med <= 0 or (np.abs(g - med) > 0.25 * med).any():
                continue
            mid = peaks[s0 + 2]
            dist = abs(mid - cy)
            if best is None or dist < best[0]:
                best = (dist, mid, med)
        if best is None:
            continue
        _, mid, sp = best
        out[i] = int(round(2 * (mid - cy) / sp))
    return out
