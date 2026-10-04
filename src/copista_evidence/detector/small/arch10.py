"""Following the conclusion: what the background-competition finding implies.

The search found that the 22-point handwritten precision gap comes from making each class compete against an
explicit background class, and that a gathered second decision buys another ~11 points of precision for ~18 points
of recall, with the transformer worth 2-4 and the context width worth nothing. These four ask what that is worth in
practice:

  softmax-verify  both mechanisms at once: the softmax-with-background readout for recall, plus the gathered MLP
                  verifier for precision. Does it reach bg-verify's precision at softmax-bg's recall, or do the two
                  suppress the same false positives twice?
  softmax-fcos    the same background competition on a completely different readout -- the dense side-distance
                  (FCOS) design, whose class head is per-class sigmoids today. If the mechanism is general it should
                  move fcos-dense the same way it moved cnet-flat; if it only helps the centre-heatmap readout, the
                  finding is narrower than it looks.
  frozen-verify   the verifier trained on top of a FROZEN one-stage trunk, its heatmap and box heads never updated.
                  If that works, a verifier can be bolted onto an already-deployed detector for the cost of one
                  small head and a re-run of training that touches nothing else.
  tiny-verify     how cheap the verifier can be: one hidden layer of width 32 over a 1x1 centre feature (no RoI
                  grid) and 400 candidates instead of 2000. Sets the real inference cost of the recipe.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_ATTR
from . import models as M
from .arch4 import FcosDense, _Flat
from .arch7 import SoftmaxBg
from .arch9 import BgVerify, _mined_negatives
from .models import Encoder, FPN, attr_loss, focal, gather_at, head


# ----------------------------------------------------------------------------------------------------------------
# 1. softmax-verify -- background competition in the dense readout AND a gathered verifier on top

class SoftmaxVerify(BgVerify):
    """bg-verify whose proposal stage is a full softmax-with-background class map rather than a class-agnostic
    objectness map: the dense stage already competes against background, and the verifier re-decides."""

    W_OBJ = 0.0                     # the dense softmax trains the proposals; do not also focal-train the derived map

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 248), depths=(1, 2, 3, 2), fpn=64, hd=64, d=128):
        super().__init__(n_cls, widths=widths, depths=depths, fpn=fpn, hd=hd, d=d)
        self.obj = None
        self.dense = head(fpn, hd, n_cls + 1)
        nn.init.constant_(self.dense[-1].bias, 0.0); self.dense[-1].bias.data[n_cls] = 4.0

    def forward(self, x):
        f = self.fpn(self.enc(x))
        dense = self.dense(f)
        # objectness for the proposal stage = 1 - p(background)
        p_bg = torch.softmax(dense.float(), 1)[:, self.n_cls:self.n_cls + 1]
        obj = torch.log((1 - p_bg).clamp(1e-6, 1 - 1e-6) / p_bg.clamp(1e-6, 1 - 1e-6))
        return {"f": f, "dense": dense, "obj": obj.to(dense.dtype), "reg": self.reg(f)}

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["dense"].shape
        # the dense stage: per-cell softmax over classes + background, hard-negative mined (as softmax-bg)
        tgt = torch.full((B, H * W), self.n_cls, dtype=torch.long, device=out["dense"].device)
        tgt.scatter_(1, t["ind"], torch.where(t["mask"], t["cls"], torch.full_like(t["cls"], self.n_cls)))
        tgt = tgt.reshape(B, H, W)
        halo = (t["hm"].amax(1) > 0.3) & tgt.eq(self.n_cls)
        pos = tgt.ne(self.n_cls)
        ce = F.cross_entropy(out["dense"].float(), tgt, reduction="none")
        n_pos = int(pos.sum())
        neg = ce.masked_fill(pos | halo, 0.0).reshape(B, -1)
        k = min(neg.shape[1], max(256, 3 * max(1, n_pos // B)))
        losses = {"dense": (ce[pos].sum() + neg.topk(k, 1).values.sum()) / max(1, n_pos + k * B) * 8.0}
        # the gathered verifier, exactly as bg-verify
        base, parts, aparts = BgVerify.loss(self, out, t, attr_cls_mask)
        return losses["dense"] + base, {**parts, "dense": float(losses["dense"])}, aparts


# ----------------------------------------------------------------------------------------------------------------
# 2. softmax-fcos -- the same competition on the dense side-distance readout

class SoftmaxFcos(FcosDense):
    """fcos-dense with its per-class sigmoid head replaced by a softmax over the classes plus background."""

    def __init__(self, n_cls, **kw):
        super().__init__(n_cls, **kw)
        fpn = self.cls[0][0].in_channels; hd = self.cls[0][0].out_channels
        self.cls = head(fpn, hd, n_cls + 1)
        nn.init.constant_(self.cls[-1].bias, 0.0); self.cls[-1].bias.data[n_cls] = 4.0

    def loss(self, out, t, attr_cls_mask):
        B, _, H, W = out["cls"].shape
        cls_t, box_t, ctr_t, pos, attrs_t = self.targets(t, H, W)
        ce = F.cross_entropy(out["cls"].float(), cls_t, reduction="none")
        n_pos = int(pos.sum())
        neg = ce.masked_fill(pos, 0.0).reshape(B, -1)
        k = min(neg.shape[1], max(256, 3 * max(1, n_pos // B)))
        losses = {"cls": (ce[pos].sum() + neg.topk(k, 1).values.sum()) / max(1, n_pos + k * B) * 8.0}
        pl = out["ltrb"].float().permute(0, 2, 3, 1)[pos]; gl = box_t.permute(0, 2, 3, 1)[pos]
        if len(pl):
            pw = torch.exp(pl.clamp(-4, 6))
            iw = torch.minimum(pw[:, 0], gl[:, 0]) + torch.minimum(pw[:, 2], gl[:, 2])
            ih = torch.minimum(pw[:, 1], gl[:, 1]) + torch.minimum(pw[:, 3], gl[:, 3])
            inter = iw.clamp(min=0) * ih.clamp(min=0)
            ap = (pw[:, 0] + pw[:, 2]) * (pw[:, 1] + pw[:, 3]); ag = (gl[:, 0] + gl[:, 2]) * (gl[:, 1] + gl[:, 3])
            losses["iou"] = -(torch.log((inter + 1.0) / (ap + ag - inter + 1.0))).mean()
            losses["ctr"] = F.binary_cross_entropy_with_logits(out["ctr"].float()[:, 0][pos], ctr_t[pos])
        ind = pos.reshape(B, -1).float().argsort(dim=1, descending=True)
        npos = pos.reshape(B, -1).sum(1); Kk = int(npos.max().clamp(min=1)); ind = ind[:, :Kk]
        tt = {"ind": ind, "mask": torch.arange(Kk, device=ind.device)[None] < npos[:, None],
              "cls": cls_t.reshape(B, -1).gather(1, ind).clamp(max=self.n_cls - 1),
              "attrs": attrs_t.reshape(B, -1, 5).gather(1, ind[..., None].expand(-1, -1, 5))}
        al, parts = attr_loss(out["attr"], tt, attr_cls_mask); losses["attr"] = al
        return sum(losses.values()), {k_: float(v) for k_, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.6):
        p = torch.softmax(out["cls"].float(), 1)[:, :self.n_cls]
        lg = torch.log(p.clamp(1e-6, 1 - 1e-6) / (1 - p).clamp(1e-6, 1 - 1e-6))
        return FcosDense.decode_out(self, {**out, "cls": lg}, K, thresh, nms_iou)


# ----------------------------------------------------------------------------------------------------------------
# 3. frozen-verify -- only the verifier learns

class FrozenVerify(BgVerify):
    """The trunk, the objectness map and the box head are frozen (loaded from a trained one-stage run if present);
    only the gathered verifier trains. The question is whether a verifier can be bolted onto a deployed detector."""

    BASE = "/root/omr/runs/ref/cnet-flat/best.pt"

    def __init__(self, n_cls, **kw):
        super().__init__(n_cls, **kw)
        import os
        if os.path.exists(self.BASE):
            ck = torch.load(self.BASE, map_location="cpu", weights_only=False)
            sd = {k: v for k, v in ck["ema"].items() if k.startswith(("enc.", "fpn.", "reg."))}
            missing = self.load_state_dict(sd, strict=False)
            print(f"frozen-verify: loaded {len(sd)} tensors from {self.BASE}; unexpected {len(missing.unexpected_keys)}", flush=True)
        for mod in (self.enc, self.fpn, self.reg):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

    def train(self, mode=True):
        super().train(mode)
        for mod in (self.enc, self.fpn, self.reg):          # keep the frozen parts in eval mode (batchnorm)
            mod.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            f = self.fpn(self.enc(x))
            reg = self.reg(f)
        return {"f": f, "obj": self.obj(f), "reg": reg}


# ----------------------------------------------------------------------------------------------------------------
# 4. tiny-verify -- the cheapest verifier that still decides

class TinyVerify(BgVerify):
    """One hidden layer of width 32 over the centre feature alone (no RoI grid), and 400 candidates at inference."""

    N_NEG = 96
    K_INFER = 400
    ROI = 1

    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, d=32):
        super().__init__(n_cls, widths=widths, depths=depths, fpn=fpn, hd=hd, d=d)
        self.proj = nn.Sequential(nn.Linear(fpn * self.ROI * self.ROI + fpn, d), nn.GELU())
        self.cls = nn.Linear(d, n_cls + 1)
        self.attr = nn.Linear(d, N_ATTR)
        nn.init.constant_(self.cls.bias, 0.0); self.cls.bias.data[n_cls] = 2.0


CONFIGS10 = {"softmax-verify": SoftmaxVerify, "softmax-fcos": SoftmaxFcos, "frozen-verify": FrozenVerify,
             "tiny-verify": TinyVerify}


def build10(name, classes, **override):
    return CONFIGS10[name](len(classes), **override)
