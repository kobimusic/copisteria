"""Data for the custom (non-ultralytics) loops: the exported pages (copista_evidence.detector.small.export) as random square crops for
training and whole pages for validation, and the dense targets a centre-heatmap model trains on, rendered on the
GPU from the box lists (a 266-channel heatmap per sample is 70 MB at stride 4 -- far too much to move per sample
from a worker).

Boxes travel as float tensors ``[N, 10]``: cls, x0, y0, x1, y1 (pixels of the returned image), pos, stem, dots,
grace, voice (attribute ids; copista_evidence.detector.small.export.ATTR_VOCAB). Staves as a list of polylines (pixels).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .export import ATTR_VOCAB, ATTRS

ATTR_SIZES = {a: len(v) for a, v in ATTR_VOCAB.items()}
N_ATTR = sum(ATTR_SIZES.values())                # 49+1 + 3 + 5 + 3 + 9 = 70 logits
ATTR_OFFSETS = {}
_o = 0
for _a in ATTRS:
    ATTR_OFFSETS[_a] = _o
    _o += ATTR_SIZES[_a]
LEARN_NA = ("dots", "grace", "stem_dir")         # <na> is a real answer there (copista_evidence.detector.det.attr_stage)


def load_meta(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _read_image(path):
    """Grayscale float32 HxW in [0, 1]. Pages are near-monochrome; one channel keeps the stem cheap."""
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


class CropDataset(Dataset):
    """Training samples: a ``crop`` x ``crop`` window of the page after a random rescale in ``scale`` (the page is
    padded with paper white when smaller than the window). Boxes are rescaled, clipped, and dropped when less than
    ``min_vis`` of their area survives; staves are rescaled and kept (points may fall outside the window)."""

    def __init__(self, root, meta_rows, crop=1024, scale=(0.8, 1.25), min_vis=0.4, seed=0):
        self.root = Path(root)
        self.rows = meta_rows
        self.crop, self.scale, self.min_vis = crop, scale, min_vis
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = _read_image(self.root / r["img"])
        s = self.rng.uniform(*self.scale)
        h, w = img.shape
        if abs(s - 1.0) > 1e-3:
            import cv2
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            h, w = img.shape
        C = self.crop
        x_off = self.rng.randint(0, max(0, w - C)) if w > C else 0
        y_off = self.rng.randint(0, max(0, h - C)) if h > C else 0
        out = np.ones((C, C), np.float32)
        hh, ww = min(C, h - y_off), min(C, w - x_off)
        out[:hh, :ww] = img[y_off:y_off + hh, x_off:x_off + ww]
        boxes = np.asarray(r["boxes"], np.float32).reshape(-1, 10)
        if len(boxes):
            b = boxes.copy()
            b[:, 1:5] = b[:, 1:5] * s - np.array([x_off, y_off, x_off, y_off], np.float32)
            area = np.maximum(0, b[:, 3] - b[:, 1]) * np.maximum(0, b[:, 4] - b[:, 2])
            c = b.copy()
            c[:, 1:5] = np.clip(c[:, 1:5], 0, C)
            vis = np.maximum(0, c[:, 3] - c[:, 1]) * np.maximum(0, c[:, 4] - c[:, 2])
            keep = (vis >= self.min_vis * np.maximum(area, 1e-3)) & (vis > 1.0)
            boxes = c[keep]
        staves = [[[px * s - x_off, py * s - y_off] for px, py in poly] for st in r["staves"] for poly in st]
        shift = np.array([x_off, y_off], np.float32)
        staffs = [[np.asarray(poly, np.float32) * s - shift for poly in st] for st in r["staves"]]      # grouped per staff
        return {"img": torch.from_numpy(out)[None], "boxes": torch.from_numpy(boxes), "staves": staves, "staffs": staffs,
                "sp": (r["sp"] or 0.0) * s, "id": r["id"]}


class PageDataset(Dataset):
    """Validation: the whole page, padded (paper white) to a multiple of ``pad_to``; boxes in page pixels."""

    def __init__(self, root, meta_rows, pad_to=32):
        self.root = Path(root); self.rows = meta_rows; self.pad_to = pad_to

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = _read_image(self.root / r["img"])
        h, w = img.shape
        H, W = math.ceil(h / self.pad_to) * self.pad_to, math.ceil(w / self.pad_to) * self.pad_to
        out = np.ones((H, W), np.float32); out[:h, :w] = img
        boxes = np.asarray(r["boxes"], np.float32).reshape(-1, 10)
        staves = [poly for st in r["staves"] for poly in st]
        staffs = [[np.asarray(poly, np.float32) for poly in st] for st in r["staves"]]
        return {"img": torch.from_numpy(out)[None], "boxes": torch.from_numpy(boxes), "staves": staves, "staffs": staffs,
                "sp": r["sp"] or 0.0, "id": r["id"], "hw": (h, w)}


def collate(batch):
    """Images stacked (same size within a batch: crops, or pages padded to the batch max); boxes kept as a list."""
    imgs = [b["img"] for b in batch]
    H = max(i.shape[1] for i in imgs); W = max(i.shape[2] for i in imgs)
    x = torch.ones(len(imgs), 1, H, W)
    for k, im in enumerate(imgs):
        x[k, :, :im.shape[1], :im.shape[2]] = im
    return {"img": x, "boxes": [b["boxes"] for b in batch], "staves": [b["staves"] for b in batch], "staffs": [b.get("staffs", []) for b in batch],
            "sp": [b["sp"] for b in batch], "id": [b["id"] for b in batch], "hw": [b.get("hw") for b in batch]}


# ----------------------------------------------------------------------------------------------------------------
# dense targets (GPU)

def render_targets(boxes_list, H, W, stride, n_cls, device, max_win=15, min_sigma=0.6):
    """Centre-heatmap targets for a batch. ``boxes_list``: list of [N, 10] tensors (image pixels). Returns dict:
      hm     [B, n_cls, H/s, W/s]  anisotropic Gaussians (sigma = box side in cells / 6, floor min_sigma), max-merged
      ind    [B, K] flat cell index of every object's centre (padded), mask [B, K] bool, K = max objects in batch
      wh     [B, K, 2] log(width, height in cells); off [B, K, 2] sub-cell offset of the centre
      cls    [B, K] class id; attrs [B, K, 5] attribute ids; fam_hm rendered by the caller if wanted
    """
    B = len(boxes_list)
    Hs, Ws = H // stride, W // stride
    hm = torch.zeros(B, n_cls, Hs, Ws, device=device)
    K = max(1, max(len(b) for b in boxes_list))
    ind = torch.zeros(B, K, dtype=torch.long, device=device)
    mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    wh = torch.zeros(B, K, 2, device=device); off = torch.zeros(B, K, 2, device=device)
    cls = torch.zeros(B, K, dtype=torch.long, device=device)
    attrs = torch.zeros(B, K, 5, dtype=torch.long, device=device)
    r = max_win // 2
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1, device=device), torch.arange(-r, r + 1, device=device), indexing="ij")
    flat = hm.view(-1)
    for b, boxes in enumerate(boxes_list):
        n = len(boxes)
        if n == 0:
            continue
        bx = boxes.to(device)
        c = bx[:, 0].long()
        x0, y0, x1, y1 = bx[:, 1] / stride, bx[:, 2] / stride, bx[:, 3] / stride, bx[:, 4] / stride
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = (x1 - x0).clamp(min=1e-3), (y1 - y0).clamp(min=1e-3)
        ci, cj = cy.floor().long().clamp(0, Hs - 1), cx.floor().long().clamp(0, Ws - 1)
        ind[b, :n] = ci * Ws + cj
        mask[b, :n] = True
        wh[b, :n, 0] = torch.log(w); wh[b, :n, 1] = torch.log(h)
        off[b, :n, 0] = cx - cj.float(); off[b, :n, 1] = cy - ci.float()
        cls[b, :n] = c
        attrs[b, :n] = bx[:, 5:10].long()
        sx = (w / 6).clamp(min=min_sigma); sy = (h / 6).clamp(min=min_sigma)
        # gaussian patches around every centre, max-merged into the class channel
        gy = ci[:, None, None] + dy[None]; gx = cj[:, None, None] + dx[None]          # [n, win, win]
        val = torch.exp(-((gx.float() - cx[:, None, None]) ** 2) / (2 * sx[:, None, None] ** 2)
                        - ((gy.float() - cy[:, None, None]) ** 2) / (2 * sy[:, None, None] ** 2))
        ok = (gy >= 0) & (gy < Hs) & (gx >= 0) & (gx < Ws) & (val > 1e-3)
        idx = ((b * n_cls + c)[:, None, None] * Hs + gy) * Ws + gx
        flat.scatter_reduce_(0, idx[ok], val[ok], reduce="amax")
        flat.view(B, n_cls, Hs, Ws)[b, c, ci, cj] = 1.0                       # the centre cell is exactly 1
    return {"hm": hm, "ind": ind, "mask": mask, "wh": wh, "off": off, "cls": cls, "attrs": attrs}


def render_family_hm(t, fam_of_cls: torch.Tensor, n_fam: int):
    """Family heatmap from the class heatmap: max over the classes of each family (one amax per family; a scatter
    over an expanded [B, C, H, W] index tensor cost 2 GB)."""
    B, C, Hs, Ws = t["hm"].shape
    out = torch.zeros(B, n_fam, Hs, Ws, device=t["hm"].device)
    fam = fam_of_cls.tolist()
    for f in range(n_fam):
        idx = [c for c in range(C) if fam[c] == f]
        if idx:
            out[:, f] = t["hm"][:, idx].amax(1)
    return out


def render_staff_map(staves_list, H, W, stride, device, sigma_px=None, sp_list=None):
    """[B, 1, H/s, W/s] soft staff-line map: for every polyline, a Gaussian across the line (sigma = a quarter staff
    space, floor 1.5 px) along its length. The line's y per cell column is interpolated in numpy for every polyline
    of the batch at once, moved to the device once, and the Gaussians are max-merged per sample in one op (the
    per-polyline version synced the GPU 800 times a step and made cnet-staff four times slower than its siblings)."""
    B = len(staves_list)
    Hs, Ws = H // stride, W // stride
    out = torch.zeros(B, 1, Hs, Ws, device=device)
    cols = np.arange(Ws, dtype=np.float32) * stride + stride / 2
    ys = torch.arange(Hs, device=device).float() * stride + stride / 2
    for b, polys in enumerate(staves_list):
        if not polys:
            continue
        sp = (sp_list[b] if sp_list else 0) or 10.0
        sigma = max(1.5, 0.25 * sp) if sigma_px is None else sigma_px
        rows = []
        for poly in polys:
            pts = np.asarray(poly, np.float32)
            if len(pts) < 2:
                continue
            xs, yl = pts[:, 0], pts[:, 1]
            if xs.max() <= 0 or xs.min() >= W:
                continue
            ly = np.interp(cols, xs, yl, left=np.nan, right=np.nan).astype(np.float32)   # NaN outside the line's span
            rows.append(ly)
        if not rows:
            continue
        line_y = torch.from_numpy(np.stack(rows)).to(device)                                 # [L, Ws]
        g = torch.exp(-((ys[None, :, None] - line_y[:, None, :]) ** 2) / (2 * sigma ** 2))   # [L, Hs, Ws], NaN off-span
        out[b, 0] = torch.nan_to_num(g, nan=0.0).amax(0)
    return out
