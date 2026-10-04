"""The v7 detector, loaded once, run over many pages (the training tokenizer and the reader both use it).

Output is one record per detection: cls, conf, xyxy, bbox, and for notes and rests their attributes (attrs,
attr_p, attr_dist), the format the reader caches as ``<stem>.dets_v7_<imgsz>.json`` beside a page. A page is scaled
so that its staff space lands at TARGET_SP (10.6 px) at the model's input.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .paths import detector

TARGET_SP = 10.6
IMGSZ_MIN, IMGSZ_MAX = 640, 4096


def imgsz_for(staff_space: float, long_side: int, target: float = TARGET_SP) -> int:
    if staff_space <= 0:
        return 2048
    return int(min(IMGSZ_MAX, max(IMGSZ_MIN, round(long_side * target / staff_space / 32) * 32)))


class Detector:
    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        """``weights``: a detector checkpoint; by default the one COPISTA_DETECTOR names (paths.detector)."""
        import torch
        weights = weights or detector()[0]

        from .detector.small import models as M
        from .detector.small.data import ATTR_OFFSETS, ATTR_SIZES
        from .detector.small.export import ATTR_VOCAB, ATTRS
        self.torch, self.M = torch, M
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(weights, map_location=self.device, weights_only=False)
        self.classes = ck["classes"]
        self.model = M.build(ck["arch"], self.classes).to(self.device).eval()
        self.model.load_state_dict(ck["ema"])
        self.attr_cls = [c.partition("_")[0] in M.ATTR_CLASS_PREFIXES for c in self.classes]
        self.slices = {a: (ATTR_OFFSETS[a], ATTR_SIZES[a]) for a in ATTRS}
        self.vocab = ATTR_VOCAB

    @staticmethod
    def prepare(img, imgsz: int):
        """PIL image -> (float array at the model's scale, scale factor). CPU work, safe in a loader process."""
        from PIL import Image
        im = img.convert("L")
        s = imgsz / max(im.width, im.height)
        w2, h2 = max(32, round(im.width * s)), max(32, round(im.height * s))
        return np.asarray(im.resize((w2, h2), Image.LANCZOS), dtype=np.float32) / 255.0, s

    def run_array(self, arr: np.ndarray, s: float, conf: float = 0.05, cand_conf: float | None = None):
        """Detections at ``conf``. With ``cand_conf`` also the detector's sub-threshold candidates (decoded
        separately at that threshold, so the detections at ``conf`` are exactly the usual ones) and the image
        features: returns (dets, cands, feats), feats = float16 [len(dets) + len(cands), FEAT_DIM], the FPN map
        at each box's centre."""
        torch = self.torch
        h2, w2 = arr.shape
        H, W = math.ceil(h2 / 32) * 32, math.ceil(w2 / 32) * 32
        x = torch.ones(1, 1, H, W, device=self.device)
        x[0, 0, :h2, :w2] = torch.from_numpy(arr).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            o = self.model(x)
        dets = self._records(self.M.decode(self.model, o, thresh=conf)[0], s)
        if cand_conf is None:
            return dets
        cands = [r for r in self._records(self.M.decode(self.model, o, thresh=cand_conf)[0], s) if r["conf"] < conf]
        boxes = [[v * s for v in r["xyxy"]] for r in dets + cands]
        return dets, cands, self.features(o["f"], boxes, H, W)

    def features(self, f, boxes, H: int, W: int) -> np.ndarray:
        """The FPN map bilinearly sampled at each box's centre (model-input pixels) -> float16 [n, C]."""
        torch = self.torch
        if not boxes:
            return np.zeros((0, f.shape[1]), np.float16)
        b = torch.tensor(boxes, dtype=torch.float32, device=f.device)
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        grid = torch.stack([cx / W * 2 - 1, cy / H * 2 - 1], -1)[None, None]           # 1, 1, n, 2
        v = torch.nn.functional.grid_sample(f.float(), grid, mode="bilinear", align_corners=False)
        return v[0, :, 0].T.cpu().numpy().astype(np.float16)

    def _records(self, d, s: float) -> list[dict]:
        logits = d["attrs"].float().cpu().numpy()
        dets = []
        for i, (b, sc, c) in enumerate(zip(d["boxes"].cpu().tolist(), d["scores"].cpu().tolist(),
                                           d["cls"].cpu().tolist())):
            b = [v / s for v in b]
            rec = {"cls": self.classes[c], "conf": round(sc, 3), "xyxy": [round(v, 1) for v in b],
                   "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]}
            if self.attr_cls[c]:
                attrs, attr_p, attr_dist = {}, {}, {}
                for a, (off, n) in self.slices.items():
                    lg = logits[i, off:off + n]
                    e = np.exp(lg - lg.max()); p = e / e.sum()
                    k = int(np.argmax(p)); voc = self.vocab[a]
                    attrs[a] = voc[k]; attr_p[a] = round(float(p[k]), 4)
                    if len(voc) <= 12:
                        attr_dist[a] = {voc[j]: round(float(p[j]), 4) for j in range(len(voc)) if p[j] >= 0.005}
                rec["attrs"], rec["attr_p"], rec["attr_dist"] = attrs, attr_p, attr_dist
            dets.append(rec)
        return dets

    def run(self, img_path: str | Path, staff_space: float, conf: float = 0.05) -> list[dict]:
        from PIL import Image
        im = Image.open(img_path)
        arr, s = self.prepare(im, imgsz_for(staff_space, max(im.width, im.height)))
        return self.run_array(arr, s, conf)
