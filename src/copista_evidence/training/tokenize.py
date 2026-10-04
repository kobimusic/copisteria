"""Training data: rendered pages -> the reader's tokens and targets.

The input is a folder of tranches, each a folder of rendered pages with an ``index.jsonl`` (one line per page:
{"image": <page image>, "labels": <labels JSON>}; docs/ARCHITECTURE.md describes the labels). For every page v7
runs on the GPU and leaves its detections (``.dets.json``), its sub-threshold candidates (``.cands.json``) and the
image features of both (``.feat.npy``) beside the page; then the front end lays the page out and match.targets
labels its symbols, and the arrays go to ``<out>/<tranche>/<page>.npz``, which train.py reads.

  python -m copista_evidence.training.tokenize data/pages data/tok [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def _rows(tranche: Path) -> list[dict]:
    idx = tranche / "index.jsonl"
    return [json.loads(line) for line in idx.read_text().splitlines()] if idx.exists() else []


class _Pages:
    """torch Dataset: label staff space -> imgsz -> the page at the model's scale (CPU side)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        from PIL import Image

        from copista_evidence.detect import Detector, imgsz_for
        r = self.rows[k]
        try:
            lab = json.loads(Path(r["labels"]).read_text())
            im = Image.open(r["image"])
            arr, s = Detector.prepare(im, imgsz_for(float(lab.get("staff_space") or 0), max(im.width, im.height)))
            return k, arr, s
        except Exception:
            return k, None, 0.0


CAND_CONF = 0.01


def detect(rows: list[dict], det) -> int:
    """v7 over every page: detections (.dets.json), sub-threshold candidates (.cands.json) and the image
    features of both (.feat.npy, float16)."""
    import torch
    todo = [r for r in rows if not Path(r["image"]).with_suffix(".feat.npy").exists()]
    if not todo:
        return 0
    dl = torch.utils.data.DataLoader(_Pages(todo), batch_size=None, num_workers=4, prefetch_factor=4)
    n = 0
    for k, arr, s in dl:
        if arr is None:
            continue
        dets, cands, feats = det.run_array(np.asarray(arr), float(s), cand_conf=CAND_CONF)
        img = Path(todo[k]["image"])
        img.with_suffix(".dets.json").write_text(json.dumps(dets))
        img.with_suffix(".cands.json").write_text(json.dumps(cands))
        np.save(img.with_suffix(".feat.npy"), feats)
        n += 1
    return n


def _tok_one(job):
    r, out = job
    try:
        from copista_evidence import features, front, match
        from PIL import Image
        lab = json.loads(Path(r["labels"]).read_text())
        img = Path(r["image"])
        dets = json.loads(img.with_suffix(".dets.json").read_text())
        cands = json.loads(img.with_suffix(".cands.json").read_text()) if img.with_suffix(".cands.json").exists() else None
        feats = np.load(img.with_suffix(".feat.npy")) if img.with_suffix(".feat.npy").exists() else None
        L = front.build(dets, lab["image"]["width"], lab["image"]["height"],
                        image=Image.open(r["image"]).convert("L"), cands=cands)
        if not L.staves:
            return "empty"
        a = features.page_arrays(L, feats)
        if len(a["sym"]) == 0:
            return "empty"
        T = match.targets(L, lab)
        a.update(features.target_arrays(a, T))
        # structure check against the labels: staves / systems found vs drawn
        gst = {(s["part"], s["staff"], s["system"]) for s in lab.get("staves", [])}
        a["meta"] = np.array(json.dumps({"source": lab.get("source", ""), "image": r["image"],
                                         "staves": len(L.staves), "gt_staves": len(gst),
                                         "systems": len(L.systems),
                                         "gt_systems": len({s["system"] for s in lab.get("staves", [])}),
                                         "n_gt": T["n_gt"], "n_matched": T["n_matched"], "sp": L.sp}))
        np.savez_compressed(out, **a)
        return "ok"
    except Exception:
        return "error: " + traceback.format_exc(limit=3)


def tokenize(out_dir: Path, rows: list[dict], pool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in rows:
        stem = Path(r["image"]).stem
        out = out_dir / f"{stem}.npz"
        if not out.exists() and Path(r["image"]).with_suffix(".feat.npy").exists():
            jobs.append((r, str(out)))
    stats: dict = {}
    for res in pool.imap_unordered(_tok_one, jobs, chunksize=4):
        key = res if not res.startswith("error") else "error"
        stats[key] = stats.get(key, 0) + 1
        if key == "error" and stats[key] <= 3:
            print(res, flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", help="a folder of tranches of rendered pages (each with an index.jsonl)")
    ap.add_argument("out", help="where the token files go, one folder per tranche")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    from copista_evidence.detect import Detector
    det = Detector()
    with Pool(a.workers) as pool:
        for tranche in sorted(d for d in Path(a.pages).iterdir() if (d / "index.jsonl").exists()):
            rows = _rows(tranche)
            n = detect(rows, det)
            stats = tokenize(Path(a.out) / tranche.name, rows, pool)
            print(f"{tranche.name}: {len(rows)} pages, detected {n}, tokens {stats}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
