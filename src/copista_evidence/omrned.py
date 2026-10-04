"""The reader over one OMR-NED set: every page image -> MusicXML in <set dir>/preds/<name>/<id>.musicxml.

A set dir holds index.json ({"ids": [...]}), images/<id>.png and, optionally, the pages' OCR text boxes in
<set dir>/texts/<id>.texts.json. A page's v7 detections are cached beside its image. Pages already written are
skipped (resumable); a page that fails gets no file (it then scores as read entirely wrong) and its error goes to
run.json. The MusicXML is scored against the set's ground truth with musicdiff (see README, Benchmark).

    python -m copista_evidence.omrned <set dir> --name evidence [--model models/evidence-2m.pt+models/evidence-refine-3m.pt] [--jobs 6]
    python -m copista_evidence.omrned <set dir> --name copista-2m --detector 2m
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

from .paths import DETECTORS, EVIDENCE

_M: dict = {}


def _init(model_path: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from copista_evidence import read
    _M["model"] = read.load_model(model_path)


def _one(job):
    png, texts_path, out = job
    t0 = time.time()
    try:
        from copista_evidence.pipeline import transcribe
        texts = json.loads(Path(texts_path).read_text()) if texts_path and Path(texts_path).exists() else None
        Path(out).write_text(transcribe(Path(png), _M["model"], texts))
        return {"id": Path(png).stem, "status": 0, "seconds": round(time.time() - t0, 2)}
    except Exception:
        return {"id": Path(png).stem, "status": 1, "seconds": round(time.time() - t0, 2),
                "error": traceback.format_exc(limit=4)[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("--model", default="+".join(str(p) for p in EVIDENCE),
                    help="evidence model checkpoint(s); several joined by + read as an ensemble")
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-texts", action="store_true")
    ap.add_argument("--texts", default="texts", help="the folder (in the set dir) of the pages' OCR text boxes")
    ap.add_argument("--detector", choices=sorted(DETECTORS), help="the symbol detector: 28m (default) or 2m (or COPISTA_DETECTOR)")
    a = ap.parse_args()
    if a.detector:
        os.environ["COPISTA_DETECTOR"] = a.detector      # before the pool: its workers inherit it
    d = Path(a.set_dir)
    ids = json.loads((d / "index.json").read_text())["ids"][:a.limit]
    out = d / "preds" / a.name
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d / "images" / f"{i}.png"), None if a.no_texts else str(d / a.texts / f"{i}.texts.json"),
             str(out / f"{i}.musicxml")) for i in ids if not (out / f"{i}.musicxml").exists()]
    t0 = time.time()
    _init(a.model)                       # fail here, not in a pool that respawns workers whose start-up fails
    if getattr(_M["model"], "feat_dim", 0):
        # image features: v7 over every page once, here on the GPU, before the CPU pool reads them
        from copista_evidence.detect import Detector
        from copista_evidence.pipeline import featcache
        det = Detector()
        for png, _, _ in jobs:
            featcache(Path(png), det)
        print(f"features cached ({time.time() - t0:.0f}s)", flush=True)
    with Pool(a.jobs, initializer=_init, initargs=(a.model,)) as pool:
        res = list(pool.imap_unordered(_one, jobs, chunksize=1))
    bad = [r for r in res if r["status"]]
    (out / "run.json").write_text(json.dumps({"model": a.model, "pages": len(ids), "run": len(jobs), "failed": len(bad),
                                              "seconds": round(time.time() - t0), "pages_detail": res}, indent=1))
    print(f"{d.name}: {len(jobs)} pages read, {len(bad)} failed, {time.time() - t0:.0f}s", flush=True)
    for r in bad[:3]:
        print(r["id"], r["error"], file=sys.stderr)


if __name__ == "__main__":
    main()
