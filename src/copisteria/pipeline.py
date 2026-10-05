"""Read pages end to end: page image -> v7 detections (cached) -> front end -> evidence model -> MusicXML + viewer.

  python -m copisteria.pipeline page.png [more pages ...] --out out
  python -m copisteria.pipeline --pdf score.pdf --range 1-4 --out out

A page's detections are taken from a cached file (``<stem>.dets_<tag>_<imgsz>.json`` beside the page, tag v7 for the
default detector) when one exists, else the detector runs here at the size the page's staff space asks for, and is
cached the same way. ``--detector 2m`` (or COPISTERIA_DETECTOR=2m) reads as copisteria-2m, with the 1.94M-parameter detector. Page texts
(``<stem>.texts.json`` beside the page: OCR text boxes, see docs/ARCHITECTURE.md) are used when present: the
title, the composer, words and multi-measure rest counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import read, view, write
from .revise import read_revised

from .paths import DETECTORS, EVIDENCE, REPO as ROOT, detector


def detections(png: Path, want_size: bool = False):
    """The page's v7 detections (``want_size``: and the input size they were read at)."""
    tag = detector()[1]
    cached = sorted(png.parent.glob(f"{png.stem}.dets_{tag}_*.json"))
    size = lambda path: int(path.stem.rsplit("_", 1)[1])                               # noqa: E731
    if len(cached) == 1:
        d = json.loads(cached[0].read_text())
        return (d, size(cached[0])) if want_size else d
    if cached:                       # several sizes cached: the one whose staff space sat nearest the detector's
        from PIL import Image
        from .detect import TARGET_SP
        with Image.open(png) as im:
            long_side = max(im.size)

        def gap(path):
            d = json.loads(path.read_text())
            hs = sorted(x["xyxy"][3] - x["xyxy"][1] for x in d if x["cls"] == "measure" and x["conf"] >= 0.5)
            if not hs:
                return 1e9, d, size(path)
            return abs(hs[len(hs) // 2] / 4 * size(path) / long_side - TARGET_SP), d, size(path)

        _, d, imgsz = min((gap(p) for p in cached), key=lambda t: t[0])
        return (d, imgsz) if want_size else d
    from .vision.page.scan import from_png
    from .vision.page.staves import find_staves

    from .detect import Detector, imgsz_for
    page = from_png(png)
    sp = find_staves(page.image()).spacing
    det = Detector()
    from PIL import Image
    im = Image.open(png)
    if sp <= 0:
        # the line finder found no staves (a small or yellowed scan): a first read at the default size, and the
        # staff space from its measure boxes (their height is four staff spaces)
        arr, s = det.prepare(im, 2048)
        first = det.run_array(arr, s)
        hs = sorted(d["xyxy"][3] - d["xyxy"][1] for d in first if d["cls"] == "measure" and d["conf"] >= 0.5)
        sp = hs[len(hs) // 2] / 4 if hs else 0.0
    imgsz = imgsz_for(sp, max(im.width, im.height))
    arr, s = det.prepare(im, imgsz)
    dets = det.run_array(arr, s)
    png.with_name(f"{png.stem}.dets_{tag}_{imgsz}.json").write_text(json.dumps(dets))
    return (dets, imgsz) if want_size else dets


_DET: list = []


def featcache(png: Path, det=None) -> Path:
    """v7 once more over the page at the size its detections were read at, keeping the sub-threshold candidates
    and the image features (Detector.run_array with candidates): ``<stem>.feat_<tag>_<imgsz>.npz`` beside the page."""
    import numpy as np
    from PIL import Image

    from .detect import Detector
    _, imgsz = detections(png, want_size=True)
    out = png.with_name(f"{png.stem}.feat_{detector()[1]}_{imgsz}.npz")
    if out.exists():
        return out
    if det is None:
        if not _DET:
            _DET.append(Detector())
        det = _DET[0]
    arr, s = det.prepare(Image.open(png), imgsz)
    dets, cands, feats = det.run_array(arr, s, cand_conf=0.01)
    np.savez_compressed(out, dets=np.array(json.dumps(dets)), cands=np.array(json.dumps(cands)), feat=feats)
    return out


def inputs(png: Path, model=None):
    """(dets, cands, feats) for a page: from its feature cache when the model reads image features (made here if
    missing), else the cached detections alone."""
    if model is not None and getattr(model, "feat_dim", 0):
        import numpy as np
        z = np.load(featcache(png))
        return json.loads(str(z["dets"])), json.loads(str(z["cands"])), z["feat"]
    return detections(png), None, None


def transcribe(png: Path, model, texts: list | None = None, device="cpu") -> str:
    """The page's MusicXML, nothing else written."""
    from PIL import Image
    dets, cands, feats = inputs(png, model)
    im = Image.open(png).convert("L")
    rd = read_revised(dets, im.width, im.height, model, device=device, attention=False, image=im, cands=cands,
                      feats=feats)
    return write.write(rd, texts=texts, image=im)


def run_page(png: Path, model, out: Path, device="cpu") -> dict:
    from PIL import Image
    dets, cands, feats = inputs(png, model)
    im = Image.open(png).convert("L")
    w, h = im.size
    tp = png.with_name(f"{png.stem}.texts.json")
    texts = json.loads(tp.read_text()) if tp.exists() else None
    rd = read_revised(dets, w, h, model, device=device, image=im, cands=cands, feats=feats)
    xml = write.write(rd, texts=texts, image=im)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{png.stem}.musicxml").write_text(xml)
    (out / f"{png.stem}.html").write_text(view.page_html(rd, png, xml, png.stem))
    ch = read.changes(rd)
    (out / f"{png.stem}.reading.json").write_text(json.dumps(
        {"tokens": {str(k): v for k, v in rd.tok.items()}, "changes": ch,
         "staves": len(rd.layout.staves), "systems": len(rd.layout.systems)}, default=str))
    return {"page": png.stem, "symbols": len(rd.tok), "changes": len(ch), "staves": len(rd.layout.staves),
            "systems": len(rd.layout.systems)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*")
    ap.add_argument("--pdf")
    ap.add_argument("--range", default="1")
    ap.add_argument("--model", default=str(EVIDENCE))
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--detector", choices=sorted(DETECTORS), help="the symbol detector: 28m (default) or 2m (or COPISTERIA_DETECTOR)")
    a = ap.parse_args(argv)
    if a.detector:
        import os
        os.environ["COPISTERIA_DETECTOR"] = a.detector
    pngs = [Path(p) for p in a.pages]
    if a.pdf:
        from .vision.page.scan import from_pdf
        lo, _, hi = a.range.partition("-")
        for n in range(int(lo), int(hi or lo) + 1):
            pngs.append(from_pdf(a.pdf, n, ROOT / "out" / "pages").path)
    model = read.load_model(a.model, a.device)
    rows = []
    for png in pngs:
        res = run_page(png, model, Path(a.out), a.device)
        rows.append(res)
        print(json.dumps(res), flush=True)
    index(Path(a.out), rows, a.model)


def index(out: Path, rows: list[dict], model: str) -> None:
    """out/index.html: one line per page read in this run, linking its viewer."""
    import html
    old = out / "index.json"
    prev = {r["page"]: r for r in json.loads(old.read_text())} if old.exists() else {}
    prev.update({r["page"]: r for r in rows})
    old.write_text(json.dumps(list(prev.values()), indent=1))
    trs = "".join(f'<tr><td><a href="{html.escape(p)}.html">{html.escape(p)}</a></td><td>{r["staves"]}</td>'
                  f'<td>{r["systems"]}</td><td>{r["symbols"]}</td><td>{r["changes"]}</td>'
                  f'<td><a href="{html.escape(p)}.musicxml">musicxml</a></td></tr>' for p, r in sorted(prev.items()))
    (out / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>copisteria pages</title><style>body{{font:14px system-ui;'
        f'margin:20px}}td,th{{padding:3px 10px;border-bottom:1px solid #ddd;text-align:left}}</style>'
        f'<h2>copisteria &mdash; {html.escape(Path(model).name)}</h2><table><tr><th>page</th><th>staves</th>'
        f'<th>systems</th><th>symbols</th><th>readings changed by context</th><th></th></tr>{trs}</table>')


if __name__ == "__main__":
    sys.exit(main())
