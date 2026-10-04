"""Export the corpus for the small-architecture search: every page resized so its long side is ``--long`` px (the
detector's training frame, JPEG q92 -- ultralytics and the custom loops then decode a 1536-px image instead of a
3000-px one), YOLO txt labels on a fixed class list with rare-class bucketing, and one meta jsonl per split that
carries every box with its attribute ids and the staff-line polylines. One export feeds every experiment.

  python -m copista_evidence.detector.small.export <index.jsonl> <data_root> <out_dir> --taxonomy models/yolo_v6_taxonomy.json
                             [--min-count 20] [--long 1536] [--max-boxes 1600] [--workers 8]

Output layout::

  out_dir/data.yaml                     ultralytics dataset file (images/{train,val}, names = the class list)
  out_dir/taxonomy.json                 {"classes": [...266], "attr_vocab": {...}} -- Taxonomy.load()-compatible
  out_dir/images/{train,val}/NNNNNN.jpg
  out_dir/labels/{train,val}/NNNNNN.txt "cls cx cy w h" normalised (ultralytics)
  out_dir/meta/{train,val}.jsonl        {"id", "img", "w", "h", "sp", "src", "boxes": [[cls, x0, y0, x1, y1, pos, stem,
                                         dots, grace, voice], ...], "staves": [[[x, y], ...] per line, ...]}

Attribute vocabularies are compact and fixed (the corpus-built ones carried 196 staff positions, most seen once):
staff_position -24..24 (49 + <na>), stem_dir / dots / grace / voice_slot as in the deployed taxonomy. Only notes
and rests carry attributes; every other class is <na> throughout. The val split is by SOURCE score (crc32 of the
row's source), as copista_evidence.detector.det.yolo_export does, so a partial export and the full one agree on which pages are val.
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from ..det.taxonomy import Taxonomy, class_name, is_text
from ..lmx import remap_path

ATTR_CLASSES = ("note", "rest", "mRest", "multiRest")
POS_MAX = 24
ATTR_VOCAB = {
    "staff_position": ["<na>"] + [str(p) for p in range(-POS_MAX, POS_MAX + 1)],
    "stem_dir": ["<na>", "down", "up", "both"],      # both: one head, a stem each way (two voices)
    "dots": ["<na>", "1", "2", "3", "4"],
    "grace": ["<na>", "acc", "unacc"],
    "voice_slot": ["<na>", "1", "2", "3", "4", "5", "6", "7", "8"],
}
ATTRS = tuple(ATTR_VOCAB)
ATTR_ID = {a: {v: i for i, v in enumerate(vs)} for a, vs in ATTR_VOCAB.items()}


def encode_attrs(e: dict) -> list[int]:
    """[pos, stem, dots, grace, voice] ids for one element (all <na> unless it is a note / rest)."""
    if e.get("type") not in ATTR_CLASSES:
        return [0, 0, 0, 0, 0]
    a = e.get("attrs") or {}
    pos = e.get("staff_position")
    if pos is None:
        pid = 0
    else:
        pid = ATTR_ID["staff_position"][str(max(-POS_MAX, min(POS_MAX, int(pos))))]
    return [pid,
            ATTR_ID["stem_dir"].get(str(a.get("stem.dir")), 0),
            ATTR_ID["dots"].get(str(a.get("dots")), 0),
            ATTR_ID["grace"].get(str(a.get("grace")), 0),
            ATTR_ID["voice_slot"].get(str(e.get("voice_slot")), 0)]


def split_of(row: dict, i: int, val_every: int = 10) -> str:
    src = row.get("source")
    return "val" if (zlib.crc32(src.encode()) if src else i) % val_every == 0 else "train"


def _one(job):
    """Resize one page and build its label lines + meta row. Runs in a worker; returns None to skip."""
    i, row, data_root, out_dir, long_side, max_boxes, tax_json, bucket = job
    from PIL import Image
    tax = Taxonomy.load(tax_json)
    tax.bucket = bucket
    try:
        labels = json.loads(remap_path(row["labels"], data_root).read_text())
    except Exception:
        return None
    els = labels.get("elements", [])
    if max_boxes and sum(1 for e in els if "bbox" in e) > max_boxes:
        return ("skipped", i)
    W, H = labels["image"]["width"], labels["image"]["height"]
    s = long_side / max(W, H)
    w2, h2 = max(1, round(W * s)), max(1, round(H * s))
    split = split_of(row, i)
    stem = f"{i:06d}"
    dst = Path(out_dir) / "images" / split / f"{stem}.jpg"
    if not dst.exists():
        try:
            im = Image.open(remap_path(row["image"], data_root)).convert("RGB")
        except Exception:
            return None
        if (im.width, im.height) != (W, H):        # labels always describe the written image; trust them
            W, H = im.width, im.height
            s = long_side / max(W, H); w2, h2 = max(1, round(W * s)), max(1, round(H * s))
        im.resize((w2, h2), Image.LANCZOS).save(dst, quality=92, optimize=False)
    lines, boxes, dropped, bucketed = [], [], Counter(), Counter()
    for e in els:
        b = e.get("bbox")
        if not b or b[2] < 1 or b[3] < 1 or (tax.drop_text and is_text(e)):
            continue
        name = class_name(e, tax.merge_text)
        cid = tax.encode_class(e)
        if cid is None:
            dropped[name] += 1
            continue
        if name in tax.bucket:
            bucketed[name] += 1
        x, y, w, h = b
        x0, y0, x1, y1 = x * s, y * s, (x + w) * s, (y + h) * s
        lines.append(f"{cid} {(x0 + x1) / 2 / w2:.6f} {(y0 + y1) / 2 / h2:.6f} {(x1 - x0) / w2:.6f} {(y1 - y0) / h2:.6f}")
        boxes.append([cid, round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1), *encode_attrs(e)])
    staves = []
    for st in labels.get("staves") or []:
        polys = st.get("lines_poly")
        if not polys:
            x0, x1 = float(st.get("x0", 0)), float(st.get("x1", 0))
            polys = [[[x0, y], [x1, y]] for y in st.get("lines", [])]
        staves.append([[[round(px * s, 1), round(py * s, 1)] for px, py in poly] for poly in polys])
    sp = labels.get("staff_space")
    meta = {"id": stem, "img": f"images/{split}/{stem}.jpg", "w": w2, "h": h2, "sp": round(sp * s, 3) if sp else None,
            "src": row.get("source"), "boxes": boxes, "staves": staves}
    (Path(out_dir) / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
    return (split, meta, dropped, bucketed)


def export(index_jsonl, data_root, out_dir, tax: Taxonomy, *, long_side=1536, max_boxes=1600, workers=8):
    out = Path(out_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    tax_json = out / "taxonomy.json"
    # the file carries the taxonomy schema's full attribute set: ``place`` (articulations / dynamics) is not one of
    # the note attributes the models predict, but copista_evidence.detector.det.attr_stage's encoder walks every schema attribute
    tax_json.write_text(json.dumps({"classes": tax.classes, "attr_vocab": {**ATTR_VOCAB, "place": ["<na>", "above", "below"]}}))
    rows = [json.loads(l) for l in Path(index_jsonl).read_text().splitlines()]
    jobs = [(i, r, data_root, str(out), long_side, max_boxes, str(tax_json), dict(tax.bucket)) for i, r in enumerate(rows)]
    n = {"train": 0, "val": 0}; skipped = 0; nbox = 0
    dropped, bucketed = Counter(), Counter()
    fh = {s: (out / "meta" / f"{s}.jsonl").open("w") for s in ("train", "val")}
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_one, jobs, chunksize=8)):
            if res is None:
                continue
            if res[0] == "skipped":
                skipped += 1
                continue
            split, meta, d, b = res
            fh[split].write(json.dumps(meta) + "\n")
            n[split] += 1; nbox += len(meta["boxes"]); dropped.update(d); bucketed.update(b)
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)
    for f in fh.values():
        f.close()
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(tax.classes))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"exported train {n['train']} val {n['val']} pages, {nbox} boxes, {tax.num_classes} classes -> {out}"
          f" ({skipped} pages over {max_boxes} boxes skipped)")
    if bucketed:
        print(f"bucketed {sum(bucketed.values())} boxes of {len(bucketed)} rare classes")
    if dropped:
        print(f"DROPPED {sum(dropped.values())} boxes of {len(dropped)} classes not in the list: "
              + ", ".join(f"{c} x{k}" for c, k in dropped.most_common(12)))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("index"); ap.add_argument("data_root"); ap.add_argument("out_dir")
    ap.add_argument("--taxonomy", required=True, help="the fixed class list (a model's taxonomy.json)")
    ap.add_argument("--min-count", type=int, default=20, help="route classes rarer than this to their family _x bucket")
    ap.add_argument("--long", type=int, default=1536, help="long side of the exported page")
    ap.add_argument("--max-boxes", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    tax = Taxonomy.load(a.taxonomy)
    counts, _ = Taxonomy.count_classes(a.index, data_root=a.data_root, merge_text=tax.merge_text, drop_text=tax.drop_text)
    print(f"fixed taxonomy {a.taxonomy}: {tax.num_classes} classes; corpus has {len(counts)} class names, "
          f"{sum(k for c, k in counts.items() if c not in tax.cls2id)} boxes outside the list")
    if a.min_count:
        b, unb = tax.bucket_rare(counts, a.min_count)
        print(f"min-count {a.min_count}: {len(b)} rare classes bucketed; {len(unb)} rare classes have no _x bucket "
              f"and stay: {', '.join(unb) or '-'}")
    export(a.index, a.data_root, a.out_dir, tax, long_side=a.long, max_boxes=a.max_boxes, workers=a.workers)


if __name__ == "__main__":
    main()
