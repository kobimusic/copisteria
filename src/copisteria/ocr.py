"""A page's text boxes from Tesseract: page image -> ``<stem>.texts.json`` beside it, the file the pipeline reads.

  python -m copisteria.ocr page.png [more pages ...] [--force]

Tesseract (the ``tesseract`` command, sparse-text mode) reads the words; their roles come from where they sit against
the page's staves, found in the ink (or, where the ink gives none, from the measure boxes of the page's v7
detections, cached beside the page or read here):

* above the first staff: the page head -- the tallest line the title, other centred lines the subtitle, lines right
  of centre the composer, the rest ``pgHead``;
* left of a staff and level with it: ``label`` (a part's name);
* under a staff (nearer its bottom than the next staff's top): ``syl``, one box per word, hyphens kept as "-";
* over a staff: ``dir``;
* far below the last staff: ``pgFoot``.

Words inside a staff's band are the music's glyphs read as letters and are dropped, as are low-confidence words and
words with no letter or digit. Words of one line with the same role (other than ``syl``) are joined into one box.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

MIN_CONF = 50                       # Tesseract's word confidence (0-100) below which a word is noise
HEAD_GAP_SP = 6.0                   # page head: text whose centre is this many staff spaces over the first staff
FOOT_GAP_SP = 10.0                  # page foot: this many staff spaces under the last staff
BAND_SP = (0.5, 1.0)                # a staff's band reaches this far over / under its lines: words there are glyphs
_WORDLIKE = re.compile(r"[^\W_]")


def tesseract(png: Path, lang: str = "eng") -> list[dict]:
    """The page's words as Tesseract reads them in sparse-text mode: text, xyxy, conf."""
    tsv = subprocess.run(["tesseract", str(png), "-", "-l", lang, "--psm", "11", "tsv"],
                         capture_output=True, text=True, check=True).stdout
    words = []
    for r in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (r.get("text") or "").strip().strip("_—")
        if r["level"] != "5" or not text:
            continue
        x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
        words.append({"text": text, "xyxy": [x, y, x + w, y + h], "conf": float(r["conf"]),
                      "line": (r["block_num"], r["par_num"], r["line_num"])})
    return words


def staves(png: Path, dets: list[dict]) -> list[dict]:
    """The page's staves, top to bottom: top, bottom, left, and the staff space. From the ink's staff lines (where
    a staff starts: the detector can miss a system's first measure), else from the measure boxes."""
    from .vision.page.scan import from_png
    from .vision.page.staves import find_staves
    ink = find_staves(from_png(png).image()).staves
    if ink:
        return [{"top": s.top, "bottom": s.bottom, "left": s.x0, "sp": s.spacing} for s in ink]
    boxes = sorted((d["xyxy"] for d in dets if d["cls"] == "measure"), key=lambda b: b[1])
    rows: list[list] = []
    for b in boxes:
        row = rows[-1] if rows else None
        if row and b[1] < statistics.median(r[3] for r in row) and b[3] > statistics.median(r[1] for r in row):
            row.append(b)
        else:
            rows.append([b])
    out = []
    for row in rows:
        top, bot = statistics.median(b[1] for b in row), statistics.median(b[3] for b in row)
        out.append({"top": top, "bottom": bot, "left": min(b[0] for b in row), "sp": (bot - top) / 4})
    return out


def role(word: dict, sts: list[dict]) -> str | None:
    """The word's role from where it sits against the staves (None: a glyph read as a letter)."""
    x0, y0, x1, y1 = word["xyxy"]
    cy = (y0 + y1) / 2
    sp = statistics.median(s["sp"] for s in sts)
    if cy < sts[0]["top"] - HEAD_GAP_SP * sp:
        return "pgHead"
    if cy > sts[-1]["bottom"] + FOOT_GAP_SP * sp:
        return "pgFoot"
    # the staff the word belongs to: the nearest staff edge (a staff's lyric line sits nearer its bottom than the
    # next staff's top)
    near = min(sts, key=lambda s: 0 if s["top"] <= cy <= s["bottom"] else min(abs(cy - s["top"]), abs(cy - s["bottom"])))
    level = near["top"] - near["sp"] <= cy <= near["bottom"] + near["sp"]
    if x1 < near["left"] + near["sp"]:
        return "label" if level else "dir"
    if near["top"] - BAND_SP[0] * near["sp"] <= cy <= near["bottom"] + BAND_SP[1] * near["sp"]:
        return None
    return "syl" if cy > near["bottom"] else "dir"


def head_roles(texts: list[dict], width: int) -> None:
    """The page head's lines: the tallest the title, other centred lines the subtitle, lines right of centre the
    composer (lyricist, arranger), the rest left as ``pgHead``."""
    head = [t for t in texts if t["role"] == "pgHead"]
    if not head:
        return
    title = max(head, key=lambda t: t["xyxy"][3] - t["xyxy"][1])
    for t in head:
        x0, _, x1, _ = t["xyxy"]
        if t is title:
            t["role"] = "pgHead_title"
        elif x0 > width * 0.55:
            t["role"] = "pgHead_composer"
        elif abs((x0 + x1) / 2 - width / 2) < width * 0.15:
            t["role"] = "pgHead_subtitle"


def lines(words: list[dict]) -> list[dict]:
    """Words of one Tesseract line and one role joined, left to right, into one box (lyric syllables stay apart)."""
    out: list[dict] = []
    for w in sorted(words, key=lambda w: (w["line"], w["xyxy"][0])):
        prev = out[-1] if out and out[-1]["line"] == w["line"] and out[-1]["role"] == w["role"] != "syl" else None
        if prev:
            prev["text"] += " " + w["text"]
            prev["xyxy"] = [min(prev["xyxy"][0], w["xyxy"][0]), min(prev["xyxy"][1], w["xyxy"][1]),
                            max(prev["xyxy"][2], w["xyxy"][2]), max(prev["xyxy"][3], w["xyxy"][3])]
        else:
            out.append(dict(w))
    return out


def page_texts(png: Path, dets: list[dict], lang: str = "eng") -> list[dict]:
    """The page's text boxes with their roles, in reading order."""
    from PIL import Image
    with Image.open(png) as im:
        width = im.size[0]
    sts = staves(png, dets)
    words = []
    for w in tesseract(png, lang):
        if w["conf"] < MIN_CONF or not (_WORDLIKE.search(w["text"]) or w["text"] == "-"):
            continue
        r = role(w, sts) if sts else "words"
        if r is None or (w["text"] == "-" and r != "syl"):
            continue
        words.append({"text": w["text"], "xyxy": w["xyxy"], "role": r, "line": w["line"]})
    texts = [{k: t[k] for k in ("text", "xyxy", "role")} for t in lines(words)]
    head_roles(texts, width)
    return sorted(texts, key=lambda t: (t["xyxy"][1], t["xyxy"][0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", type=Path, help="page images (PNG)")
    ap.add_argument("--lang", default="eng", help="Tesseract's language(s), e.g. eng+por")
    ap.add_argument("--force", action="store_true", help="read pages that already have a texts.json again")
    a = ap.parse_args(argv)
    from .pipeline import detections
    for png in a.pages:
        dest = png.with_name(f"{png.stem}.texts.json")
        if dest.exists() and not a.force:
            print(f"{png.name}: {dest.name} exists (--force reads it again)", file=sys.stderr)
            continue
        texts = page_texts(png, detections(png), a.lang)
        dest.write_text(json.dumps(texts, ensure_ascii=False, indent=1))
        n = sum(t["role"] == "syl" for t in texts)
        print(f"{png.name}: {len(texts)} texts, {n} lyric syllables -> {dest}")


if __name__ == "__main__":
    main()
