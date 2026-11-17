"""Run the copista symbol detector over one page image and write its detections as a JSON file, and with
``--add-png-output`` also as a PNG of the page with every box drawn on it.

    python src/tools/detector_visualiser.py page.png --out detections.json
    python src/tools/detector_visualiser.py page.png --out detections.json --add-png-output --png overlay.png --width 4

The page is read the way the reader reads it: the staff space from the ink, checked against the measure boxes of a
first read at 2048 px, then a run at the size that puts that staff space at 10.6 px (this page: 1760 px). Nothing
is cached beside the page; the detections go only to the files named on the command line (the PNG is ``--png``
where that is given, else ``--out`` with a .png suffix), whose paths are checked (not directories, the parents
exist) before the detector runs. The detector in use is the default copista-28m, or the one
COPISTERIA_DETECTOR names (see paths.detector); a run of the test page gives these records.

Output format: a JSON list, one record per detection at confidence 0.05 and up (630 over the test page, a 1272 x
1759 px Galuppi scan), in the detector's own order (not reading order). Every record has:

  cls    the class name, from the v7 taxonomy (models/yolo_v7_taxonomy.json, 270 classes): "note_half",
         "rest_quarter", "beam", "barLine_single", "measure", "clef_G2", "keySig_2f", "accid_natural", ...
  conf   the detection confidence, 3 decimals (this page: 0.062 to 1.0).
  xyxy   the box [x0, y0, x1, y1] in the page's own pixels (the model input scaled back), 1 decimal.
  bbox   the same box as [x, y, w, h], unrounded.

Notes and rests (the classes whose family is note or rest) carry the detector's attribute readings too, over all
five vocabularies of the taxonomy; "<na>" means that attribute does not apply to this symbol:

  attrs       each attribute's argmax value: {"staff_position": "6", "stem_dir": "down", "dots": "<na>",
              "grace": "<na>", "voice_slot": "1"}
  attr_p      the probability of that argmax, 4 decimals: {"staff_position": 0.9998, ...}
  attr_dist   for the vocabularies of at most 12 values (all but staff_position), the whole distribution over
              its values with p >= 0.005: {"stem_dir": {"down": 0.9999}, "dots": {"<na>": 1.0}, ...}

A record of each kind, from the test run:

  {"cls": "accid_natural", "conf": 1.0, "xyxy": [358.7, 876.2, 367.3, 903.8],
   "bbox": [358.728129196167, 876.208353250677, 8.532136743718922, 27.629678136652274]}

  {"cls": "note_half", "conf": 0.999, "xyxy": [215.7, 636.0, 228.8, 647.8], "bbox": [...],
   "attrs": {"staff_position": "6", "stem_dir": "down", "dots": "<na>", "grace": "<na>", "voice_slot": "1"},
   "attr_p": {"staff_position": 0.9998, "stem_dir": 0.9999, "dots": 1.0, "grace": 1.0, "voice_slot": 0.9997},
   "attr_dist": {"stem_dir": {"down": 0.9999}, "dots": {"<na>": 1.0}, "grace": {"<na>": 1.0},
                 "voice_slot": {"1": 0.9997}}}

The PNG, written only when ``--add-png-output`` is given, is the page as it is, converted to RGB, with the box of
every detection drawn on it (xyxy, page pixels), ``--width`` page pixels thick (default 2). Nothing is written on
the boxes: class and confidence are in the JSON, record i is box i of the PNG. The colour is decided before any
page is read: ``COLORS`` at the top of this file maps all 270 classes of the v7 taxonomy
(copisteria.vocab.CLASSES, the same list models/yolo_v7_taxonomy.json gives the detector) to a colour, every class
wearing its family's, so a family reads the same colour on every page.

This is exactly the format the reader caches beside a page (``<stem>.dets_v7_<imgsz>.json``); a file written here
with the same detector and size can be dropped in next to the page and copisteria will read from it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import copisteria  # noqa: F401  (installed editable, per the README)
except ImportError:  # running from the repository without one
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from copisteria.detect import Detector, imgsz_for, measure_space, staff_space
from copisteria.vocab import CLASSES, family

# One colour per class family of the v7 taxonomy, spread over the colour wheel in the golden-ratio order and in two
# brightness levels so neighbours stay apart. Fixed by family name, so a family reads the same colour on every page.
FAM_COLORS = {
    "accid": "#f22424",
    "arpeg": "#113fad",
    "artic": "#9df224",
    "bTrem": "#ad119a",
    "barLine": "#24f2d0",
    "beam": "#ad6611",
    "breath": "#5824f2",
    "caesura": "#18ad11",
    "clef": "#f22469",
    "curve": "#1173ad",
    "dots": "#e1f224",
    "dynam": "#8d11ad",
    "ending": "#24f28b",
    "fTrem": "#ad3211",
    "fermata": "#2436f2",
    "flag": "#4cad11",
    "gliss": "#f224ae",
    "grpSym": "#11a7ad",
    "hairpin": "#f2be24",
    "keySig": "#5811ad",
    "mRest": "#24f246",
    "mRpt": "#ad1125",
    "measure": "#247bf2",
    "meterSig": "#80ad11",
    "mordent": "#f224f2",
    "multiRest": "#11ad7f",
    "multiRpt": "#f27924",
    "note": "#2411ad",
    "octave": "#47f224",
    "pedal": "#ad1159",
    "repeatMark": "#24c0f2",
    "rest": "#ada611",
    "spanEnd": "#ad24f2",
    "trill": "#11ad4b",
    "tuplet": "#f23524",
    "tupletBracket": "#1133ad",
    "tupletNum": "#8cf224",
}

# class -> colour for all 270 v7 classes, taken from the taxonomy before any page is read: a class wears its
# family's colour, so the two readings of one glyph (say a sharp and its accidental) share it.
COLORS = {c: FAM_COLORS[family(c)] for c in CLASSES}


def detect(png: Path):
    """The page's detections, exactly as the reader's cache holds them, plus the input size they were read at.

    Same reading as ``copisteria.pipeline.detections`` on an uncached page: the staff space from the ink, checked
    against a first read at 2048 px, the page re-read at the size that puts that staff space at 10.6 px."""
    from copisteria.vision.page.scan import from_png
    from copisteria.vision.page.staves import find_staves

    page = from_png(png)
    ink = find_staves(page.image()).spacing
    det = Detector()
    im = Image.open(png)
    arr, s = det.prepare(im, 2048)
    first = det.run_array(arr, s)
    imgsz = imgsz_for(staff_space(ink, measure_space(first)), max(im.width, im.height))
    if imgsz == 2048:
        return first, imgsz
    arr, s = det.prepare(im, imgsz)
    return det.run_array(arr, s), imgsz


def overlay(png: Path, dets: list[dict], out: Path, width: int):
    """The page with every detection's box drawn on it, ``width`` page pixels thick, each in its class's colour."""
    from PIL import ImageDraw

    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    for r in dets:
        d.rectangle(r["xyxy"], outline=COLORS[r["cls"]], width=width)
    im.save(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the copista symbol detector over a page image and write its "
                                             "detections as JSON (the format of the reader's "
                                             "<stem>.dets_v7_<imgsz>.json cache).")
    ap.add_argument("image", help="the page to run the detector over (png)")
    ap.add_argument("--out", "-o", required=True, help="where to write the detections JSON")
    ap.add_argument("--add-png-output", action="store_true",
                    help="also write a PNG of the page with every box drawn on it")
    ap.add_argument("--png", help="with --add-png-output: where to write it (default: --out with a .png suffix)")
    ap.add_argument("--width", type=int, default=2, help="bounding box thickness in page pixels (default 2)")
    a = ap.parse_args(argv)

    png = Path(a.image)
    out = Path(a.out)
    if a.png and not a.add_png_output:
        sys.exit("--png names the overlay file: ask for the overlay with --add-png-output too")
    dest = None
    if a.add_png_output:
        dest = Path(a.png) if a.png else out.with_suffix(".png")
        if dest == out:                   # --out names a .png: the two files would be the one
            dest = out.with_name(out.stem + "_boxes.png")

    # the detector run is the slow part: fail on a bad output path before it, not after
    for p in [out] + ([dest] if dest else []):
        if p.is_dir():
            sys.exit(f"{p}: an output is a directory")
        if not p.parent.is_dir():
            sys.exit(f"{p.parent}: no such directory for an output file")

    dets, imgsz = detect(png)
    out.write_text(json.dumps(dets, indent=2))
    wrote = [str(out)]
    if dest:
        overlay(png, dets, dest, a.width)
        wrote.append(str(dest))
    fams = len({family(r["cls"]) for r in dets})
    print(f"{png.name}: {len(dets)} detections in {fams} families" + (f", boxes {a.width} px" if dest else "")
          + f" (read at {imgsz} px) -> {', '.join(wrote)}", file=sys.stderr)


if __name__ == "__main__":
    main()
