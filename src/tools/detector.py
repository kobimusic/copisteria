"""Run the copista symbol detector over one page image and write its detections as a JSON file.

    python src/tools/detector.py page.png --out detections.json

The page is read the way the reader reads it: the staff space from the ink, checked against the measure boxes of a
first read at 2048 px, then a run at the size that puts that staff space at 10.6 px (this page: 1760 px). Nothing
is cached beside the page; the detections go only to ``--out``, whose path is checked (not a directory, its
parent exists) before the detector runs. The detector in use is the default copista-28m, or the one
COPISTERIA_DETECTOR names (see paths.detector); a run of the test page gives these records.

Output format: a JSON object with two keys:

  detection    a list, one record per detection at confidence 0.05 and up (630 over the test page, a 1272 x 1759
               px Galuppi scan), in the detector's own order (not reading order). Every record has:
  input_path   the page the detections came from, absolute (so the file says what it is a reading of).

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

The ``detection`` list alone is the format the reader caches beside a page (``<stem>.dets_v7_<imgsz>.json``), which
holds the list un-wrapped; saving this file's list under that name, from the same detector and size, lets
copisteria read from it. ``src/tools/visualizer.py`` draws these boxes over the page in the browser.
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
from copisteria.vocab import family


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the copista symbol detector over a page image and write its "
                                             "detections as JSON (the format of the reader's "
                                             "<stem>.dets_v7_<imgsz>.json cache).")
    ap.add_argument("image", help="the page to run the detector over (png)")
    ap.add_argument("--out", "-o", required=True, help="where to write the detections JSON")
    a = ap.parse_args(argv)

    png = Path(a.image)
    out = Path(a.out)
    # the detector run is the slow part: fail on a bad output path before it, not after
    if out.is_dir():
        sys.exit(f"{out}: the output is a directory")
    if not out.parent.is_dir():
        sys.exit(f"{out.parent}: no such directory for the output file")

    dets, imgsz = detect(png)
    out.write_text(json.dumps({"detection": dets, "input_path": str(png.resolve())}, indent=2))
    fams = len({family(r["cls"]) for r in dets})
    print(f"{png.name}: {len(dets)} detections in {fams} families (read at {imgsz} px) -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
