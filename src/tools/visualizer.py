"""Serve a detections JSON as a web page: the scanned image with every box overlaid, and a checkbox per class.

    python src/tools/visualizer.py detections.json [--port 5000]

Reads the file written by ``src/tools/detector.py``: the ``detection`` list of records and the ``input_path`` of
the page they came from (``--image`` overrides it when the page lives elsewhere; a bare reader-cache list, without
the wrapper, works too). Opens a flask server on 127.0.0.1 that shows the page with the box of every detection
drawn over it, in its class's colour, record i of the file being box i of the page, and opens it in the default
browser (webbrowser, from the standard library: nothing new to install). Hovering a box shows the class and the
detection confidence beside it. A side column lists the classes the file holds, one checkbox each (all checked to
start): only the boxes of the checked classes are drawn, so the overlay can be read class by class;
``show all`` / ``hide all`` toggle them together.

The web source is not in this file: ``templates/visualizer.html`` is the markup, rendered with the detections as
``DATA`` by flask's template engine, and ``static/visualizer.css`` and ``static/visualizer.js`` are the style and
the drawing, hover and checkbox logic (both beside this file, in the repository's ``src/tools/``). A box is drawn
in its class's colour, from ``COLORS`` at the top of this file.

Needs flask: pip install -e ".[tools]".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import copisteria  # noqa: F401  (installed editable, per the README)
except ImportError:  # running from the repository without one
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copisteria.vocab import CLASSES, family

HERE = Path(__file__).resolve().parent

# One colour per class family of the v7 taxonomy, spread over the colour wheel in the golden-ratio order and in
# two brightness levels so neighbours stay apart. Fixed by family name, so a family reads the same colour on
# every page and in every run of this visualizer.
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


def load(dets_json: Path):
    """(the detection records, the page image the file says it came from) of a detections JSON."""
    d = json.loads(dets_json.read_text())
    if isinstance(d, dict):
        return d["detection"], Path(d.get("input_path", ""))
    return d, None                                     # the bare reader-cache list: no path in it


def open_browser(port: int, timeout: float = 15.0):
    """Open the page in the default browser (stdlib webbrowser) once the server accepts connections on ``port``."""
    import socket
    import threading
    import time
    import webbrowser

    def wait():
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                with socket.create_connection(("127.0.0.1", port), 0.5):
                    break
            except OSError:
                time.sleep(0.2)
        webbrowser.open(f"http://127.0.0.1:{port}/")

    threading.Thread(target=wait, daemon=True).start()


def main(argv=None):
    try:
        from flask import Flask, render_template, send_file
    except ImportError:
        sys.exit('the visualizer needs flask: pip install -e ".[tools]"')

    ap = argparse.ArgumentParser(description="Show a detections JSON (src/tools/detector.py's output) in the "
                                             "browser: the page with its boxes overlaid and a checkbox per class.")
    ap.add_argument("detections", help="the detections JSON to show")
    ap.add_argument("--image", help="the page image (default: the JSON's input_path)")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args(argv)

    dj = Path(a.detections)
    if not dj.is_file():
        sys.exit(f"{dj}: no such file")
    recs, src = load(dj)
    image = Path(a.image) if a.image else src
    if image is None or not image.is_file():
        sys.exit(f"{image}: the page image is not there (the JSON's input_path; give --image)")

    classes = sorted({r["cls"] for r in recs})
    cid = {c: i for i, c in enumerate(classes)}
    cnt = Counter(r["cls"] for r in recs)
    data = {"syms": [[cid[r["cls"]], r["conf"]] + list(r["xyxy"]) for r in recs],
            "classes": classes, "counts": [cnt[c] for c in classes],
            "cols": [COLORS[c] for c in classes]}

    app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))

    @app.route("/")
    def index():
        return render_template("visualizer.html", title=image.name, n=len(recs), m=len(classes), data=data)

    @app.route("/image")
    def img():
        return send_file(image)

    print(f"serving {dj.name} ({len(recs)} detections over {image.name}) on http://127.0.0.1:{a.port}/",
          file=sys.stderr)
    open_browser(a.port)
    app.run(host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
