"""Every vocabulary the reader reads or predicts, in one place.

The detector's classes are the v7 checkpoint's own list (270), so a token's class posterior and the model's class
head share one index. The other heads are small closed vocabularies; index 0 is never "unknown" -- a head that does
not apply to a token is simply not trained or read on it.
"""
from __future__ import annotations

import json
from fractions import Fraction

from .paths import REPO as ROOT
TAXONOMY = ROOT / "models" / "yolo_v7_taxonomy.json"

CLASSES: list[str] = json.loads(TAXONOMY.read_text())["classes"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}
N_CLS = len(CLASSES)


def family(cls: str) -> str:
    return cls.split("_", 1)[0]


FAMILIES = sorted({family(c) for c in CLASSES})
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}
N_FAM = len(FAMILIES)
CLS_FAM = [FAM_ID[family(c)] for c in CLASSES]

# the detector's per-note attribute vocabularies (detector.small.export.ATTR_VOCAB)
POS_MAX = 24
STEM = ["<na>", "down", "up", "both"]
DOTS_IN = ["<na>", "1", "2", "3", "4"]
GRACE_IN = ["<na>", "acc", "unacc"]
VOICE_IN = ["<na>", "1", "2", "3", "4", "5", "6", "7", "8"]

# ---- heads -------------------------------------------------------------------------------------------------
N_POS = 2 * POS_MAX + 1                 # staff position -24..24 (0 = middle line, + = up)
N_DOTS = 4                              # 0..3
# a note's alteration as the bar shows it: "key" = whatever the key signature in effect gives its step (no
# accidental in force), else the value an accidental (on it, earlier in the bar, or across a tie) puts in force
ALTERS = ["key", -2, -1, 0, 1, 2]
N_ALTER = len(ALTERS)
SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"


def key_alter(step: str, fifths: int) -> int:
    """The alteration the key signature gives a step."""
    if fifths > 0:
        return 1 if step in SHARP_ORDER[:fifths] else 0
    if fifths < 0:
        return -1 if step in FLAT_ORDER[:-fifths] else 0
    return 0
N_VOICE = 4                             # voice slot on its staff 1..4 (5+ folded into 4)
GRACE = ["none", "acc", "unacc"]
TUPLETS = ["none", "3/2", "2/3", "5/4", "6/4", "4/3", "7/4", "other"]
N_TUP = len(TUPLETS)
# bar heads: the clef / key / meter a bar SETS -- printed in it (or the page's first bar) -- else None (carried on)
KEYS = list(range(-7, 8)) + [None]      # fifths
N_KEY = len(KEYS)
TIMES = ["2/2", "3/2", "4/2", "2/4", "3/4", "4/4", "5/4", "6/4", "7/4", "3/8", "5/8", "6/8", "7/8", "9/8", "12/8",
         "1/4", "1/8", "2/8", "4/8", "8/8", "9/4", "12/16", "other", None]
TIME_ID = {t: i for i, t in enumerate(TIMES)}
N_TIME = len(TIMES)
CLEFS = ["G2", "F4", "C3", "C4", "C1", "C2", "C5", "G1", "F3", "F5", "G2_8b", "G2_8a", "G2_15b", "G2_15a", "F4_8b",
         "F4_8a", "F4_15b", "F4_15a", "perc", "other", None]
CLEF_ID = {c: i for i, c in enumerate(CLEFS)}
N_CLEF = len(CLEFS)

# families whose tokens carry note-level heads
NOTELIKE = {"note", "rest"}
NOTE_FAM = FAM_ID["note"]
REST_FAM = FAM_ID["rest"]
MEASURE_FAM = FAM_ID["measure"]

DUR_TYPES = ["long", "breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th", "256", "512"]
TYPE_QUARTERS = {"long": Fraction(16), "breve": Fraction(8), "whole": Fraction(4), "half": Fraction(2),
                 "quarter": Fraction(1), "eighth": Fraction(1, 2), "16th": Fraction(1, 4), "32nd": Fraction(1, 8),
                 "64th": Fraction(1, 16), "128th": Fraction(1, 32), "256": Fraction(1, 64), "512": Fraction(1, 128)}


def time_id(beats, beat_type) -> int:
    return TIME_ID.get(f"{beats}/{beat_type}", TIME_ID["other"])


def clef_id(sign: str) -> int:
    return CLEF_ID.get(sign, CLEF_ID["other"])


def tuplet_id(tm) -> int:
    """GT time modification ('3/2', '3:2', [3, 2] or None) -> TUPLETS index."""
    if not tm:
        return 0
    if isinstance(tm, (list, tuple)):
        a, n = tm[:2]
    else:
        s = str(tm).replace(":", "/")
        if "/" not in s:
            return 0
        a, n = s.split("/")[:2]
    try:
        key = f"{int(a)}/{int(n)}"
    except ValueError:
        return len(TUPLETS) - 1
    if key in ("1/1",):
        return 0
    return TUPLETS.index(key) if key in TUPLETS else len(TUPLETS) - 1


def gt_class(e: dict) -> str | None:
    """The detector class a label element is drawn as (None when the detector has no class for it)."""
    t, sub = e.get("type", ""), str(e.get("subtype", "") or "")
    for c in (f"{t}_{sub}" if sub else None, t):
        if c and c in CLS_ID:
            return c
    if t == "dynam" and "dynam" in CLS_ID:
        return "dynam"
    if t in ("slur", "tie"):
        return "curve"
    if t in ("tuplet",):
        return "tupletBracket" if "tupletBracket" in CLS_ID else None
    # unknown subtype of a known family -> the family's catch-all where one exists
    for c in (f"{t}_x",):
        if c in CLS_ID:
            return c
    return None
