"""A page's reading -> MusicXML. Transcription only: every decision was the model's.

The writer lays the model's readings out the way MusicXML wants them and adds nothing to the page: a bar's notes
are its real note / rest symbols in x order per voice, pitches are staff position under the bar's clef (a clef
symbol inside the bar takes over from its x on), alterations are decoded per bar from the model's readings and
the accidental glyphs, keys / clefs are the bar heads' readings and are printed where they change, the meter is
decoded for the page with its bars and written from the first bar on (as not printed where the page does not print
it). Nothing is drawn that the page does not show: a bar that holds nothing gets a <forward> (an empty bar), and
time a staff's first voice leaves before the bar line is a rest that is not printed (print-object="no"), so every
bar has its meter's length. A bar column is as long as most of the page's staves make it (the meter's length unless
most run past it): a voice that runs past that is cut there (_chop), so one overfull staff does not stretch the bar
on every other part.

Rhythm (COPISTERIA_ONSET=1, the default): each voice's notes are placed at the onsets the model reads for them, and
a voice's durations and tuplets are decoded jointly (Viterbi over 48ths of a quarter) from the model's onset,
duration and tuplet distributions, so one misread duration does not shift the rest of the bar. COPISTERIA_ONSET=0
writes the read durations one after another.
"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

from .read import Reading
from .vocab import TYPE_QUARTERS, key_alter

REAL = 0.5
ONSET_GAPS = os.environ.get("COPISTERIA_ONSET", "1") == "1"
ONSET_P = float(os.environ.get("COPISTERIA_ONSET_P", "0.6"))
RHYTHM_DECODE = os.environ.get("COPISTERIA_RHYTHM", "1") == "1"
STEPS = "CDEFGAB"
# diatonic index (octave * 7 + step) of the middle staff line under each clef
MID_LINE = {"G2": 34, "G1": 36, "F4": 22, "F3": 24, "F5": 20, "C1": 32, "C2": 30, "C3": 28, "C4": 26, "C5": 24,
            "G2_8b": 27, "G2_8a": 41, "G2_15b": 20, "G2_15a": 48, "F4_8b": 15, "F4_8a": 29, "F4_15b": 8, "F4_15a": 36,
            "perc": 34, "other": 34}
CLEF_XML = {"G2": ("G", 2, 0), "G1": ("G", 1, 0), "F4": ("F", 4, 0), "F3": ("F", 3, 0), "F5": ("F", 5, 0),
            "C1": ("C", 1, 0), "C2": ("C", 2, 0), "C3": ("C", 3, 0), "C4": ("C", 4, 0), "C5": ("C", 5, 0),
            "G2_8b": ("G", 2, -1), "G2_8a": ("G", 2, 1), "G2_15b": ("G", 2, -2), "G2_15a": ("G", 2, 2),
            "F4_8b": ("F", 4, -1), "F4_8a": ("F", 4, 1), "F4_15b": ("F", 4, -2), "F4_15a": ("F", 4, 2),
            "perc": ("percussion", None, 0), "other": ("G", 2, 0)}
DYN = {"p", "pp", "ppp", "pppp", "ppppp", "mp", "mf", "f", "ff", "fff", "ffff", "fffff", "fp", "fz", "sf", "sfz",
       "sffz", "sfp", "sfpp", "rf", "rfz", "pf"}
ARTIC = {"artic_stacc": "staccato", "artic_acc": "accent", "artic_ten": "tenuto", "artic_marc": "strong-accent",
         "artic_stacciss": "staccatissimo", "artic_acc-soft": "soft-accent"}
TECH = {"artic_upbow": "up-bow", "artic_dnbow": "down-bow", "artic_harm": "harmonic", "artic_open": "open-string",
        "artic_stop": "stopped", "artic_snap": "snap-pizzicato"}
ACCID_XML = {"sharp": "sharp", "flat": "flat", "natural": "natural", "dsharp": "double-sharp", "dflat": "flat-flat",
             "natsharp": "natural-sharp", "natflat": "natural-flat"}
ACCID_ALTER = {"sharp": 1, "flat": -1, "natural": 0, "dsharp": 2, "dflat": -2, "natsharp": 1, "natflat": -1}
ORNAMENT = {"trill": "trill-mark", "mordent": "mordent"}
AFTER_MARK = {"breath": "breath-mark", "caesura": "caesura"}
OCTAVE = {"octave_8a": (1, "down", 8), "octave_8b": (-1, "up", 8), "octave_15a": (2, "down", 15),
          "octave_15b": (-2, "up", 15)}
BEAM_LEVEL = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4, "128th": 5}
BAR_RIGHT = {"barLine_double": "light-light", "barLine_final": "light-heavy", "barLine_repeat_end": "light-heavy",
             "barLine_dashed": "dashed", "barLine_dotted": "dotted"}


@dataclass
class Event:
    x: float
    kind: str                      # note / rest / mrest
    typ: str = "quarter"
    dots: int = 0
    voice: int = 1
    chord: bool = False
    grace: str = "none"
    tup: str = "none"
    step: str = "C"
    octave: int = 4
    alter: int = 0
    tie: bool = False
    tie_stop: bool = False
    unpitched: bool = False
    marks: list = field(default_factory=list)       # articulation / technical / fermata names
    dur: Fraction = Fraction(1)
    si: int = -1
    tuplet_mark: str = ""                           # start / stop
    accidental: str = ""                            # the accidental glyph printed before it, if any
    stem: str = ""                                  # up / down, as the detector saw the stem
    beams: list = field(default_factory=list)       # [(level, begin|continue|end|forward hook|backward hook)]
    slurs: list = field(default_factory=list)       # [(start|stop, number)]
    arpeggiate: bool = False
    pre: list = field(default_factory=list)         # directions written before the note: (kind, payload, below)
    post: list = field(default_factory=list)        # ... and after it
    copied: bool = False                            # a measure-repeat copy (spanners attach to the original only)
    onset: float | None = None                      # the model's onset in the bar (quarters) and its confidence
    onset_p: float = 0.0
    dist: dict | None = None                        # the model's onset / tuplet distributions (joint decoding)


def _real(rd: Reading, si: int) -> bool:
    r = rd.tok.get(si)
    return r is not None and r["real"]["p"] >= REAL


def _cls(rd: Reading, si: int) -> str:
    return rd.tok[si]["cls"]["v"]


def _fam(c: str) -> str:
    return c.split("_", 1)[0]


def _duration(typ: str, dots: int, tup: str) -> Fraction:
    d = TYPE_QUARTERS.get(typ, Fraction(1))
    d = d * (2 - Fraction(1, 2 ** dots))
    if tup not in ("none", "other"):
        a, n = (int(v) for v in tup.split("/"))
        d = d * n / a
    return d


def _type_of(cls: str) -> str:
    t = cls.split("_", 1)[1] if "_" in cls else "quarter"
    return t if t in TYPE_QUARTERS else "quarter"


def _clef_of_cls(cls: str) -> str:
    name = cls[len("clef_"):]
    if name in MID_LINE:
        return name
    return "perc" if name == "perc" else "other"


def _pitch(clef: str, pos: int) -> tuple[str, int]:
    idx = MID_LINE.get(clef, 34) + pos
    return STEPS[idx % 7], idx // 7


@dataclass
class PartPlan:
    staves: list[int]                  # staff positions (within a system) of this part's staves


def _plan_parts(rd: Reading) -> list[PartPlan]:
    L = rd.layout
    if not L.systems:
        return []
    tmpl = max(L.systems, key=lambda sy: len(sy.staves))
    K = len(tmpl.staves)
    taken = [False] * K
    plans = []
    for si, s in enumerate(L.syms):
        if not (_real(rd, si) and _cls(rd, si) == "grpSym_brace"):
            continue
        cov = [p for p, k in enumerate(tmpl.staves) if s.box[1] - L.staves[k].sp <= L.staves[k].mid
               <= s.box[3] + L.staves[k].sp]
        if len(cov) == 2 and cov[1] == cov[0] + 1 and not any(taken[p] for p in cov):
            for p in cov:
                taken[p] = True
            plans.append(PartPlan(staves=cov))
    for p in range(K):
        if not taken[p]:
            plans.append(PartPlan(staves=[p]))
    plans.sort(key=lambda pl: pl.staves[0])
    return plans


def _part_groups(rd: Reading, plans: list) -> list[tuple[int, int, str]]:
    """(first part, last part, bracket|brace) for every detected bracket or brace that spans more than one part of
    the fullest system."""
    L = rd.layout
    if not L.systems or len(plans) < 2:
        return []
    tmpl = max(L.systems, key=lambda sy: len(sy.staves))
    part_of = {p: n for n, pl in enumerate(plans) for p in pl.staves}
    out = []
    for s in L.syms:
        if not _real(rd, s.i) or _cls(rd, s.i) not in ("grpSym_bracket", "grpSym_brace"):
            continue
        cov = sorted({part_of[p] for p, k in enumerate(tmpl.staves) if p in part_of and
                      s.box[1] - L.staves[k].sp <= L.staves[k].mid <= s.box[3] + L.staves[k].sp})
        if len(cov) >= 2 and (cov[0], cov[-1]) not in [(a, b) for a, b, _ in out]:
            out.append((cov[0], cov[-1], "bracket" if s.cls.endswith("bracket") else "brace"))
    return out


def _bar_state(rd: Reading, staff: int, bar: int, prev: dict) -> dict:
    """What the bar sets (its heads' readings) over what it carries from the bar before."""
    si = rd.layout.staves[staff].bars[bar]
    r = rd.tok.get(si)
    state = dict(prev)
    if r is not None and "key" in r:
        for k in ("key", "time", "clef"):
            if r[k]["v"] is not None:
                state[k] = r[k]["v"]
    return state


def _texts_near(texts, box, sp):
    """OCR digits over a multi-measure rest: its count."""
    if not texts:
        return None
    x0, y0, x1, y1 = box
    for t in texts:
        tx0, ty0, tx1, ty1 = t["xyxy"]
        if re.fullmatch(r"\d{1,3}", str(t.get("text", "")).strip()) and tx1 > x0 and tx0 < x1 and \
                y0 - 6 * sp < (ty0 + ty1) / 2 < y0 + (y1 - y0) / 2:
            return int(t["text"])
    return None


def _events(rd: Reading, staff: int, bar: int, clef: str, key: int, texts) -> tuple[list[Event], list, dict]:
    """The bar's events, plus its directions [(x, kind, payload)] and bar-line marks."""
    L = rd.layout
    st = L.staves[staff]
    sp = st.sp
    syms = [s for s in L.syms if s.staff == staff and s.bar == bar and _real(rd, s.i) and s.fam != "measure"]
    syms.sort(key=lambda s: s.cx)
    clef_changes = sorted((s.cx, _clef_of_cls(_cls(rd, s.i))) for s in syms if _fam(_cls(rd, s.i)) == "clef")
    notes_x = [s.cx for s in syms if _fam(_cls(rd, s.i)) in ("note", "rest")]
    first_x = min(notes_x) if notes_x else math.inf
    mid_clefs = [(x, c) for x, c in clef_changes if x > first_x]
    events: list[Event] = []
    dirs: list = []
    marks = {"left": None, "right": None, "endings": [], "multirest": None, "repeat_bar": False,
             "end_clef": mid_clefs[-1][1] if mid_clefs else None,
             "meter": any(_fam(_cls(rd, s.i)) == "meterSig" for s in syms)}
    bx0, _, bx1, _ = L.bar_box(staff, bar)
    for s in syms:
        r = rd.tok[s.i]
        c = r["cls"]["v"]
        f = _fam(c)
        if f == "note":
            cur = clef
            for x, cc in mid_clefs:
                if x < s.cx:
                    cur = cc
            pos = r["pos"]["v"]
            step, octv = _pitch(cur, pos)
            alt = r["alter"]["v"]
            alt = key_alter(step, key) if alt == "key" else alt
            e = Event(x=s.cx, kind="note", typ=_type_of(c), dots=r["dots"]["v"], voice=r["voice"]["v"],
                      chord=r.get("chord", 0) > 0.5, grace=r["grace"]["v"], tup=r["tup"]["v"], step=step,
                      octave=octv, alter=alt, tie=r.get("tie", 0) > 0.5, unpitched=cur == "perc",
                      si=s.i, stem=s.attrs.get("stem_dir") if s.attrs.get("stem_dir") in ("up", "down") else "",
                      onset=r.get("onset"), onset_p=r.get("onset_p", 0.0), dist=_dists(r))
            events.append(e)
        elif f == "rest":
            events.append(Event(x=s.cx, kind="rest", typ=_type_of(c), dots=r["dots"]["v"], voice=r["voice"]["v"],
                                tup=r["tup"]["v"], si=s.i, onset=r.get("onset"), onset_p=r.get("onset_p", 0.0),
                                dist=_dists(r)))
        elif f == "mRest":
            events.append(Event(x=s.cx, kind="mrest", voice=1, si=s.i))
        elif f == "multiRest":
            n = _texts_near(texts, s.box, sp)
            marks["multirest"] = n or 1
            events.append(Event(x=s.cx, kind="mrest", voice=1, si=s.i))
        elif f in ("mRpt", "multiRpt"):
            marks["repeat_bar"] = True
        elif f == "dynam" and c[len("dynam_"):] in DYN:
            dirs.append((s.cx, "dyn", c[len("dynam_"):], s.cy > st.mid))
        elif f == "hairpin" and c in ("hairpin_cres", "hairpin_dim"):
            dirs.append((s.box[0], "wedge", "crescendo" if c == "hairpin_cres" else "diminuendo", s.cy > st.mid))
            dirs.append((s.box[2], "wedge", "stop", s.cy > st.mid))
        elif c in ARTIC or c in TECH or c in ORNAMENT or c == "fermata":
            dirs.append((s.cx, "mark", c, s.cy))
        elif c in AFTER_MARK:                       # breath mark / caesura: after the note on its left
            left = [e for e in events if e.kind == "note" and e.x < s.cx and s.cx - e.x < 5 * sp]
            if left:
                max(left, key=lambda e: e.x).marks.append(c)
        elif f == "barLine":
            if c == "barLine_repeat_start" and s.cx < (bx0 + bx1) / 2:
                marks["left"] = "repeat_start"
            elif c in BAR_RIGHT and s.cx >= (bx0 + bx1) / 2:
                marks["right"] = c
        elif f == "ending":
            marks["endings"].append((c[len("ending_"):], s.box))
        elif f == "clef" and s.cx > first_x:
            dirs.append((s.cx, "clef", _clef_of_cls(c), None))
    _decode_alters(rd, syms, events, sp)
    # marks on notes: the nearest note in x (within 1.5 staff spaces)
    for x, kind, c, y in [d for d in dirs if d[1] == "mark"]:
        cands = [e for e in events if e.kind == "note" and abs(e.x - x) < 1.5 * sp]
        if cands:
            best = min(cands, key=lambda e: abs(e.x - x))
            best.marks.append(c)
    dirs = [d for d in dirs if d[1] != "mark"]
    return events, dirs, marks


ALTER_KEEP_P = 0.97              # a model this sure of another alteration keeps it against the notation's rule
# where an accidental glyph's box sits against the head it belongs to (staff spaces, + is down): a flat's bowl, at
# the head's height, is the low part of a glyph that reaches up; sharps and naturals are centred on it
ACC_ANCHOR = {"flat": 0.46, "dflat": 0.46, "natflat": 0.46}


def _decode_alters(rd: Reading, syms: list, events: list[Event], sp: float) -> None:
    """The bar's sounding alterations as notation defines them: a printed accidental sets its head's alteration
    and holds for the later heads on the same line (step and octave) to the bar's end; a head with neither takes
    the model's reading (the key's, or what the model saw that the glyphs do not show), and a model that is sure
    (ALTER_KEEP_P) of another alteration than the rule's keeps its own. Each glyph goes to one head, the nearest
    at its anchor's height to its right (one-to-one, the closest pairs first); it is printed where it says what
    the alteration is."""
    L = rd.layout
    heads = [e for e in events if e.kind == "note" and not e.unpitched]
    pairs = []
    for s in syms:
        c = _cls(rd, s.i)
        kind = c[len("accid_"):]
        if _fam(c) != "accid" or kind not in ACCID_ALTER:
            continue
        ay = s.cy + ACC_ANCHOR.get(kind, 0.0) * sp
        for e in heads:
            h = L.syms[e.si]
            gap = h.box[0] - s.box[2]
            dy = abs(h.cy - ay)
            if -0.5 * sp < gap < 5 * sp and dy < 0.4 * sp:
                pairs.append((3 * dy / sp + max(0.0, gap) / sp, s.i, e.si, ACCID_ALTER[kind], kind))
    pairs.sort()
    glyph: dict[int, tuple[int, str]] = {}
    used = set()
    for _, gi, ei, alt, kind in pairs:
        if gi in used or ei in glyph:
            continue
        used.add(gi)
        glyph[ei] = (alt, kind)
    held: dict[tuple, int] = {}
    for e in sorted(heads, key=lambda e: L.syms[e.si].cx):
        line = (e.step, e.octave)
        kind = None
        if e.si in glyph:
            rule, kind = glyph[e.si]
            held[line] = rule
        elif line in held:
            rule = held[line]
        else:
            continue
        if rule != e.alter and rd.tok[e.si]["alter"].get("p", 0.0) < ALTER_KEEP_P:
            e.alter = rule
        if kind is not None and ACCID_ALTER[kind] == e.alter and kind in ACCID_XML:
            e.accidental = ACCID_XML[kind]


def _voices(events: list[Event], sp: float, single_staff_part: bool = True,
            bar_len: Fraction | None = None, score: list | None = None) -> dict[int, list[Event]]:
    """Events per voice, in time order. A chord member (chord head > 0.5) joins the root nearest it in x within
    1.5 staff spaces in its voice (a root is a note the model did not call a member); without one it stands alone."""
    vs: dict[int, list[Event]] = {}
    for e in sorted(events, key=lambda e: e.x):
        vs.setdefault(e.voice, []).append(e)
    for v, evs in list(vs.items()):
        roots = [e for e in evs if e.kind == "note" and e.grace == "none" and not e.chord]
        members: dict[int, list[Event]] = {}
        order = []
        for e in evs:
            if e.kind == "note" and e.chord and e.grace == "none":
                near = [r for r in roots if abs(r.x - e.x) <= 1.5 * sp]
                if near:
                    members.setdefault(id(min(near, key=lambda r: abs(r.x - e.x))), []).append(e)
                    continue
                e.chord = False
            order.append(e)
        if ONSET_GAPS and RHYTHM_DECODE:
            # one voice on a one-staff part (a string, a singer): its events cannot overlap; a keyboard staff's
            # notes may sit in voices the reading merged
            sc = _decode_rhythm(order, strict=single_staff_part and len(vs) == 1, bar_len=bar_len)
            if score is not None and sc is not None:
                score.append(sc)
        elif ONSET_GAPS:
            _tuplets_from_onsets(order)
        out = []
        cursor = Fraction(0)
        for e in order:
            e.dur = Fraction(0) if e.grace != "none" else _duration(e.typ, e.dots, e.tup)
            if ONSET_GAPS and e.grace == "none" and e.onset is not None and e.onset_p >= ONSET_P:
                # the model's onset places the event; the running sum of durations only where it is unsure
                at = Fraction(round(e.onset * 48), 48)
                if at != cursor:
                    out.append(Event(x=e.x - 0.01, kind="gap", voice=e.voice, dur=at - cursor))
                    cursor = at
            out.append(e)
            cursor += e.dur
            for m in members.get(id(e), []):
                m.dur, m.tup = e.dur, e.tup
                out.append(m)
        vs[v] = out
        evs = out
        # tuplet brackets: runs of one ratio, closed once the run spans actual x the first note's type (or where
        # the run is broken: another ratio, a plain note, the bar's end)
        def close(run):
            if len(run) > 1:
                run[-1].tuplet_mark = "stop"
            elif run:
                run[0].tuplet_mark = ""

        run, nominal, target = [], Fraction(0), None
        for e in evs:
            if e.chord or e.grace != "none":
                continue
            if e.tup in ("none", "other"):
                close(run)
                run, nominal, target = [], Fraction(0), None
                continue
            a, n = (int(x) for x in e.tup.split("/"))
            if not run or run[0].tup != e.tup:
                close(run)
                run, nominal = [e], Fraction(0)
                target = a * TYPE_QUARTERS.get(e.typ, Fraction(1))
                e.tuplet_mark = "start"
            else:
                run.append(e)
            nominal += TYPE_QUARTERS.get(e.typ, Fraction(1)) * (2 - Fraction(1, 2 ** e.dots))
            if nominal >= target:
                close(run)
                run, nominal, target = [], Fraction(0), None
        close(run)
    return vs


BEAM_REACH = 0.7                # how far past a beam box's ends a stem may stand (staff spaces)


def _attach(rd: Reading, built: list, plans: list, texts: list, mask=None) -> None:
    """Page-wide marks attached to the notes they belong to, by position: beams (the notes under a beam box,
    levels from their types), slurs (the notes nearest a curve's two ends, unless the model read that curve's
    notes as tied; a curve running off the end of its system continues to the next one), arpeggios, octave
    lines (which also move the sounding pitch), pedal lines, segno / coda, and OCR'd direction words."""
    L = rd.layout
    notes_on: dict[int, list] = {}                 # staff -> its note events (originals only), in system order
    for measures in built:
        for m in measures:
            for st in m["staves"]:
                if st is None:
                    continue
                for evs in st["voices"].values():
                    for e in evs:
                        if e.kind == "note" and not e.copied and e.si >= 0:
                            notes_on.setdefault(L.syms[e.si].staff, []).append(e)
    for evs in notes_on.values():
        evs.sort(key=lambda e: e.x)
    sym = L.syms
    spanners = [s for s in sym if s.staff >= 0 and _real(rd, s.i)]

    # beams: every beamed-note root under the box, in the box's voice; levels from the note types
    claimed: set = set()
    part_of = {pos: pi for pi, pl in enumerate(plans) for pos in pl.staves}    # staff position -> its part

    def under_beam(b):
        sp = L.staves[b.staff].sp
        # the beam's own staff, and a neighbour in its system whose notes' stems reach it (a beam between the two
        # staves of a keyboard part joins notes of both)
        st_b = L.staves[b.staff]
        near_staves = [b.staff] + [k for k in (b.staff - 1, b.staff + 1) if 0 <= k < len(L.staves) and
                                   L.staves[k].system == st_b.system and
                                   part_of.get(L.staves[k].pos, -1) == part_of.get(st_b.pos, -2)]

        def reach(e):
            # a beam meets its notes at their stems: a head under the beam has its stem up, at its right side; a
            # head over it has it down, at its left
            h = sym[e.si].box
            x = h[2] if b.cy < sym[e.si].cy else h[0]
            return b.box[0] - BEAM_REACH * sp <= x <= b.box[2] + BEAM_REACH * sp
        return [e for k in near_staves for e in notes_on.get(k, []) if not e.chord and e.grace == "none"
                and id(e) not in claimed and reach(e)
                and abs(sym[e.si].cy - b.cy) <= (8 if k == b.staff else 4.5) * sp and e.typ in BEAM_LEVEL]

    for b in sorted((s for s in spanners if _cls(rd, s.i) == "beam"), key=lambda s: s.box[0]):
        sp = L.staves[b.staff].sp
        under = under_beam(b)
        if len(under) < 2:
            continue
        voice = max({e.voice for e in under}, key=lambda v: sum(e.voice == v for e in under))
        group = sorted((e for e in under if e.voice == voice), key=lambda e: e.x)
        if len(group) < 2:
            continue
        lv = [BEAM_LEVEL[e.typ] for e in group]
        for n in range(1, max(lv) + 1):
            for i, e in enumerate(group):
                if lv[i] < n:
                    continue
                prev = i > 0 and lv[i - 1] >= n
                nxt = i + 1 < len(group) and lv[i + 1] >= n
                val = ("continue" if prev and nxt else "begin" if nxt else "end" if prev else
                       "forward hook" if i + 1 < len(group) else "backward hook")
                if n == 1 and val.endswith("hook"):
                    val = "begin" if i == 0 else "end"
                e.beams.append((n, val))
        for e in group:
            claimed.add(id(e))
            if not e.stem:
                e.stem = "up" if b.cy < sym[e.si].cy else "down"

    # slurs
    nums: dict[int, list] = {}                      # staff -> numbers in use, as (end order, number)
    order = {id(e): (L.staves[sym[e.si].staff].system, e.x) for evs in notes_on.values() for e in evs}
    for c in sorted((s for s in spanners if _cls(rd, s.i) == "curve"), key=lambda s: (s.staff, s.box[0])):
        st = L.staves[c.staff]
        sp = st.sp
        cand = [e for e in notes_on.get(c.staff, []) if abs(sym[e.si].cy - c.cy) <= 7 * sp]
        if not cand:
            continue
        # an arc over the notes ends at its box's lower corners, one under them at its upper corners: the end
        # notes are the ones nearest those corners
        near = sorted(sym[e.si].cy for e in cand if c.box[0] - sp <= sym[e.si].cx <= c.box[2] + sp)
        over = bool(near) and c.cy < near[len(near) // 2]
        yend = c.box[3] if over else c.box[1]

        def cost(e, x):
            ls = sym[e.si]
            return abs(ls.cx - x) / sp + 0.5 * abs(ls.cy - yend) / sp

        a = min(cand, key=lambda e: cost(e, c.box[0]))
        z = min(cand, key=lambda e: cost(e, c.box[2]))
        if abs(sym[a.si].cx - c.box[0]) > 2.5 * sp:
            continue                                 # a curve's tail from the system before: drawn there
        if abs(sym[z.si].cx - c.box[2]) > 2.5 * sp:
            if c.box[2] < st.x1 - 3 * sp:
                continue
            nxt = _next_system_first(L, notes_on, c.staff)       # runs off the system: to the next one's first note
            if nxt is None:
                continue
            z = nxt
        if a is z:
            continue
        if a.tie and a.step == z.step and a.octave == z.octave:
            continue                                 # the model read these two as tied: that curve is the tie
        busy = [num for end, num in nums.get(c.staff, []) if end >= order[id(a)]]
        num = next(n for n in range(1, 7) if n not in busy) if len(busy) < 6 else 1
        nums.setdefault(c.staff, []).append((order[id(z)], num))
        a.slurs.append(("start", num))
        z.slurs.append(("stop", num))

    for s in spanners:
        c = _cls(rd, s.i)
        sp = L.staves[s.staff].sp
        evs = notes_on.get(s.staff, [])
        if c == "arpeg":                            # the chord to its right
            for e in evs:
                ls = sym[e.si]
                if s.box[2] - 0.5 * sp <= ls.cx <= s.box[2] + 3 * sp and s.box[1] - sp <= ls.cy <= s.box[3] + sp:
                    e.arpeggiate = True
        elif c in OCTAVE or c == "octave_8":
            shift, typ, size = OCTAVE.get(c, (1, "down", 8) if s.cy < L.staves[s.staff].mid else (-1, "up", 8))
            inside = [e for e in evs if s.box[0] - sp <= sym[e.si].cx <= s.box[2] + sp]
            if inside:
                for e in inside:
                    e.octave += shift
                inside[0].pre.append(("octave", (typ, size), shift < 0))
                inside[-1].post.append(("octave", ("stop", size), shift < 0))
        elif c == "pedal":
            inside = [e for e in evs if s.box[0] - sp <= sym[e.si].cx <= s.box[2] + sp]
            if inside:
                inside[0].pre.append(("pedal", "start", True))
                inside[-1].post.append(("pedal", "stop", True))
        elif c in ("repeatMark_segno", "repeatMark_coda") and evs:
            e = min(evs, key=lambda e: abs(sym[e.si].cx - s.cx))
            e.pre.append(("segno" if c.endswith("segno") else "coda", None, False))
    # OCR'd words over / under a staff (tempo and expression marks): before the nearest note
    for t in texts or ():
        if t.get("role") not in ("dir", "tempo", "expression", "words"):
            continue
        word = str(t.get("text", "")).strip()
        # bar and page numbers are not directions; a word is mostly letters
        if not word or sum(ch.isalpha() for ch in word) < max(2, 0.6 * len(word.replace(" ", ""))):
            continue
        x0, y0, x1, y1 = t["xyxy"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best = None
        for k, evs in notes_on.items():
            st = L.staves[k]
            d = max(0.0, st.y0 - cy, cy - st.y1) / st.sp
            if d <= 6 and evs and st.x0 - 2 * st.sp <= cx <= st.x1 + 2 * st.sp and (best is None or d < best[0]):
                best = (d, k)
        if best is not None:
            k = best[1]
            e = min(notes_on[k], key=lambda e: abs(sym[e.si].cx - x0))
            below = cy > L.staves[k].mid
            end = _dashes_after(mask, t["xyxy"], L.staves[k].sp) if mask is not None else None
            if end is None:
                e.pre.append(("words", str(t["text"]).strip(), below))
                continue
            # "cresc. - - - -": the word with its dashes, which stop at the last note they reach
            e.pre.append(("words_dashes", str(t["text"]).strip(), below))
            reached = [n for n in notes_on[k] if sym[n.si].cx <= end]
            (max(reached, key=lambda n: sym[n.si].cx) if reached else e).post.append(("dashes_stop", None, below))


def _dashes_after(mask, box, sp: float) -> float | None:
    """x where a run of dashes after a word ends ("cresc. - - - -"), or None: at least two short, thin
    horizontal strokes at the word's height, one after another (a stem or a bar line crossing them is passed
    over), the first within 3 staff spaces of the word."""
    x0, y0, x1, y1 = box
    h = max(1.0, y1 - y0)
    H, W = mask.shape
    a, b = max(0, int(y0 + 0.2 * h)), min(H, int(y1 - 0.1 * h))
    xa, xb = int(x1), min(W, int(x1 + 80 * sp))
    if b - a < 2 or xb - xa < 4:
        return None
    band = mask[a:b, xa:xb]
    on = band.any(axis=0)
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(on) - 1):
            segs.append((start, i + 1 if v else i))
            start = None
    dashes, last = [], 0
    for s0, s1 in segs:
        w = s1 - s0
        if w < 0.25 * sp:                       # a stem or a bar line through the run
            continue
        if s0 - last > 3 * sp:
            break
        if w <= 2.0 * sp and band[:, s0:s1].any(axis=1).sum() <= max(2, 0.4 * sp):
            dashes.append((s0, s1))
            last = s1
        else:
            break
    return float(xa + dashes[-1][1]) if len(dashes) >= 2 else None


def _next_system_first(L, notes_on, staff: int):
    """The first note of the same staff position in the next system, if there is one."""
    st = L.staves[staff]
    if st.system + 1 >= len(L.systems):
        return None
    nxt = L.systems[st.system + 1]
    if st.pos >= len(nxt.staves):
        return None
    evs = notes_on.get(nxt.staves[st.pos], [])
    return evs[0] if evs else None


def _dists(r: dict) -> dict | None:
    if "onset_beat_p" not in r:
        return None
    return {"beat": r["onset_beat_p"], "frac": r["onset_frac_p"], "tup": r.get("tup_p")}


TUP_CHOICES = ["none", "3/2"]
GAP_LOGP = math.log(0.02)          # a stretch of the voice with no event read in it (a note the detector missed)
OVERLAP_LOGP = math.log(0.02)
SAME_ONSET_LOGP = math.log(0.05)
PAST_BAR_LOGP = math.log(0.01)
END_GAP_LOGP = GAP_LOGP            # a voice ending short of the bar: an event missed there (scores a meter only)


def _decode_rhythm(order: list, strict: bool = False, bar_len: Fraction | None = None) -> float | None:
    """The onsets and tuplet ratios of one voice's events in a bar, decoded jointly from the model's own
    distributions: the most probable sequence in which every event starts where the one before it ends (or later,
    at a price, where an event may have been missed). Arithmetic of time over the model's probabilities -- the
    model reads each note, this keeps the readings consistent with each other.

    Returns the decoded reading's log-score under ``bar_len``, time left over before the bar's end priced as a
    missed event (END_GAP_LOGP, which does not change the reading): how well the bar's length fits its notes, for
    decoding the page's meter (None: nothing to decode)."""
    import numpy as np
    evs = [e for e in order if e.grace == "none" and e.kind in ("note", "rest") and e.dist is not None]
    if not evs:
        return None
    G = 48
    K = 16 * G
    NEG = -1e18

    L48 = int(bar_len * G) if bar_len else K

    def emit(e):
        pb = np.log(np.clip(np.asarray(e.dist["beat"], float), 1e-6, 1))
        pf = np.log(np.clip(np.asarray(e.dist["frac"], float), 1e-6, 1))
        out = (pb[:, None] + pf[None, :]).reshape(-1)[:K]
        out[L48:] += PAST_BAR_LOGP                  # an onset past the bar's end (by the meter read or carried)
        return out

    def tup_logp(e, t):
        if e.dist.get("tup") is None:
            return 0.0 if t == e.tup else math.log(0.05)
        from .vocab import TUPLETS
        return math.log(max(1e-6, e.dist["tup"][TUPLETS.index(t)]))

    def dur48(e, t):
        d = _duration(e.typ, e.dots, t) * G
        return int(d) if d.denominator == 1 else None

    def end_logp(e, t):
        """Per state of the last event: what its end leaves of the bar (past it, or short of it)."""
        d = dur48(e, t) if t not in ("other",) else None
        if not d:
            return np.zeros(K), np.zeros(K)
        end = np.arange(K) + d
        return np.where(end > L48, PAST_BAR_LOGP, 0.0), np.where(end < L48, END_GAP_LOGP, 0.0)

    if len(evs) == 1:
        over, short = end_logp(evs[0], evs[0].tup)
        return float(np.max(emit(evs[0]) + over + short))
    idx = np.arange(K)
    choices = [[t for t in dict.fromkeys(TUP_CHOICES + [e.tup]) if t not in ("other",) and dur48(e, t)]
               for e in evs]
    score = emit(evs[0]).copy()
    back = []
    for i in range(1, len(evs)):
        prev, e = evs[i - 1], evs[i]
        em = emit(e)
        best = np.full(K, NEG); arg = np.zeros((K, 3), int)       # (prev state, choice index, gap flag)
        pref = np.maximum.accumulate(score)                       # best score at or before each state
        pref_arg = np.maximum.accumulate(np.where(score >= pref, idx, 0))           # (the latest such state)
        suf = np.maximum.accumulate(score[::-1])[::-1]            # best score at or after each state
        suf_arg = np.minimum.accumulate(np.where(score >= suf, idx, K)[::-1])[::-1]   # (the earliest such)
        for c, t in enumerate(choices[i - 1]):
            d = dur48(prev, t)
            if d is None or d <= 0 or d >= K:
                continue
            tl = tup_logp(prev, t)
            # exact: j = k + d
            cand = np.full(K, NEG); src = np.zeros(K, int)
            cand[d:] = score[:K - d] + tl; src[d:] = np.arange(K - d)
            better = cand > best
            best[better] = cand[better]; arg[better] = np.stack([src[better], np.full(better.sum(), c),
                                                                 np.zeros(better.sum(), int)], 1)
            # with a gap: j > k + d
            g = np.full(K, NEG); gs = np.zeros(K, int)
            if d + 1 < K:
                g[d + 1:] = pref[:K - d - 1] + tl + GAP_LOGP; gs[d + 1:] = pref_arg[:K - d - 1]
            better = g > best
            best[better] = g[better]; arg[better] = np.stack([gs[better], np.full(better.sum(), c),
                                                              np.ones(better.sum(), int)], 1)
            # overlapping: j < k + d (a chord the chord head missed, two voices read as one) -- dear, but the
            # onsets the model is sure of are not pushed along by it
            o = np.full(K, NEG); osrc = np.zeros(K, int)
            lo = np.arange(K) - d + 1                              # the earliest k with k + d > j
            ok = lo < K
            lo_c = np.clip(lo, 0, K - 1)
            if strict:
                continue
            o[ok] = suf[lo_c[ok]] + tl + OVERLAP_LOGP; osrc[ok] = suf_arg[lo_c[ok]]
            # the same onset as the event before (a chord the chord head did not call) is cheaper than other
            # overlaps
            same = score + tl + SAME_ONSET_LOGP
            take = same > o
            o[take] = same[take]; osrc[take] = np.arange(K)[take]
            better = o > best
            best[better] = o[better]; arg[better] = np.stack([osrc[better], np.full(better.sum(), c),
                                                              np.full(better.sum(), 2)], 1)
        score = best + em
        back.append(arg)
    # the last event's own tuplet: its head's choice; a voice running past the bar's end pays for it
    over, short = end_logp(evs[-1], evs[-1].tup)
    score = score + over
    j = int(np.argmax(score))
    total = float(score[j] + short[j])
    states = [j]
    picks = []
    for arg in reversed(back):
        k, c, _ = arg[j]
        picks.append(c)
        states.append(int(k))
        j = int(k)
    states.reverse(); picks.reverse()
    for i, e in enumerate(evs):
        e.onset, e.onset_p = states[i] / G, 1.0
        if i < len(picks):
            e.tup = choices[i][picks[i]]
    return total


def _tuplets_from_onsets(order: list) -> None:
    """Where two neighbouring events of a voice both have a confident onset, the time between them says whether
    the first is a triplet: two thirds of its written value is one, exactly its written value is not."""
    timed = [e for e in order if e.grace == "none" and e.kind in ("note", "rest") and e.onset is not None and
             e.onset_p >= ONSET_P]
    for e, f in zip(timed, timed[1:]):
        gap = Fraction(round((f.onset - e.onset) * 48), 48)
        nominal = TYPE_QUARTERS.get(e.typ, Fraction(1)) * (2 - Fraction(1, 2 ** e.dots))
        if gap == nominal * Fraction(2, 3) and e.tup == "none":
            e.tup = "3/2"
        elif gap == nominal and e.tup == "3/2":
            e.tup = "none"


def _time_len(time: str) -> Fraction:
    if time == "other" or "/" not in time:
        return Fraction(4)
    b, t = time.split("/")
    return Fraction(int(b) * 4, int(t))


SHORT_RIGHT = ("barLine_final", "barLine_repeat_end")      # a bar before these may be short (an upbeat's complement)


def _voice_starts(evs: list[Event], mrest_len: Fraction) -> tuple[list, Fraction]:
    """Each event's start in the bar as the writer places it (a chord member's is its root's, a gap has none) and
    where the voice ends."""
    starts, pos, end, root = [], Fraction(0), Fraction(0), Fraction(0)
    for e in evs:
        if e.kind == "gap":
            starts.append(None)
            pos += e.dur
        elif e.chord:
            starts.append(root)
        else:
            starts.append(pos)
            root = pos
            pos += mrest_len if e.kind == "mrest" else e.dur
            end = max(end, pos)
    return starts, end


def _column_lengths(built: list) -> list:
    """Each bar column's length: the one most of the page's staves give it as written (a staff's first voice is
    filled to its meter's length unless the bar may be short); on a tie, the meter's."""
    out = []
    for m_i in range(max((len(ms) for ms in built), default=0)):
        votes: Counter = Counter()
        meters = set()
        for ms in built:
            first = next((s for s in ms[m_i]["staves"] if s), None) if m_i < len(ms) else None
            if first is None:
                continue
            bar_len = _time_len(first["state"]["time"])
            meters.add(bar_len)
            short = (m_i == 0 and first["marks"]["meter"]) or first["marks"]["right"] in SHORT_RIGHT
            for st in ms[m_i]["staves"]:
                if st is None:
                    continue
                n = max((_voice_starts(evs, bar_len)[1] for evs in st["voices"].values()), default=bar_len)
                votes[n if short or not st["voices"] else max(n, bar_len)] += 1
        out.append(max(votes, key=lambda n: (votes[n], n in meters, -n)) if votes else None)
    return out


def _fit_value(q: Fraction) -> tuple[str, int] | None:
    """The note value (type, dots) that lasts ``q`` quarters, if one does."""
    for typ, t in TYPE_QUARTERS.items():
        for dots in range(3):
            if t * (2 - Fraction(1, 2 ** dots)) == q:
                return typ, dots
    return None


def _chop(evs: list[Event], cut: Fraction) -> list[Event]:
    """A voice cut at ``cut`` quarters, its bar column's length: what starts at or after it is dropped, a plain note
    or rest that runs past it ends there (a tuplet member, or one no note value fits, is dropped); the beams, tuplet
    brackets, ties and slurs the cut leaves open are closed. A voice that ends in time comes back as it is."""
    starts, end = _voice_starts(evs, cut)
    if end <= cut:
        return evs
    kept, dropped, root, shortened = [], [], None, False
    for e, s in zip(evs, starts):
        if e.kind == "gap":
            continue
        if e.chord:
            if root is None:
                dropped.append(e)
                continue
            if shortened:                                    # a chord ends with its root
                e.typ, e.dots, e.dur, e.beams = root.typ, root.dots, root.dur, []
            kept.append((e, s))
            continue
        length = cut if e.kind == "mrest" else e.dur
        root, shortened = None, False
        if s >= cut:
            dropped.append(e)
            continue
        if s + length > cut:
            fit = _fit_value(cut - s) if e.kind in ("note", "rest") and e.tup in ("none", "other") else None
            if fit is None:
                dropped.append(e)
                continue
            e.typ, e.dots = fit
            e.dur = cut - s
            e.beams = []
            shortened = True
        root = e
        kept.append((e, s))
    out, cursor = [], Fraction(0)
    for e, s in kept:
        if not e.chord:
            if s != cursor:
                out.append(Event(x=e.x - 0.01, kind="gap", voice=e.voice, dur=s - cursor))
                cursor = s
            cursor += cut if e.kind == "mrest" else e.dur
        out.append(e)
    notes = [e for e in out if e.kind == "note"]
    for d in dropped:
        if d.tie_stop:                                       # its tie's start, now without an end
            for e in reversed(notes):
                if e.tie and e.step == d.step and e.octave == d.octave:
                    e.tie = False
                    break
        for typ, num in d.slurs:                             # a slur that ended on it ends on the last note kept
            if typ == "stop" and notes:
                last = next((e for e in reversed(notes) if not e.chord), notes[-1])
                if ("start", num) in last.slurs:
                    last.slurs.remove(("start", num))
                else:
                    last.slurs.append(("stop", num))
    heads = [e for e in notes if not e.chord and e.grace == "none"]
    for lvl in sorted({lv for e in heads for lv, _ in e.beams}):
        group: list[Event] = []
        for e in heads:
            val = dict(e.beams).get(lvl)
            if val == "begin" or (val == "continue" and not group):
                group = [e]
                e.beams = [(lv, "begin" if lv == lvl else v) for lv, v in e.beams]
            elif val == "continue":
                group.append(e)
            elif val == "end":
                if not group:
                    e.beams = [(lv, v) for lv, v in e.beams if lv < lvl]
                group = []
        if len(group) > 1:
            group[-1].beams = [(lv, "end" if lv == lvl else v) for lv, v in group[-1].beams]
        elif group:
            group[0].beams = [(lv, v) for lv, v in group[0].beams if lv < lvl]
    run: list[Event] = []
    for e in out:
        if e.kind in ("note", "rest") and not e.chord and e.grace == "none":
            if e.tuplet_mark == "start":
                run = [e]
            elif run and e.tup == run[0].tup:
                run.append(e)
                if e.tuplet_mark == "stop":
                    run = []
            else:
                run = []
    if len(run) > 1:
        run[-1].tuplet_mark = "stop"
    elif run:
        run[0].tuplet_mark = ""
    return out


def _sub(parent, tag, text=None, **attrs):
    el = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
    if text is not None:
        el.text = str(text)
    return el


METER_LENGTHS = 6                  # bar lengths tried for a page's meter: those of the model's likeliest meters
BAR_VOTE_FLOOR = math.log(0.01)    # the most one voice's bar can count against a length: a bar its notes fit under no
                                   # length (a misread) must not outvote the rest of the page


def _page_meter(rd: Reading, build) -> str | None:
    """The meter of the page's opening bars, decoded with the bars: for each bar length the model's meter head
    finds likely at the first system's bars (its distribution is the prior; a printed meter it reads near-certain),
    the page is read through the rhythm decode at that length, and the meter whose prior and fit together score
    best is the page's -- a page that prints no meter (a continuation page) gets the one its bars add up to: time
    left over before a bar's end is priced as missed notes, time past it as notes past the bar line. Each voice's
    bar votes with a floor (BAR_VOTE_FLOOR), so misread bars do not decide the meter."""
    import numpy as np

    from .vocab import TIMES
    L = rd.layout
    if not L.systems:
        return None
    heads = [rd.tok.get(L.staves[k].bars[0]) for k in L.systems[0].staves if L.staves[k].bars]
    dists = [np.asarray(r["time_p"], float) for r in heads if r is not None and "time_p" in r]
    if not dists:
        return None
    p = np.mean(dists, axis=0)
    meters = [(t, _time_len(t), float(p[i])) for i, t in enumerate(TIMES) if t not in (None, "other")]
    z = sum(m[2] for m in meters) or 1.0
    lengths: dict = {}
    for t, ln, pr in sorted(meters, key=lambda m: -m[2]):
        if ln not in lengths and len(lengths) < METER_LENGTHS:
            lengths[ln] = t
    runs = {ln: build(t)[2] for ln, t in lengths.items()}
    # each voice's bar votes: its decode's score at a length against its best length, floored (robust to bars that
    # fit no length); the same bars are decoded under every length, in the same order
    n = min(len(v) for v in runs.values())
    fit = {ln: 0.0 for ln in runs}
    for i in range(n):
        best = max(runs[ln][i] for ln in runs)
        for ln in runs:
            fit[ln] += max(runs[ln][i] - best, BAR_VOTE_FLOOR)
    return max((m for m in meters if m[1] in fit),
               key=lambda m: math.log(max(m[2] / z, 1e-9)) + fit[m[1]])[0]


# part names, decoded: which OCR'd text names which part. A label ends left of its part's first symbols (the clef),
# centred on its staves: measured on the benchmark's quartet pages (aggregate), the gap from a label's right edge
# to the clef is 2.2-3.0 staff spaces on renders and 2.9-7.6 on scans (p10-p90), its centre at 0.40-0.58 of the
# part's height; texts in the music (directions, fingerings, lyrics) sit 15-150 staff spaces past the clef
LABEL_GAP = (3.5, 2.5)           # the staff's first symbol - the label's right edge, staff spaces: centre, spread
LABEL_INTO = 0.5                 # spread where the text reaches into the staves (a negative gap: it is in the music)
LABEL_HEIGHT = (0.5, 0.3)        # the label's centre down its part's staves (0 the top line, 1 the bottom): centre, spread
LABEL_ROLE_LOGP = {"label": 0.0, "labelAbbr": math.log(0.7)}     # the OCR's own role for the text; any other role:
OTHER_ROLE_LOGP = math.log(0.05)                                 # (a name the OCR filed as a direction, say)
NAME_LOGP = {"exact": 0.0, "alias": 0.0, "context": 0.0, "subset": math.log(0.9), "fuzzy": math.log(0.7)}
UNKNOWN_NAME_LOGP = math.log(0.01)   # the text names no instrument the table knows (a lyric, a bar number, noise)
NO_NAME_LOGP = math.log(0.02)        # a part the page names nowhere
NOT_NAMES = ("pgHead_title", "pgHead_composer", "pgHead_subtitle", "pgFoot")


def _part_names(rd: Reading, plans: list, texts: list) -> list[str | None]:
    """Each part's name, decoded jointly: every OCR'd text could name any part, scored by where it sits against
    that part's staves (on any system), what it says (an instrument the table knows, instruments.match) and the
    role the OCR gave it; the parts and texts are paired one-to-one for the best total, a part taking no name where
    no text beats NO_NAME_LOGP. A bar number over a staff, a lyric under it, a stray read of noise lose on where
    they sit and on what they say; one stray text no longer shifts the others' names."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from .instruments import clean_label, match
    L = rd.layout
    cands = [t for t in texts if str(t.get("text", "")).strip() and t.get("role") not in NOT_NAMES and "xyxy" in t]
    if not plans or not cands:
        return [None] * len(plans)
    context = [t["text"] for t in texts if t.get("role") in ("label", "labelAbbr")]
    said = []
    for t in cands:
        m = match(t["text"], context)
        said.append((NAME_LOGP.get(m.how, math.log(0.7)) if m else UNKNOWN_NAME_LOGP) +
                    LABEL_ROLE_LOGP.get(t.get("role"), OTHER_ROLE_LOGP))
    first: dict[int, float] = {}         # staff -> x of its first symbol (its clef, mostly)
    for sym in L.syms:
        if sym.staff >= 0 and sym.fam in ("clef", "keySig", "meterSig", "note", "rest") and sym.p >= 0.5:
            first[sym.staff] = min(first.get(sym.staff, math.inf), sym.box[0])
    S = np.full((len(plans), len(cands)), -1e9)
    for n, pl in enumerate(plans):
        for sy in L.systems:
            ks = [sy.staves[pos] for pos in pl.staves if pos < len(sy.staves)]
            st = [L.staves[k] for k in ks]
            if not st:
                continue
            top, bot = min(s.y0 for s in st), max(s.y1 for s in st)
            sp = max(1.0, sorted(s.sp for s in st)[len(st) // 2])
            x0 = min(first.get(k, L.staves[k].x0) for k in ks)
            for j, t in enumerate(cands):
                tx0, ty0, tx1, ty1 = t["xyxy"]
                gap = (x0 - tx1) / sp
                into = (ty0 + ty1) / 2 - top
                h = into / max(1.0, bot - top)
                where = -0.5 * ((gap - LABEL_GAP[0]) / (LABEL_GAP[1] if gap >= 0 else LABEL_INTO)) ** 2 \
                    - 0.5 * ((h - LABEL_HEIGHT[0]) / LABEL_HEIGHT[1]) ** 2
                S[n, j] = max(S[n, j], where + said[j])
    P, T = S.shape
    cost = np.full((P, T + P), 1e9)
    cost[:, :T] = -S
    cost[np.arange(P), T + np.arange(P)] = -NO_NAME_LOGP
    names: list[str | None] = [None] * P
    for i, j in zip(*linear_sum_assignment(cost)):
        if j < T:
            names[i] = clean_label(str(cands[j]["text"]).strip())
    return names


def write(rd: Reading, texts: list[dict] | None = None, title: str | None = None, image=None) -> str:
    """``image``: the page (PIL image or grey array), for what is read off the ink here (dashes after words)."""
    L = rd.layout
    plans = _plan_parts(rd)
    root = ET.Element("score-partwise", version="4.0")
    texts = texts or []
    t_title = title or next((t["text"] for t in texts if t.get("role") == "pgHead_title"), None)
    if t_title:
        _sub(_sub(root, "work"), "work-title", t_title)
    composer = next((t["text"] for t in texts if t.get("role") == "pgHead_composer"), None)
    ident = _sub(root, "identification")
    if composer:
        _sub(ident, "creator", composer, type="composer")
    enc = _sub(ident, "encoding")
    _sub(enc, "software", "copisteria")
    plist = _sub(root, "part-list")
    names = _part_names(rd, plans, texts)
    groups = _part_groups(rd, plans)
    for n, pl in enumerate(plans):
        for g, (a, b, sym) in enumerate(groups):
            if a == n:
                pg = _sub(plist, "part-group", type="start", number=g + 1)
                _sub(pg, "group-symbol", sym)
        sp_ = _sub(plist, "score-part", id=f"P{n + 1}")
        _sub(sp_, "part-name", names[n] or f"Part {n + 1}")
        for g, (a, b, sym) in enumerate(groups):
            if b == n:
                _sub(plist, "part-group", type="stop", number=g + 1)

    # every part's measures first (as events), so ties can be resolved across bars before writing
    columns = []                                         # (system, column)
    for s_i, sy in enumerate(L.systems):
        ncol = 1 + max((max(c) for c in sy.columns if c), default=-1)
        columns += [(s_i, c) for c in range(ncol)]
    single_part = len(plans) == 1

    def build(opening: str | None):
        """Every part's measures as events (ties are resolved across them next); ``opening``: the meter of the page's
        opening bars, over the bar heads' own reading (None: as read). Returns (measures, durations, the rhythm
        decode's total log-score)."""
        all_durs: set = set()
        built = []
        fit: list = []
        for n, pl in enumerate(plans):
            state = [{"key": 0, "time": "4/4", "clef": "G2"} for _ in pl.staves]
            first = [True] * len(pl.staves)
            measures = []
            prev_events = [None] * len(pl.staves)
            for s_i, c in columns:
                sy = L.systems[s_i]
                m = {"staves": [], "time": None, "key": None}
                for k_i, pos in enumerate(pl.staves):
                    if pos >= len(sy.staves):
                        m["staves"].append(None)
                        continue
                    staff = sy.staves[pos]
                    cols = sy.columns[pos]
                    bars = [j for j, cc in enumerate(cols) if cc == c]
                    if not bars:
                        m["staves"].append(None)
                        continue
                    j = bars[0]
                    state[k_i] = _bar_state(rd, staff, j, state[k_i])
                    if first[k_i]:
                        first[k_i] = False
                        if opening is not None:
                            state[k_i]["time"] = opening
                    ev, dirs, marks = _events(rd, staff, j, state[k_i]["clef"], state[k_i]["key"], texts)
                    for j2 in bars[1:]:            # two boxes of this staff under one column: one bar
                        ev2, dirs2, marks2 = _events(rd, staff, j2, state[k_i]["clef"], state[k_i]["key"], texts)
                        ev += ev2; dirs += dirs2
                        marks["right"] = marks2["right"] or marks["right"]
                        marks["endings"] += marks2["endings"]
                        marks["end_clef"] = marks2["end_clef"] or marks["end_clef"]
                        marks["repeat_bar"] |= marks2["repeat_bar"]
                    if marks["repeat_bar"] and not ev and prev_events[k_i]:
                        ev = [Event(**{**e.__dict__, "tie": False, "tie_stop": False, "marks": [], "beams": [],
                                       "slurs": [], "pre": [], "post": [], "copied": True})
                              for e in prev_events[k_i]]
                    vs = _voices(ev, L.staves[staff].sp, single_staff_part=len(pl.staves) == 1,
                                 bar_len=_time_len(state[k_i]["time"]), score=fit)
                    prev_events[k_i] = ev or prev_events[k_i]
                    m["staves"].append({"staff": staff, "bar": j, "voices": vs, "dirs": dirs, "marks": marks,
                                        "state": dict(state[k_i])})
                    if marks["end_clef"]:                      # a clef inside the bar holds for the bars after it
                        state[k_i]["clef"] = marks["end_clef"]
                    for evs in vs.values():
                        all_durs.update(e.dur for e in evs)
                measures.append(m)
            built.append(measures)
        return built, all_durs, fit

    built, all_durs, _ = build(_page_meter(rd, build))

    # ties: a tied note's partner is the next note of its voice with the same step and octave (this bar or next)
    for measures in built:
        for k_i in range(max((len(m["staves"]) for m in measures), default=0)):
            streams: dict[int, list[Event]] = {}
            for m in measures:
                st = m["staves"][k_i] if k_i < len(m["staves"]) else None
                if st is None:
                    continue
                for v, evs in st["voices"].items():
                    streams.setdefault(v, []).extend(evs)
            for v, evs in streams.items():
                for a_i, e in enumerate(evs):
                    if e.kind != "note" or not e.tie:
                        continue
                    partner = None
                    for f in evs[a_i + 1:a_i + 12]:
                        if f.kind == "note" and not f.chord and f.step == e.step and f.octave == e.octave:
                            partner = f
                            break
                        if f.kind == "note" and f.chord and f.step == e.step and f.octave == e.octave:
                            partner = f
                            break
                    if partner is None:
                        e.tie = False
                    else:
                        partner.tie_stop = True
                        partner.alter = e.alter

    mask = None
    if image is not None:
        import numpy as np

        from .front import ink_mask
        mask = ink_mask(np.asarray(image.convert("L") if hasattr(image, "convert") else image))
    _attach(rd, built, plans, texts, mask)

    # each bar column is as long as most staves make it; a voice running past that is cut there
    bar_ends = _column_lengths(built)
    for measures in built:
        for m_i, m in enumerate(measures):
            for st in m["staves"]:
                if st is not None and bar_ends[m_i] is not None:
                    for v, evs in st["voices"].items():
                        st["voices"][v] = _chop(evs, bar_ends[m_i])
                        all_durs.update(e.dur for e in st["voices"][v])
    all_durs.update(n for n in bar_ends if n)

    all_durs |= {_time_len(m["staves"][0]["state"]["time"]) for ms in built for m in ms
                 if m["staves"] and m["staves"][0]}
    div = 1
    for d in all_durs:
        if d:
            div = div * d.denominator // math.gcd(div, d.denominator)
    div = min(div, 10080)

    def units(q: Fraction) -> int:
        return max(0, round(q * div))

    for n, (pl, measures) in enumerate(zip(plans, built)):
        part = _sub(root, "part", id=f"P{n + 1}")
        last = {"key": None, "time": None, "clefs": [None] * len(pl.staves)}
        meter_shown = False                  # the meter in effect is one the page prints (else written as not)
        number = 0
        pending_endings: list = []
        for m_i, m in enumerate(measures):
            number += 1
            staves = m["staves"]
            first = next((s for s in staves if s), None)
            meas = _sub(part, "measure", number=number)
            if first is None:
                if m_i == 0:
                    _sub(_sub(meas, "attributes"), "divisions", div)
                continue
            key, time = first["state"]["key"], first["state"]["time"]
            clefs = [s["state"]["clef"] if s else last["clefs"][i] for i, s in enumerate(staves)]
            attrs = None
            if m_i == 0 or key != last["key"] or time != last["time"] or clefs != last["clefs"] or \
                    first["marks"]["multirest"]:
                attrs = _sub(meas, "attributes")
                if m_i == 0:
                    _sub(attrs, "divisions", div)
                if m_i == 0 or key != last["key"]:
                    _sub(_sub(attrs, "key"), "fifths", key)
                # the meter is written from the first bar on, so the bars are measured by it; where the page does
                # not print it there (a continuation page), it is written as not printed
                if (m_i == 0 or time != last["time"]) and time != "other":
                    printed = any(st_ is not None and st_["marks"]["meter"] for st_ in staves)
                    meter_shown = printed
                    t = _sub(attrs, "time", **({} if printed else {"print-object": "no"}))
                    b, bt = time.split("/")
                    _sub(t, "beats", b); _sub(t, "beat-type", bt)
                if m_i == 0 and len(pl.staves) > 1:
                    _sub(attrs, "staves", len(pl.staves))
                for i, cl in enumerate(clefs):
                    if cl is not None and (m_i == 0 or cl != last["clefs"][i]):
                        sign, line, octc = CLEF_XML.get(cl, ("G", 2, 0))
                        ce = _sub(attrs, "clef", number=i + 1) if len(pl.staves) > 1 else _sub(attrs, "clef")
                        _sub(ce, "sign", sign)
                        if line:
                            _sub(ce, "line", line)
                        if octc:
                            _sub(ce, "clef-octave-change", octc)
            last.update(key=key, time=time, clefs=clefs)
            mr = first["marks"]["multirest"]
            if mr and single_part and mr > 1:
                ms = _sub(attrs, "measure-style")
                _sub(ms, "multiple-rest", mr)
            bar_len = _time_len(time)
            bar_end = bar_ends[m_i] or bar_len          # the column's length: the meter's unless most staves run past it
            # left bar line: forward repeat, ending starts
            starts = [num for num, box in first["marks"]["endings"]]
            if first["marks"]["left"] or starts:
                bl = _sub(meas, "barline", location="left")
                if first["marks"]["left"]:
                    _sub(bl, "bar-style", "heavy-light")
                for num in starts[:1]:
                    _sub(bl, "ending", number=num, type="start")
                if first["marks"]["left"]:
                    _sub(bl, "repeat", direction="forward")
                pending_endings = starts[:1]
            # a bar is left short where notation makes it so: the opening bar of a piece (the page prints its meter
            # there), an upbeat; the bar before a final or repeat bar line, its complement
            opening = m_i == 0 and first["marks"]["meter"]
            keep_short = opening or first["marks"]["right"] in SHORT_RIGHT
            pos_q = Fraction(0)
            max_end = Fraction(0)
            for k_i, st in enumerate(staves):
                if st is None:
                    continue
                staff_no = k_i + 1
                vs = st["voices"]
                if not vs:
                    if pos_q:
                        _sub(_sub(meas, "backup"), "duration", units(pos_q))
                        pos_q = Fraction(0)
                    fw = _sub(meas, "forward")
                    _sub(fw, "duration", units(bar_end))
                    if len(pl.staves) > 1:
                        _sub(fw, "staff", staff_no)
                    pos_q = bar_end
                    continue
                for v_i, (v, evs) in enumerate(sorted(vs.items())):
                    if pos_q:
                        _sub(_sub(meas, "backup"), "duration", units(pos_q))
                        pos_q = Fraction(0)
                    voice_no = v + 4 * (staff_no - 1)
                    dirs = sorted(st["dirs"]) if v_i == 0 else []
                    # a chord member's directions go with its chord's root: written between the root and a
                    # <chord/> note they would break the chord (the member read as the next note, time shifted)
                    # (pre-directions before the root, post-directions after the chord's last note)
                    c_root, c_last, c_members = None, None, []
                    for e in evs + [None]:
                        if e is not None and e.kind == "note" and e.chord and c_root is not None:
                            c_root.pre += e.pre
                            e.pre = []
                            c_members.append(e)
                            c_last = e
                            continue
                        if c_last is not None:                # the chord just closed: its posts go after its last note
                            posts = c_root.post + [d for m in c_members for d in m.post]
                            c_root.post = []
                            for m in c_members:
                                m.post = []
                            c_last.post = posts
                        c_root, c_last, c_members = (e, None, []) if e is not None and e.kind in ("note", "rest") \
                            else (None, None, [])
                        if e is None:
                            break
                    for e in evs:
                        while dirs and not (e.kind == "note" and e.chord) and \
                                dirs[0][0] <= e.x + 0.5 * L.staves[st["staff"]].sp:
                            _direction(meas, dirs.pop(0), staff_no if len(pl.staves) > 1 else None)
                        dur = bar_end if e.kind == "mrest" else e.dur
                        if e.kind == "gap":
                            if dur > 0:
                                fw = _sub(meas, "forward")
                                _sub(fw, "duration", units(dur))
                                _sub(fw, "voice", voice_no)
                                if len(pl.staves) > 1:
                                    _sub(fw, "staff", staff_no)
                            elif dur < 0:
                                _sub(_sub(meas, "backup"), "duration", units(-dur))
                            pos_q += dur
                            continue
                        for kind, payload, below in e.pre:
                            _direction(meas, (e.x, kind, payload, below), staff_no if len(pl.staves) > 1 else None)
                        _note(meas, e, units(dur), voice_no, staff_no if len(pl.staves) > 1 else None,
                              whole_rest=not meter_shown)
                        for kind, payload, below in e.post:
                            _direction(meas, (e.x, kind, payload, below), staff_no if len(pl.staves) > 1 else None)
                        if not e.chord:
                            pos_q += dur
                    if v_i == 0 and pos_q < bar_end and not keep_short:
                        # time a staff's first voice leaves before the bar line (what the decode reads as notes the
                        # detector missed) is the bar's all the same: written as a rest that is not printed, so the
                        # bar has its length on every staff and nothing is drawn that the page does not show
                        ne = _sub(meas, "note", print_object="no")
                        _sub(ne, "rest")
                        _sub(ne, "duration", units(bar_end - pos_q))
                        _sub(ne, "voice", voice_no)
                        if len(pl.staves) > 1:
                            _sub(ne, "staff", staff_no)
                        pos_q = bar_end
                    max_end = max(max_end, pos_q)
                    for d in dirs:
                        _direction(meas, d, staff_no if len(pl.staves) > 1 else None)
            if opening and 0 < max_end < bar_len:
                meas.set("implicit", "yes")          # an upbeat: the bar is not counted
            for i, st_ in enumerate(staves):        # a clef changed inside the bar is printed already
                if st_ is not None and st_["marks"]["end_clef"]:
                    last["clefs"][i] = st_["marks"]["end_clef"]
            right = first["marks"]["right"]
            if right or pending_endings:
                bl = _sub(meas, "barline", location="right")
                if right:
                    _sub(bl, "bar-style", BAR_RIGHT[right])
                for num in pending_endings:
                    _sub(bl, "ending", number=num, type="stop")
                if right == "barLine_repeat_end":
                    _sub(bl, "repeat", direction="backward")
                pending_endings = []
            if mr and single_part and mr > 1:
                for _ in range(mr - 1):
                    number += 1
                    extra = _sub(part, "measure", number=number)
                    ne = _sub(extra, "note")
                    _sub(ne, "rest", measure="yes")
                    _sub(ne, "duration", units(bar_len))
                    _sub(ne, "voice", 1)
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + \
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" ' \
        '"http://www.musicxml.org/dtds/partwise.dtd">\n' + ET.tostring(root, encoding="unicode")


def _direction(meas, d, staff_no):
    x, kind, payload, below = d
    if kind == "clef":
        attrs = _sub(meas, "attributes")
        sign, line, octc = CLEF_XML.get(payload, ("G", 2, 0))
        ce = _sub(attrs, "clef", number=staff_no) if staff_no else _sub(attrs, "clef")
        _sub(ce, "sign", sign)
        if line:
            _sub(ce, "line", line)
        if octc:
            _sub(ce, "clef-octave-change", octc)
        return
    de = _sub(meas, "direction", placement="below" if below else "above")
    dt = _sub(de, "direction-type")
    if kind == "dyn":
        _sub(_sub(dt, "dynamics"), payload)
    elif kind == "wedge":
        _sub(dt, "wedge", type=payload)
    elif kind == "octave":
        typ, size = payload
        _sub(dt, "octave-shift", type=typ, size=size)
    elif kind == "pedal":
        _sub(dt, "pedal", type=payload, line="yes")
    elif kind in ("segno", "coda"):
        _sub(dt, kind)
    elif kind == "words":
        _sub(dt, "words", payload)
    elif kind == "words_dashes":
        _sub(dt, "words", payload)
        _sub(_sub(de, "direction-type"), "dashes", type="start", number=1)
    elif kind == "dashes_stop":
        _sub(dt, "dashes", type="stop", number=1)
    if staff_no:
        _sub(de, "staff", staff_no)


def _note(meas, e: Event, dur: int, voice: int, staff_no, whole_rest: bool = False):
    """``whole_rest``: a whole-bar rest's type is written (whole, as it is drawn in any meter) -- where the meter is
    written as not printed, so a reader that skips such a meter does not type the rest from its own default."""
    ne = _sub(meas, "note")
    if e.grace != "none":
        _sub(ne, "grace", **({"slash": "yes"} if e.grace == "unacc" else {}))
    if e.chord:
        _sub(ne, "chord")
    if e.kind in ("rest", "mrest"):
        _sub(ne, "rest", **({"measure": "yes"} if e.kind == "mrest" else {}))
    elif e.unpitched:
        up = _sub(ne, "unpitched")
        _sub(up, "display-step", e.step); _sub(up, "display-octave", e.octave)
    else:
        p = _sub(ne, "pitch")
        _sub(p, "step", e.step)
        if e.alter:
            _sub(p, "alter", e.alter)
        _sub(p, "octave", e.octave)
    if e.grace == "none":
        _sub(ne, "duration", dur)
    if e.tie_stop:
        _sub(ne, "tie", type="stop")
    if e.tie:
        _sub(ne, "tie", type="start")
    _sub(ne, "voice", voice)
    if e.kind == "mrest":
        if whole_rest:
            _sub(ne, "type", "whole")
    else:
        _sub(ne, "type", e.typ if e.typ not in ("256", "512") else e.typ + "th")
        for _ in range(e.dots):
            _sub(ne, "dot")
    if e.accidental:
        _sub(ne, "accidental", e.accidental)
    if e.tup not in ("none", "other") and e.grace == "none":
        a, n = e.tup.split("/")
        tm = _sub(ne, "time-modification")
        _sub(tm, "actual-notes", a); _sub(tm, "normal-notes", n)
    if e.stem and e.kind == "note":
        _sub(ne, "stem", e.stem)
    if staff_no:
        _sub(ne, "staff", staff_no)
    for level, val in e.beams:
        _sub(ne, "beam", val, number=level)
    nots = []
    if e.tie_stop:
        nots.append(("tied", {"type": "stop"}))
    if e.tie:
        nots.append(("tied", {"type": "start"}))
    for typ, num in e.slurs:
        nots.append(("slur", {"type": typ, "number": num}))
    if e.tuplet_mark:
        nots.append(("tuplet", {"type": e.tuplet_mark}))
    if e.arpeggiate:
        nots.append(("arpeggiate", {}))
    arts = [ARTIC[c] for c in e.marks if c in ARTIC] + [AFTER_MARK[c] for c in e.marks if c in AFTER_MARK]
    techs = [TECH[c] for c in e.marks if c in TECH]
    orns = [ORNAMENT[c] for c in e.marks if c in ORNAMENT]
    ferm = "fermata" in e.marks
    if nots or arts or techs or ferm or orns:
        nn = _sub(ne, "notations")
        for tag, at in nots:
            _sub(nn, tag, **at)
        if orns:
            oe = _sub(nn, "ornaments")
            for o_ in orns:
                _sub(oe, o_)
        if arts:
            ae = _sub(nn, "articulations")
            for a_ in arts:
                _sub(ae, a_)
        if techs:
            te = _sub(nn, "technical")
            for t_ in techs:
                _sub(te, t_)
        if ferm:
            _sub(nn, "fermata")
