"""Training targets: the front end's symbols against the renderer's labels (every label carries ``src``, what it
means in the score; docs/ARCHITECTURE.md describes the label format).

A symbol is *real* when a label of its family overlaps it (greedy one-to-one by IoU); a real symbol's targets are
that label's: class, and for notes / rests dots, staff position, sounding alteration, voice slot, chord, tie,
tuplet, grace, onset. Every bar (measure symbol) gets the clef / key / meter in effect at its start.
"""
from __future__ import annotations

from fractions import Fraction

from .front import Layout, iou
from .vocab import (ALTERS, CLEFS, CLS_ID, KEYS, N_VOICE, POS_MAX, TIMES, clef_id, family, gt_class, key_alter,
                    time_id, tuplet_id)

MATCH_IOU = 0.3
IGNORE = -100


def _xyxy(e: dict):
    x, y, w, h = e["bbox"]
    return (x, y, x + w, y + h)


def _frac(s) -> Fraction | None:
    try:
        return Fraction(str(s))
    except (ValueError, ZeroDivisionError):
        return None


def _clef_sign(src: dict) -> str:
    c = str(src.get("clef") or "")
    if not c:
        return "other"
    if c.startswith("perc") or c == "percussion":
        return "perc"
    o = int(src.get("octave") or 0)
    return c + {-1: "_8b", 1: "_8a", -2: "_15b", 2: "_15a"}.get(o, "")


def bar_states(lab: dict) -> dict:
    """measure label id -> {clef, key, time} in effect at the bar's start, and which of them the bar SETS: a sign
    printed in its first half (a cautionary sign at a bar's end belongs to the next system), or the first bar of
    its staff on the page."""
    els = lab["elements"]
    start = {(s["part"], s["staff"], s["system"]): s for s in lab.get("staves", [])}
    bars = [e for e in els if e["type"] == "measure" and e.get("src")]
    changes = sorted((e for e in els if e["type"] in ("clef", "keySig", "meterSig") and e.get("src")),
                     key=lambda e: _xyxy(e)[0])
    first: dict = {}
    for b in bars:
        s = b["src"]
        k = (s.get("part"), s.get("staff"))
        if k not in first or (s.get("system"), s.get("index")) < first[k][0]:
            first[k] = ((s.get("system"), s.get("index")), b["id"])
    firsts = {v[1] for v in first.values()}
    out = {}
    for b in bars:
        s = b["src"]
        key = (s.get("part"), s.get("staff"), s.get("system"))
        st = start.get(key)
        if st is None:
            continue
        t = str(st.get("time") or "4/4").split("/")
        state = {"clef": str(st.get("clef") or "G2"), "key": int(st.get("key") or 0),
                 "time": (t[0], t[1] if len(t) > 1 else "4")}
        sets = {"clef": False, "key": False, "time": False}
        bx0, _, bx1, _ = _xyxy(b)
        for c in changes:
            cs = c["src"]
            if (cs.get("part"), cs.get("staff"), cs.get("system")) != key:
                continue
            cx0, _, cx1, _ = _xyxy(c)
            cx = (cx0 + cx1) / 2
            here = bx0 <= cx <= bx1 and cx - bx0 < 0.5 * (bx1 - bx0)
            if not (cx < bx0 or here):
                continue
            if c["type"] == "clef":
                state["clef"] = _clef_sign(cs); sets["clef"] |= here
            elif c["type"] == "keySig" and cs.get("fifths") is not None:
                state["key"] = int(cs["fifths"]); sets["key"] |= here
            elif c["type"] == "meterSig" and cs.get("beats"):
                state["time"] = (str(cs["beats"]), str(cs.get("beat_type") or "4")); sets["time"] |= here
        if b["id"] in firsts:
            sets = {k: True for k in sets}
        state["sets"] = sets
        state["where"] = (s.get("part"), s.get("staff"), s.get("system"), str(s.get("number")))
        out[b["id"]] = state
    return out


def _bar_targets(T: dict, i: int, st: dict) -> None:
    T["key"][i] = KEYS.index(max(-7, min(7, st["key"]))) if st["sets"]["key"] else KEYS.index(None)
    T["time"][i] = time_id(*st["time"]) if st["sets"]["time"] else TIMES.index(None)
    T["clef"][i] = clef_id(st["clef"]) if st["sets"]["clef"] else CLEFS.index(None)


def targets(layout: Layout, lab: dict) -> dict:
    """Per symbol: lists aligned with layout.syms (IGNORE where a head does not apply)."""
    syms = layout.syms
    n = len(syms)
    gts = []
    for e in lab["elements"]:
        c = gt_class(e)
        if c is not None:
            gts.append((c, _xyxy(e), e))
    # greedy one-to-one matching within a family, best IoU first
    by_fam: dict[str, list[int]] = {}
    for g, (c, _, _) in enumerate(gts):
        by_fam.setdefault(family(c), []).append(g)
    pairs = []
    for s in syms:
        for g in by_fam.get(s.fam, ()):
            v = iou(s.box, gts[g][1])
            if v >= MATCH_IOU:
                pairs.append((v, s.i, g))
    pairs.sort(reverse=True)
    sym_gt, used = {}, set()
    for v, i, g in pairs:
        if i in sym_gt or g in used:
            continue
        sym_gt[i] = g; used.add(g)

    T = {k: [IGNORE] * n for k in ("real", "cls", "dots", "pos", "alter", "voice", "chord", "tie", "tup", "grace",
                                    "key", "time", "clef")}
    T["onset"] = [None] * n
    T["gt_id"] = [None] * n
    states = bar_states(lab)
    state_at = {st["where"]: st for st in states.values()}
    # chord groups: (part, staff, measure index, voice, onset) -> matched symbols in x order
    groups: dict[tuple, list[int]] = {}
    for s in syms:
        g = sym_gt.get(s.i)
        if s.synthetic:            # a bar the front end made: no reading to judge, but the bar heads apply
            if g is not None:
                T["gt_id"][s.i] = gts[g][2].get("id")
                st = states.get(gts[g][2].get("id"))
                if st is not None:
                    _bar_targets(T, s.i, st)
            continue
        T["real"][s.i] = int(g is not None)
        if g is None:
            continue
        c, _, e = gts[g]
        T["cls"][s.i] = CLS_ID[c]
        T["gt_id"][s.i] = e.get("id")
        src = e.get("src") or {}
        if s.fam in ("note", "rest"):
            T["dots"][s.i] = min(3, int(src.get("dots") or 0))
            sp = e.get("staff_position")
            if sp is not None:
                T["pos"][s.i] = max(-POS_MAX, min(POS_MAX, int(sp))) + POS_MAX
            vs = e.get("voice_slot")
            if vs is not None:
                T["voice"][s.i] = min(N_VOICE, max(1, int(vs))) - 1
            T["tup"][s.i] = tuplet_id(src.get("tm"))
            on = _frac(src.get("onset"))
            T["onset"][s.i] = float(on) if on is not None else None
            if s.fam == "note":
                a = src.get("alter")
                try:
                    a = int(round(float(a or 0)))
                except (TypeError, ValueError):
                    a = 0
                a = max(-2, min(2, a))
                st = state_at.get((src.get("part"), src.get("staff"), src.get("system"), str(src.get("measure"))))
                pitch = str(src.get("pitch") or "")
                if st is not None and pitch:
                    T["alter"][s.i] = ALTERS.index("key") if a == key_alter(pitch[0].upper(), st["key"]) \
                        else ALTERS.index(a)
                T["tie"][s.i] = int(bool(src.get("tie_start")))
                T["grace"][s.i] = 0 if not src.get("grace") else (2 if src.get("grace_slash") else 1)
                if not src.get("grace") and on is not None:
                    key = (src.get("part"), src.get("staff"), src.get("mi"), src.get("voice"), on)
                    groups.setdefault(key, []).append(s.i)
        if s.fam == "measure":
            st = states.get(e.get("id"))
            if st is not None:
                _bar_targets(T, s.i, st)
    # a chord's root is its lowest note (lowest staff position, then leftmost): 0; every other member: 1
    for members in groups.values():
        members.sort(key=lambda i: (T["pos"][i] if T["pos"][i] != IGNORE else 99, syms[i].cx))
        for k, i in enumerate(members):
            T["chord"][i] = int(k > 0)
    for s in syms:
        if s.fam == "note" and T["real"][s.i] == 1 and T["chord"][s.i] == IGNORE and T["grace"][s.i] == 0:
            T["chord"][s.i] = 0
    T["n_gt"] = len(gts)
    T["n_matched"] = len(sym_gt)
    return T
