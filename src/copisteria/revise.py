"""Structure from the readings: bar lines the detector missed, found because the music says a bar is there.

The front end cuts bars from what it sees -- a detected bar line, a stroke over the staff's height -- and on one
staff a stroke alone is not enough (a stem is one too). A bar line the detector missed then leaves one bar holding
two bars of music. Once the page is read, the readings say so: in such a bar the main voice's durations add up to
twice (or more) the meter in effect, and the running sum reaches exactly one bar between two notes, where a stroke
over the staff's full height stands that no note's stem explains. There the bar line is added, and the page is read
again with it (``read_revised``). Ink-path pages only (the cuts are the joint segmentation's).
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

OVERFULL = Fraction(7, 4)          # a voice holding this many bars of the meter is a candidate for a missed line
STROKE = 0.9                       # share of the staff's height a bar-line stroke covers


def _bar_len(time) -> Fraction | None:
    try:
        a, b = (int(v) for v in str(time).split("/"))
        return Fraction(4 * a, b)
    except (ValueError, ZeroDivisionError):
        return None


def missed_bar_lines(rd, gray) -> dict:
    """{top staff of a system: [x of each bar line to add]}."""
    import numpy as np

    from . import front
    from .write import _duration, _type_of
    L = rd.layout
    if not L.ink or gray is None:
        return {}
    mask = front.ink_mask(np.asarray(gray))
    tok = rd.tok
    # the meter in effect per bar column, carried over the page from the bars that print one
    time_at: dict = {}
    cur = None
    for n, sy in enumerate(L.systems):
        ncol = 1 + max((max(c) for c in sy.columns if c), default=0)
        for col in range(ncol):
            for pos, k in enumerate(sy.staves):
                st = L.staves[k]
                for j, i in enumerate(st.bars):
                    if sy.columns[pos][j] == col and i in tok and tok[i].get("time", {}).get("v") not in (None, "other"):
                        cur = tok[i]["time"]["v"]
            time_at[(n, col)] = cur
    out: dict = defaultdict(list)
    for n, sy in enumerate(L.systems):
        S = [L.staves[k] for k in sy.staves]
        for pos, k in enumerate(sy.staves):
            st = L.staves[k]
            sp = st.sp
            for j, bi in enumerate(st.bars):
                blen = _bar_len(time_at.get((n, sy.columns[pos][j])))
                if not blen:
                    continue
                bx0, _, bx1, _ = L.syms[bi].box
                evs = defaultdict(list)
                heads = []
                for s in L.syms:
                    if s.staff != k or s.bar != j or s.i not in tok or tok[s.i]["real"]["p"] < 0.5:
                        continue
                    r = tok[s.i]
                    c = r["cls"]["v"]
                    fam = c.split("_", 1)[0]
                    if fam not in ("note", "rest"):
                        continue
                    if fam == "note":
                        heads.append(s.box)
                        if r.get("chord", 0) > 0.5 or r.get("grace", {}).get("v", "none") != "none":
                            continue
                    d = _duration(_type_of(c), r["dots"]["v"], r["tup"]["v"])
                    evs[r["voice"]["v"]].append((s.cx, s.box, d))
                for v, es in evs.items():
                    es.sort(key=lambda e: e[0])
                    if sum(e[2] for e in es) < OVERFULL * blen:
                        continue
                    cum = Fraction(0)
                    for a, b in zip(es, es[1:]):
                        cum += a[2]
                        if cum % blen:
                            continue
                        # the running sum fills a whole number of bars here: a stroke over the staff's height between
                        # the two notes, clear of every note head's column (a stem stands next to its head)
                        lo, hi = int(a[1][2]), int(b[1][0])
                        xs = [x for x in range(lo, hi + 1)
                              if front._ink_run(mask, x, st.y0 + 1, st.y1 - 1, reach=1) >= STROKE
                              and not any(h[0] - 0.4 * sp <= x <= h[2] + 0.4 * sp for h in heads)]
                        if not xs:
                            continue
                        x = sorted(xs)[len(xs) // 2]
                        others = [o for o in S if o is not st]
                        if not others or sum(front._ink_run(mask, x, o.y0 + 1, o.y1 - 1, reach=2) >= STROKE
                                             for o in others) >= len(others) / 2:
                            out[sy.staves[0]].append(float(x))
    return {k: sorted(set(v)) for k, v in out.items()}


def read_revised(dets, width, height, model, device="cpu", attention=True, image=None, **kw):
    """read.read, then again with the bar lines the first reading shows were missed (if any)."""
    from . import read
    rd = read.read(dets, width, height, model, device=device, attention=attention, image=image, **kw)
    cuts = missed_bar_lines(rd, image)
    if cuts:
        rd = read.read(dets, width, height, model, device=device, attention=attention, image=image, cuts=cuts, **kw)
        rd.revised = cuts
    return rd
