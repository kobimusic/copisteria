"""The minimal front end: detector boxes -> symbols -> staves, bars and systems. No music in here.

Only geometry is decided here, and only the parts the writer cannot do without:

* **merge** -- boxes of one family that overlap (IoU >= MERGE_IOU) are one symbol; its class posterior is the
  boxes' confidence mass per class, its P(real) their noisy-or. Different families never merge: two readings of
  one glyph in different families stay two symbols, and the model decides which one is real.
* **staves** -- from the staff lines in the page's ink (five long thin runs at one spacing), checked against the
  staff space of the detector's ``measure`` boxes; a row of measure boxes no ink staff explains is a staff too.
  Without the page image (or when the line finder fails on it), a staff is a chain of side-by-side, vertically
  overlapping measure boxes, and a gap between two of them that holds notes or rests becomes a bar of its own.
* **systems** -- two consecutive staves share a system when an ink run crosses the gap between them at the
  system's left (its opening line, brace or bracket) and are separate where that gap is white; in between, a group
  symbol spanning them or the ``links`` model decides.
* **bars** -- a system's bar lines are decided once for all its staves: where most staves show a bar line (a
  detection and a vertical stroke over the staff's full height, or a stroke through every staff); slivers and
  empty edge stretches fold away.
* **assignment** -- every other symbol goes to the staff it is nearest to (a note to the staff where its own
  staff-position reading fits), and to the bar its centre falls in.

Everything musical -- which readings are real, durations, pitches, voices, keys, meters -- is the model's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .vocab import CLASSES, CLS_ID, family

MERGE_IOU = 0.5
MEASURE_MIN_P = 0.25          # measure boxes below this do not build staves
MEASURE_MIN_W = 2.0           # nor boxes narrower than this many staff spaces
MEASURE_LOW_P = 0.08          # ... except where no confident staff is: then down to this
GRP_MIN_P = 0.3
ALIGN_TOL_SP = 2.0            # bar lines this close (in staff spaces) may be partners
ALIGN_FRAC = 0.7
ASSIGN_MAX_SP = 10.0          # farther than this from every staff: not part of the score


Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@dataclass
class Sym:
    i: int
    cls: str                       # the posterior's argmax
    post: dict                     # class id -> probability (sums to 1 over the classes the boxes read)
    p: float                       # P(real): noisy-or of the merged boxes' confidences
    n: int                         # boxes merged
    box: Box
    attrs: dict = field(default_factory=dict)
    attr_p: dict = field(default_factory=dict)
    attr_dist: dict = field(default_factory=dict)
    staff: int = -1                # page-global staff index
    bar: int = -1                  # bar index within its staff
    synthetic: bool = False        # a bar the front end made from a gap (no detector box)
    det: int = -1                  # the detection it was made from (its most confident box)
    cand: bool = False             # a sub-threshold candidate of the detector's

    @property
    def fam(self) -> str:
        return family(self.cls)

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    def to_dict(self) -> dict:
        return {"i": self.i, "cls": self.cls, "post": {str(k): round(v, 4) for k, v in self.post.items()},
                "p": round(self.p, 4), "n": self.n, "box": [round(v, 1) for v in self.box], "attrs": self.attrs,
                "attr_p": self.attr_p, "attr_dist": self.attr_dist, "staff": self.staff, "bar": self.bar,
                "synthetic": self.synthetic, "cand": self.cand}


@dataclass
class Staff:
    bars: list[int]                # symbol index of each bar's measure box, x order
    sp: float                      # staff space (px)
    y0: float
    y1: float
    x0: float
    x1: float
    system: int = -1
    pos: int = -1                  # index within its system (0 = top)

    @property
    def mid(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class System:
    staves: list[int]
    columns: list[list[int]] = field(default_factory=list)   # per staff (system order): bar index -> column


@dataclass
class Layout:
    syms: list[Sym]
    staves: list[Staff]
    systems: list[System]
    sp: float
    width: float
    height: float

    def bar_box(self, staff: int, bar: int) -> Box:
        return self.syms[self.staves[staff].bars[bar]].box

    def column(self, staff: int, bar: int) -> int:
        st = self.staves[staff]
        return self.systems[st.system].columns[st.pos][bar]


# ------------------------------------------------------------------------------------------------- merge
def merge(dets: list[dict], offset: int = 0) -> list[Sym]:
    """Boxes of one family with IoU >= MERGE_IOU -> one symbol, greedy from the most confident box. A symbol's
    ``det`` is its head box's index in ``dets`` plus ``offset``."""
    order = sorted(range(len(dets)), key=lambda k: -float(dets[k]["conf"]))
    groups: list[list[int]] = []
    heads: list[tuple[str, Box]] = []
    for k in order:
        d = dets[k]
        if d["cls"] not in CLS_ID:
            continue
        fam, box = family(d["cls"]), tuple(float(v) for v in d["xyxy"])
        for g, (gf, gb) in enumerate(heads):
            if gf == fam and iou(gb, box) >= MERGE_IOU:
                groups[g].append(k)
                break
        else:
            groups.append([k]); heads.append((fam, box))
    syms = []
    for g in groups:
        head = dets[g[0]]
        mass: dict[int, float] = {}
        miss = 1.0
        for k in g:
            c = float(dets[k]["conf"])
            mass[CLS_ID[dets[k]["cls"]]] = mass.get(CLS_ID[dets[k]["cls"]], 0.0) + c
            miss *= 1.0 - min(c, 0.999)
        tot = sum(mass.values())
        post = {c: m / tot for c, m in mass.items()}
        best = max(post, key=post.get)
        syms.append(Sym(i=len(syms), cls=CLASSES[best], post=post, p=1.0 - miss, n=len(g), box=tuple(float(v) for v in head["xyxy"]),
                        attrs=dict(head.get("attrs") or {}), attr_p=dict(head.get("attr_p") or {}),
                        attr_dist=dict(head.get("attr_dist") or {}), det=g[0] + offset))
    return syms


# ------------------------------------------------------------------------------------------------- staves
def _staves(syms: list[Sym]) -> list[Staff]:
    """Confident measure boxes make staves; where none sits, weak ones (down to MEASURE_LOW_P) may."""
    def wide(s):     # a bar is at least MEASURE_MIN_W staff spaces wide (its own height is four)
        return s.box[2] - s.box[0] >= MEASURE_MIN_W * (s.box[3] - s.box[1]) / 4

    staves = _rows([s for s in syms if s.fam == "measure" and s.p >= MEASURE_MIN_P and wide(s)])
    # weak boxes stay clear of the confident staves: staves sit at least ~6 staff spaces apart, centre to centre
    weak = [s for s in syms if s.fam == "measure" and MEASURE_LOW_P <= s.p < MEASURE_MIN_P and wide(s) and
            not any(abs(s.cy - st.mid) < 5 * st.sp for st in staves)]
    for st in _rows(weak):          # a weak staff must hold music of its own
        held = sum(1 for s in syms if s.fam in ("note", "rest", "clef") and s.p >= 0.3 and st.x0 <= s.cx <= st.x1
                   and st.y0 - 0.5 * st.sp <= s.cy <= st.y1 + 0.5 * st.sp)
        if held >= 3:
            staves.append(st)
    staves.sort(key=lambda st: st.mid)
    return staves


def _rows(bars: list[Sym]) -> list[Staff]:
    if not bars:
        return []
    parent = list(range(len(bars)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(bars)):
        A = bars[a].box
        for b in range(a + 1, len(bars)):
            B = bars[b].box
            ov = min(A[3], B[3]) - max(A[1], B[1])
            if ov < 0.5 * min(A[3] - A[1], B[3] - B[1]):
                continue
            # side by side (a missed bar between them is fine), not stacked readings of one bar
            xov = min(A[2], B[2]) - max(A[0], B[0])
            if xov < 0.5 * min(A[2] - A[0], B[2] - B[0]):
                parent[find(a)] = find(b)
    rows: dict[int, list[Sym]] = {}
    for a, s in enumerate(bars):
        rows.setdefault(find(a), []).append(s)
    staves = []
    for members in rows.values():
        members.sort(key=lambda s: s.cx)
        kept: list[Sym] = []
        for s in members:          # two boxes over most of one bar: the likelier one is the bar
            if kept:
                q = kept[-1]
                ov = min(q.box[2], s.box[2]) - max(q.box[0], s.box[0])
                if ov > 0.5 * min(q.box[2] - q.box[0], s.box[2] - s.box[0]):
                    if s.p > q.p:
                        kept[-1] = s
                    continue
            kept.append(s)
        y0 = sorted(s.box[1] for s in kept)[len(kept) // 2]
        y1 = sorted(s.box[3] for s in kept)[len(kept) // 2]
        staves.append(Staff(bars=[s.i for s in kept], sp=max(1.0, (y1 - y0) / 4), y0=y0, y1=y1,
                            x0=kept[0].box[0], x1=kept[-1].box[2]))
    staves.sort(key=lambda st: st.mid)
    return staves


def _fill_gaps(syms: list[Sym], staves: list[Staff]) -> None:
    """Stretches of a staff with notes or rests outside every bar box are bars the detector missed: a gap between
    two bars, or the run before the first / after the last. Detected bar lines inside a stretch split it."""
    _assign(syms, staves)
    for k, st in enumerate(staves):
        boxes = [syms[i].box for i in st.bars]
        loose = [s for s in syms if s.staff == k and s.fam in ("note", "rest") and s.p >= 0.25 and
                 not any(b[0] - 0.5 * st.sp <= s.cx <= b[2] + 0.5 * st.sp for b in boxes)]
        if not loose:
            continue
        spans = []                                  # (x0, x1) stretches between bars
        edges = [(-math.inf, boxes[0][0])] + [(a[2], b[0]) for a, b in zip(boxes, boxes[1:])] + \
                [(boxes[-1][2], math.inf)]
        for x0, x1 in edges:
            inside = [s for s in loose if x0 <= s.cx <= x1]
            if not inside:
                continue
            if x0 == -math.inf:
                x0 = min(s.box[0] for s in inside) - 1.5 * st.sp
            if x1 == math.inf:
                x1 = max(s.box[2] for s in inside) + 1.5 * st.sp
            cuts = sorted(s.cx for s in syms if s.staff == k and s.fam == "barLine" and s.p >= 0.5
                          and x0 + st.sp < s.cx < x1 - st.sp)
            if x1 - x0 < 1.5 * st.sp:                 # a sliver between two boxes: its notes join a neighbour
                continue
            xs = [x0, *cuts, x1]
            spans += [(a, b) for a, b in zip(xs, xs[1:]) if any(a <= s.cx <= b for s in inside)]
        new = []
        for x0, x1 in spans:
            _, y0, y1, _ = _band(syms, st, (x0 + x1) / 2)
            s = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                    box=(x0, y0, x1, y1), synthetic=True)
            syms.append(s); new.append(s.i)
        if new:
            st.bars = sorted(st.bars + new, key=lambda i: syms[i].cx)
            st.x0, st.x1 = min(st.x0, syms[st.bars[0]].box[0]), max(st.x1, syms[st.bars[-1]].box[2])
    for s in syms:                                  # assigned again once the systems exist
        s.staff, s.bar = -1, -1


def _snap_bars(syms: list[Sym], staves: list[Staff]) -> None:
    """The boundary between two neighbouring bars sits on the bar line the detector saw there (measure boxes
    are loose at their edges); without one, overlapping boxes meet halfway."""
    lines = [s for s in syms if s.fam == "barLine" and s.p >= 0.4]
    for st in staves:
        for a, b in zip(st.bars, st.bars[1:]):
            A, B = syms[a], syms[b]
            lo, hi = min(A.box[2], B.box[0]) - 4 * st.sp, max(A.box[2], B.box[0]) + 4 * st.sp
            mid = (A.box[2] + B.box[0]) / 2
            near = [s for s in lines if lo <= s.cx <= hi and st.y0 - st.sp <= s.cy <= st.y1 + st.sp]
            if near:
                x = min(near, key=lambda s: abs(s.cx - mid)).cx
            elif A.box[2] > B.box[0]:
                x = mid
            else:
                continue
            if A.box[0] < x < B.box[2]:
                A.box = (A.box[0], A.box[1], x, A.box[3])
                B.box = (x, B.box[1], B.box[2], B.box[3])
        # a filled-in bar the snapping squeezed to a sliver is no bar: the next one takes its stretch
        keep = []
        for k, i in enumerate(st.bars):
            s = syms[i]
            if s.synthetic and s.box[2] - s.box[0] < 1.5 * st.sp and k + 1 < len(st.bars):
                n = syms[st.bars[k + 1]]
                n.box = (s.box[0], n.box[1], n.box[2], n.box[3])
                continue
            keep.append(i)
        st.bars = keep


def _aligned(syms: list[Sym], a: Staff, b: Staff) -> bool:
    """Bar lines of the two staves line up: over their common stretch, at least ALIGN_FRAC of each staff's
    detected internal bar lines have a partner on the other within ALIGN_TOL_SP, and the partners sit at one
    consistent offset (a warped page shifts a whole system's staves together; two systems whose bars merely
    happen to fall near each other scatter)."""
    sp = (a.sp + b.sp) / 2
    lo, hi = max(a.x0, b.x0) - sp, min(a.x1, b.x1) + sp

    def edges(st):
        return [syms[k].box[2] for k in st.bars[:-1] if not syms[k].synthetic and lo <= syms[k].box[2] <= hi]

    ea, eb = edges(a), edges(b)
    if not ea or not eb:
        return False
    tol = ALIGN_TOL_SP * sp
    diffs = [min(eb, key=lambda y: abs(x - y)) - x for x in ea]
    diffs = [d for d in diffs if abs(d) <= tol]
    hit2 = sum(1 for y in eb if any(abs(x - y) <= tol for x in ea))
    if len(diffs) < ALIGN_FRAC * len(ea) or hit2 < ALIGN_FRAC * len(eb):
        return False
    if len(diffs) == 1:
        return abs(diffs[0]) <= 0.5 * sp
    med = sorted(diffs)[len(diffs) // 2]
    return max(abs(d - med) for d in diffs) <= 0.6 * sp


def _braced(syms: list[Sym], a: Staff, b: Staff) -> bool:
    for s in syms:
        if s.fam != "grpSym" or s.p < GRP_MIN_P:
            continue
        x0, y0, x1, y1 = s.box
        if y0 <= a.mid + a.sp and y1 >= b.mid - b.sp and x1 >= min(a.x0, b.x0) - 6 * a.sp and \
                x0 <= min(a.x0, b.x0) + 3 * a.sp:
            return True
    return False


INK_LINK = 0.85               # share of the gap a vertical ink run must cover to join two staves
INK_SPLIT = 0.4               # below this (and no bar line through the gap): two systems


def ink_mask(gray):
    """Ink pixels of a page: darker than 35 % of the way from its paper (median) to its ink (1st percentile), so a
    thin anti-aliased line on a small render counts as well as a thick stroke on a 1-bit scan."""
    import numpy as np
    paper, ink = np.percentile(gray, 50), np.percentile(gray, 1)
    return gray < paper - 0.35 * max(1.0, paper - ink)


def _ink_run(mask, x: float, y0: float, y1: float, reach: int = 2) -> float:
    """Share of the rows between y0 and y1 that hold ink within ``reach`` px of column x."""
    H, W = mask.shape
    a, b = max(0, int(round(y0))), min(H, int(round(y1)))
    xa, xb = max(0, int(round(x)) - reach), min(W, int(round(x)) + reach + 1)
    if b - a < 2 or xb <= xa:
        return 1.0 if b - a < 2 else 0.0
    return float(mask[a:b, xa:xb].any(axis=1).mean())


def _ink_links(gray, syms: list[Sym], a: Staff, b: Staff, page_left: float) -> tuple[float, float]:
    """(left, bars): how much of the gap between two staves a vertical ink run crosses at their left end (the
    system's opening line, brace or bracket -- the measure boxes start after the clef, so from the page's leftmost
    staff start on), and the share of a's detected bar lines whose ink runs on into b."""
    sp = (a.sp + b.sp) / 2
    y0, y1 = a.y1 + 1, b.y0 - 1
    lo, hi = int(min(page_left, a.x0, b.x0) - 3 * sp), int(min(a.x0, b.x0) + sp)
    left = max((_ink_run(gray, x, y0, y1) for x in range(max(0, lo), hi + 1, 2)), default=0.0)
    lines = [s for s in syms if s.fam == "barLine" and s.p >= 0.5 and s.staff < 0 and
             a.y0 - sp <= s.cy <= a.y1 + sp and max(a.x0, b.x0) <= s.cx <= min(a.x1, b.x1)]
    bars = (sum(_ink_run(gray, s.cx, y0, y1) >= INK_LINK for s in lines) / len(lines)) if lines else 0.0
    return left, bars


def _systems(syms: list[Sym], staves: list[Staff], gray=None, ink: bool = False) -> list[System]:
    from . import links
    w = links.load()
    if w is not None and len(staves) > 1:
        gaps = sorted((staves[k + 1].y0 - staves[k].y1) / staves[k].sp for k in range(len(staves) - 1))
        gm = gaps[len(gaps) // 2]
    if w is not None and len(staves) > 1:
        ps = [links.prob(w, links.pair_features(syms, staves[k - 1], staves[k], gm, len(staves),
                                                _braced(syms, staves[k - 1], staves[k])))
              for k in range(1, len(staves))]
        decided = links.decode(ps)
    if gray is not None and len(staves) > 1 and w is None:
        decided = [_braced(syms, staves[k - 1], staves[k]) or _aligned(syms, staves[k - 1], staves[k])
                   for k in range(1, len(staves))]
    if gray is not None and len(staves) > 1:
        # the ink decides where it is clear: a line through the gap joins, a white gap separates; a brace or
        # bracket the detector saw over both staves joins them whatever the ink
        page_left = min(st.x0 for st in staves)
        for k in range(1, len(staves)):
            left, bars = _ink_links(gray, syms, staves[k - 1], staves[k], page_left)
            if left >= INK_LINK or bars >= 0.5:
                decided[k - 1] = True
            elif left < INK_SPLIT and bars < 0.1:
                decided[k - 1] = False            # white between them: a brace box claiming otherwise is wrong
            elif _braced(syms, staves[k - 1], staves[k]):
                decided[k - 1] = True
    systems: list[System] = []
    for k, st in enumerate(staves):
        if k == 0:
            linked = False
        elif w is not None or gray is not None:
            linked = decided[k - 1]
        else:
            linked = _braced(syms, staves[k - 1], st) or _aligned(syms, staves[k - 1], st)
        if linked:
            systems[-1].staves.append(k)
        else:
            systems.append(System(staves=[k]))
    for n, sy in enumerate(systems):
        for pos, k in enumerate(sy.staves):
            staves[k].system, staves[k].pos = n, pos
        ref = max(sy.staves, key=lambda k: (sum(not syms[i].synthetic for i in staves[k].bars),
                                            -sum(syms[i].synthetic for i in staves[k].bars)))
        if ink:
            _segment(syms, staves, sy, gray)
        elif gray is not None and len(sy.staves) > 1:
            _joint_bars(syms, staves, sy, gray)
        else:
            _reconcile(syms, staves, sy, ref)
        # bar columns: every bar maps to the reference staff's bar it overlaps most in x
        rb = [syms[i].box for i in staves[ref].bars]
        cols = []
        for k in sy.staves:
            row = []
            for i in staves[k].bars:
                b = syms[i].box
                ov = [min(b[2], r[2]) - max(b[0], r[0]) for r in rb]
                row.append(max(range(len(rb)), key=lambda j: ov[j]))
            cols.append(row)
        sy.columns = cols
    return systems


def _joint_bars(syms: list[Sym], staves: list[Staff], sy: System, gray) -> None:
    """A system's bar lines, decided once for all its staves: a candidate x (a measure box's edge or a detected
    bar line, on any staff) is a bar line where most staves show one there -- a detection, or a vertical ink
    stroke over the staff's whole height. Every staff then gets exactly those bars; a detected measure box that
    matches one keeps its identity, the others are filled in."""
    S = [staves[k] for k in sy.staves]
    sp = sum(st.sp for st in S) / len(S)
    left = min(syms[st.bars[0]].box[0] for st in S)
    right = max(syms[st.bars[-1]].box[2] for st in S)
    cand = []                                     # (x, staff index in S)
    for j, st in enumerate(S):
        for i in st.bars[:-1]:
            if not syms[i].synthetic:
                cand.append((syms[i].box[2], j))
        for b in syms:
            if b.fam == "barLine" and b.p >= 0.4 and st.y0 - sp <= b.cy <= st.y1 + sp:
                cand.append((b.cx, j))
    cand = sorted((x, j) for x, j in cand if left + 3 * sp < x < right - 1.5 * sp)
    clusters: list[list] = []
    for x, j in cand:
        if clusters and x - clusters[-1][-1][0] <= sp:
            clusters[-1].append((x, j))
        else:
            clusters.append([(x, j)])
    cuts = []
    for cl in clusters:
        x = sorted(v for v, _ in cl)[len(cl) // 2]
        seen = {j for _, j in cl}
        ev = 0
        for j, st in enumerate(S):
            if j in seen or _ink_run(gray, x, st.y0 + 1, st.y1 - 1, reach=1) >= 0.9:
                ev += 1
        if ev >= max(2, 0.75 * len(S)):
            cuts.append(x)
    xs = [left, *cuts, right]
    for st in S:
        old = [syms[i] for i in st.bars]
        new = []
        for a, b in zip(xs, xs[1:]):
            match = [o for o in old if not o.synthetic and abs(o.box[0] - a) < 1.5 * sp and abs(o.box[2] - b) < 1.5 * sp]
            if match:
                o = match[0]
                o.box = (a, o.box[1], b, o.box[3])
                new.append(o.i)
            else:
                band = [o for o in old if o.box[0] < b and o.box[2] > a] or old
                yy0 = sorted(o.box[1] for o in band)[len(band) // 2]
                yy1 = sorted(o.box[3] for o in band)[len(band) // 2]
                t = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                        box=(a, yy0, b, yy1), synthetic=True)
                syms.append(t)
                new.append(t.i)
        st.bars = new
        st.x0, st.x1 = left, right


def _reconcile(syms: list[Sym], staves: list[Staff], sy: System, ref: int) -> None:
    """In a system, a staff's filled-in bars take the bar lines of the staff that has the most detected bars:
    its bars at that x are better evidence than lone bar-line boxes (stems read as bar lines split a gap)."""
    rb = [syms[i].box for i in staves[ref].bars]
    for k in sy.staves:
        st = staves[k]
        if k == ref or not any(syms[i].synthetic for i in st.bars):
            continue
        new = []
        for i in st.bars:
            s = syms[i]
            if not s.synthetic:
                new.append(i)
                continue
            x0, y0, x1, y1 = s.box
            cuts = [x0] + [r[2] for r in rb if x0 + st.sp < r[2] < x1 - st.sp] + [x1]
            # the stretch keeps its ends; inside it, the reference's bar lines
            first = True
            for a, b in zip(cuts, cuts[1:]):
                if first:
                    s.box = (a, y0, b, y1)
                    new.append(i)
                    first = False
                else:
                    t = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                            box=(a, y0, b, y1), synthetic=True)
                    syms.append(t)
                    new.append(t.i)
        # filled bars that start on a reference bar's start take its left edge (a missed first bar)
        st.bars = sorted(new, key=lambda i: syms[i].cx)
    # the merged-away splits: synthetic bars inside one reference bar fold together
    for k in sy.staves:
        st = staves[k]
        if k == ref:
            continue
        keep = []
        for i in st.bars:
            s = syms[i]
            if keep and (s.synthetic or syms[keep[-1]].synthetic):
                prev = syms[keep[-1]]
                mid_prev, mid = (prev.box[0] + prev.box[2]) / 2, (s.box[0] + s.box[2]) / 2
                same = [j for j, r in enumerate(rb) if r[0] <= mid_prev <= r[2] and r[0] <= mid <= r[2]]
                if same:                       # one reference bar: one bar (the detected box, widened)
                    a, b = (prev, s) if not prev.synthetic or s.synthetic else (s, prev)
                    a.box = (min(prev.box[0], s.box[0]), a.box[1], max(prev.box[2], s.box[2]), a.box[3])
                    keep[-1] = a.i
                    continue
            keep.append(i)
        st.bars = keep


# ------------------------------------------------------------------------------------------------- ink path
def _ink_staves(gray, syms: list[Sym]) -> list[Staff] | None:
    """Staves from the staff lines in the ink (vision.page.staves: five long thin runs at one spacing), each
    holding the detected measure boxes that sit on it. A row of measure boxes no ink staff explains (a staff the
    line finder missed) is a staff of its own, as on the box path."""
    from PIL import Image

    from .vision.page.staves import find_staves
    m = find_staves(Image.fromarray(gray))
    if not m.staves or m.spacing <= 0:
        return None
    # the measure boxes' own staff space checks the line finder (a skewed or curved page gives it false staves)
    hs = sorted(s.box[3] - s.box[1] for s in syms if s.fam == "measure" and s.p >= 0.5)
    box_sp = hs[len(hs) // 2] / 4 if hs else None
    found = [st for st in m.staves if box_sp is None or 0.6 * box_sp <= st.spacing <= 1.6 * box_sp]
    if not found:
        return None
    if hs and len(found) < 0.8 * len(_rows([s for s in syms if s.fam == "measure" and s.p >= MEASURE_MIN_P])):
        return None                               # the lines were found on too few staves: the box path
    staves = [Staff(bars=[], sp=max(1.0, st.spacing), y0=st.top, y1=st.bottom, x0=float(st.x0), x1=float(st.x1))
              for st in found]
    claimed = set()
    for s in syms:
        if s.fam != "measure" or s.p < MEASURE_LOW_P:
            continue
        h = s.box[3] - s.box[1]
        ov = [min(s.box[3], st.y1) - max(s.box[1], st.y0) for st in staves]
        k = max(range(len(staves)), key=lambda j: ov[j])
        if ov[k] >= 0.5 * min(h, staves[k].y1 - staves[k].y0):
            staves[k].bars.append(s.i)
            claimed.add(s.i)
    rest = [s for s in syms if s.fam == "measure" and s.p >= 0.5 and s.i not in claimed and
            s.box[2] - s.box[0] >= MEASURE_MIN_W * (s.box[3] - s.box[1]) / 4]
    for st in _rows(rest):
        if not any(abs(st.mid - o.mid) < 6 * o.sp for o in staves):
            staves.append(st)
    for st in staves:
        st.bars.sort(key=lambda i: syms[i].cx)
    staves.sort(key=lambda st: st.mid)
    return staves


def _segment(syms: list[Sym], staves: list[Staff], sy: System, gray) -> None:
    """A system's bars from the evidence of all its staves at once: a candidate x (a measure box's edge or a
    detected bar line) is a bar line where a vertical stroke crosses the staff's full height -- on most staves of a
    multi-staff system, with a detection behind it; on a single staff, a detection and the stroke (or a firm
    bar-line detection alone). Every staff gets exactly those bars, from the staff lines' left end to their right
    end; a detected measure box that matches a bar keeps its identity, the others are filled in."""
    S = [staves[k] for k in sy.staves]
    n = len(S)
    sp = sum(st.sp for st in S) / n
    left, right = min(st.x0 for st in S), max(st.x1 for st in S)
    cand = []                                       # (x, staff index, kind, p)
    for j, st in enumerate(S):
        for i in st.bars:
            b = syms[i]
            if b.p >= 0.6 and not b.synthetic:
                cand += [(b.box[0], j, "edge", b.p), (b.box[2], j, "edge", b.p)]
        for b in syms:
            if b.fam == "barLine" and b.p >= 0.3 and st.y0 - sp <= b.cy <= st.y1 + sp:
                cand.append((b.cx, j, "line", b.p))
    if n >= 2:
        # a column of ink over the full height of every staff at once is a bar line, detected or not
        import numpy as np
        xa, xb = int(left + 2 * sp), int(right - 1.5 * sp)
        if xb > xa:
            rows = [gray[int(st.y0) + 1:int(st.y1), xa:xb].mean(axis=0) for st in S]
            if n == 2:                  # two hands' stems line up; a grand staff's bar line also crosses the gap
                rows.append(gray[int(S[0].y1) + 1:int(S[1].y0), xa:xb].mean(axis=0))
            cover = np.min(rows, axis=0)
            run = None
            for x, v in enumerate(cover >= 0.97):
                if v and run is None:
                    run = x
                if (not v or x == len(cover) - 1) and run is not None:
                    cand.append((xa + (run + x - 1) / 2, -1, "ink", 1.0))
                    run = None
    cand = sorted(c for c in cand if left + 2 * sp < c[0] < right - 1.5 * sp)
    clusters: list[list] = []
    for c in cand:
        if clusters and c[0] - clusters[-1][-1][0] <= sp:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    cuts = []
    for cl in clusters:
        lines = [c for c in cl if c[2] == "line"]
        x = sorted(c[0] for c in (lines or cl))[len(lines or cl) // 2]
        ink = [_ink_run(gray, x, st.y0 + 1, st.y1 - 1, reach=2) >= 0.9 for st in S]
        with_line = {c[1] for c in lines}
        firm = max((c[3] for c in lines), default=0.0)
        if n >= 2:
            # stems of notes struck together line up across the staves too: a bar line needs detections on at
            # least half the staves and the stroke on most -- or the stroke through every staff's full height
            ok = (len(with_line) >= max(1, n / 2) and sum(ink) >= 0.75 * n) or len(with_line) >= 0.75 * n or \
                any(c[2] == "ink" for c in cl)
        else:
            edges = [c for c in cl if c[2] == "edge"]
            ok = (ink[0] and (firm >= 0.5 or len(edges) >= 2)) or firm >= 0.85
        if ok:
            cuts.append(x)
    xs = [left, *cuts, right]
    # no bar is narrower than MEASURE_MIN_W staff spaces: a sliver (a double bar line's second stroke, the staff
    # lines running on past the last bar line) joins its neighbour
    while len(xs) > 2:
        w = [b - a for a, b in zip(xs, xs[1:])]
        k = min(range(len(w)), key=lambda j: w[j])
        if w[k] >= MEASURE_MIN_W * sp:
            break
        del xs[k + 1 if k == 0 else k]
    # a first or last stretch with no note or rest in it is no bar: the staff lines' run into a margin, or the
    # cautionary key / clef after a system's last bar line
    music = [m for m in syms if m.fam in ("note", "rest", "mRest", "multiRest", "mRpt", "multiRpt") and m.p >= 0.4
             and S[0].y0 - 4 * sp <= m.cy <= S[-1].y1 + 4 * sp]

    def empty(a, b):
        return not any(a <= m.cx <= b for m in music)

    while len(xs) > 2 and empty(xs[-2], xs[-1]):
        del xs[-2]
    while len(xs) > 2 and empty(xs[0], xs[1]):
        del xs[1]
    for st in S:
        old = [syms[i] for i in st.bars]
        new = []
        for a, b in zip(xs, xs[1:]):
            match = [o for o in old if not o.synthetic and abs(o.box[0] - a) < 1.5 * sp and abs(o.box[2] - b) < 1.5 * sp]
            if match:
                o = max(match, key=lambda o: o.p)
                o.box = (a, o.box[1], b, o.box[3])
                new.append(o.i)
            else:
                t = Sym(i=len(syms), cls="measure", post={CLS_ID["measure"]: 1.0}, p=0.0, n=0,
                        box=(a, st.y0, b, st.y1), synthetic=True)
                syms.append(t)
                new.append(t.i)
        st.bars = new
        st.x0, st.x1 = left, right


# ------------------------------------------------------------------------------------------------- assignment
def _band(syms: list[Sym], st: Staff, x: float) -> tuple[int, float, float, float]:
    """(bar index, top, bottom, staff space) of staff ``st`` at x."""
    best, bd = 0, math.inf
    for j, i in enumerate(st.bars):
        b = syms[i].box
        d = 0.0 if b[0] <= x <= b[2] else min(abs(x - b[0]), abs(x - b[2]))
        if d < bd:
            best, bd = j, d
    b = syms[st.bars[best]].box
    if syms[st.bars[best]].synthetic:
        return best, st.y0, st.y1, st.sp
    return best, b[1], b[3], max(1.0, (b[3] - b[1]) / 4)


def _assign(syms: list[Sym], staves: list[Staff], only: list[Sym] | None = None) -> None:
    left = min((st.x0 for st in staves), default=0.0)
    right = max((st.x1 for st in staves), default=0.0)
    for k, st in enumerate(staves):
        for j, i in enumerate(st.bars):
            syms[i].staff, syms[i].bar = k, j
    for s in (syms if only is None else only):
        if s.staff >= 0 or s.fam == "measure":
            continue
        x = s.box[2] if s.cls == "barLine_repeat_start" else s.cx      # it opens the bar on its right
        best, bc, bbar = -1, math.inf, -1
        pos = None
        if s.fam in ("note", "rest") and s.attrs.get("staff_position") not in (None, "<na>"):
            try:
                pos = int(s.attrs["staff_position"])
            except ValueError:
                pos = None
            # a saturated (the head's last value) or unsure reading says nothing about the staff
            if pos is not None and (abs(pos) >= 20 or float(s.attr_p.get("staff_position", 1.0)) < 0.5):
                pos = None
        for k, st in enumerate(staves):
            # any staff across the page's width: a staff whose first bars the detector missed starts late
            if not (left - 4 * st.sp <= x <= right + 4 * st.sp):
                continue
            j, y0, y1, sp = _band(syms, st, x)
            d = max(0.0, y0 - s.cy, s.cy - y1) / sp
            if pos is not None and s.fam == "note":
                implied = ((y0 + y1) / 2 - s.cy) / (sp / 2)
                d = min(d + 1.0, abs(implied - pos) / 2) if d > 0 else abs(implied - pos) / 2
            if d < bc:
                best, bc, bbar = k, d, j
        if best >= 0 and bc <= ASSIGN_MAX_SP:
            s.staff, s.bar = best, bbar


CAND_SKIP = ("measure", "barLine")         # structure is decided before candidates join


def build(dets: list[dict], width: float, height: float, image=None, cands: list[dict] | None = None) -> Layout:
    """``image``: the page (PIL image or grey array); with it, the ink between staves decides the systems.
    ``cands``: the detector's sub-threshold candidates; once the structure stands, the ones no symbol of their
    family already covers join as symbols flagged ``cand`` (the model decides whether they are real)."""
    import numpy as np
    gray = None
    if image is not None:
        gray = np.asarray(image.convert("L") if hasattr(image, "convert") else image)
    syms = merge(dets)
    staves = _ink_staves(gray, syms) if gray is not None else None
    mask = ink_mask(gray) if gray is not None else None
    if staves:
        systems = _systems(syms, staves, mask, ink=True)
    else:
        staves = _staves(syms)
        _fill_gaps(syms, staves)
        _snap_bars(syms, staves)
        systems = _systems(syms, staves, mask)
    _assign(syms, staves)
    if cands:
        new = []
        for c in merge(cands, offset=len(dets)):
            if c.fam in CAND_SKIP or any(o.fam == c.fam and iou(o.box, c.box) >= 0.3 for o in syms):
                continue
            c.i, c.cand = len(syms), True
            syms.append(c)
            new.append(c)
        _assign(syms, staves, only=new)
    sps = sorted(st.sp for st in staves)
    sp = sps[len(sps) // 2] if sps else 10.0
    return Layout(syms=syms, staves=staves, systems=systems, sp=sp, width=width, height=height)
