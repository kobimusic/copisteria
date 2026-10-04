"""Staff lines from the ink, without a model.

A staff line is the one thing on a page that is a long, thin, horizontal run of ink. Every row of the
page is scored by how much of it is covered by ink runs at least ``min_run`` wide; rows that score are
lines, and five lines at one spacing are a staff. The row profile is kept so the viewer can show why a
row was, or was not, taken as a line.

What counts as scoring is relative to the page: a clean 300 ppi print covers 0.8 of a row with one line, a
dithered one-bit scan whose lines are broken and a little skewed covers 0.3, so the cutoff is 0.7 of the
page's 95th-percentile row coverage (never under ``floor_cover``); a row that still scores without being a
line (a beam, a slur's tail) is left to the staff grouping, which passes over it. Breaks under ``close_px``
are bridged before runs are measured, and a line the cutoff cut in two is joined again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Line:
    y: float            # centre row
    x0: int             # leftmost ink of its long runs
    x1: int             # rightmost
    thickness: int      # rows
    score: float        # peak row coverage, 0..1


@dataclass
class Staff:
    lines: list[Line]   # five, top to bottom

    @property
    def spacing(self) -> float:
        return (self.bottom - self.top) / (len(self.lines) - 1)

    @property
    def top(self) -> float:
        return self.lines[0].y

    @property
    def bottom(self) -> float:
        return self.lines[-1].y

    @property
    def x0(self) -> int:
        return min(line.x0 for line in self.lines)

    @property
    def x1(self) -> int:
        return max(line.x1 for line in self.lines)


@dataclass
class StaffMap:
    staves: list[Staff]
    lines: list[Line]           # every line found, staffed or not
    profile: np.ndarray         # per-row long-run coverage, 0..1
    ink_threshold: int
    cover_threshold: float

    @property
    def spacing(self) -> float:
        """The page's staff space: the median staff's, 0.0 on a page without staves."""
        sps = sorted(s.spacing for s in self.staves)
        return sps[len(sps) // 2] if sps else 0.0


def find_staves(image: Image.Image, min_run_frac: float = 0.05, rel_cover: float = 0.7, floor_cover: float = 0.05,
                close_px: int = 2) -> StaffMap:
    gray = np.asarray(image.convert("L"))
    thr = otsu(gray)
    ink = close_gaps((gray < thr).astype(np.int8), close_px)
    min_run = max(8, int(min_run_frac * gray.shape[1]))
    coverage, x0, x1 = long_run_profile(ink, min_run)
    min_cover = max(floor_cover, rel_cover * float(np.percentile(coverage, 95)))
    lines = join_split_lines(find_lines(coverage, x0, x1, min_cover))
    return StaffMap(staves=group_staves(lines), lines=lines, profile=coverage, ink_threshold=thr,
                    cover_threshold=min_cover)


def otsu(gray: np.ndarray) -> int:
    # NOTE: not quite Otsu. The upper mean at split t divides the level sum above t by the pixel count above
    # t + 1, so on a page with pure white the split at 254 divides 255 x (white pixels) by max(0, 1), swamps
    # every other split, and the threshold is 254: ink is anything short of pure white, on every page seen
    # so far. The staff finder was tuned that way; a true Otsu moves every page's threshold (and with it the
    # size each page is detected at), so it is a change of its own.
    levels = np.arange(256)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    m0 = np.cumsum(hist * levels) / np.maximum(w0, 1)
    m1 = np.cumsum((hist * levels)[::-1])[::-1] / np.maximum(w1, 1)
    between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[1:]) ** 2
    return int(np.argmax(between))


def close_gaps(ink: np.ndarray, k: int) -> np.ndarray:
    """Ink with horizontal gaps shorter than ``k`` filled: a pixel becomes ink when ink lies within k on
    both sides (a one-dimensional closing along the row)."""
    if k <= 0:
        return ink
    a = ink.astype(bool)
    left, right = a.copy(), a.copy()
    for shift in range(1, k + 1):
        left[:, shift:] |= a[:, :-shift]
        right[:, :-shift] |= a[:, shift:]
    return (a | (left & right)).astype(np.int8)


def long_run_profile(ink: np.ndarray, min_run: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: the fraction of the width covered by ink runs at least ``min_run`` long, and the x-extent
    of those runs (x0 = width, x1 = 0 on a row without one)."""
    h, w = ink.shape
    edges = np.diff(np.pad(ink.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)           # row-major, so the k-th end closes the k-th start
    long = ends - starts >= min_run
    rows, starts, ends = rows[long], starts[long], ends[long]
    coverage = np.bincount(rows, weights=ends - starts, minlength=h) / w
    x0 = np.full(h, w, dtype=np.int64)
    x1 = np.zeros(h, dtype=np.int64)
    np.minimum.at(x0, rows, starts)
    np.maximum.at(x1, rows, ends)
    return coverage, x0, x1


def find_lines(coverage: np.ndarray, x0: np.ndarray, x1: np.ndarray, min_cover: float) -> list[Line]:
    """Each run of rows covered ``min_cover`` or more is a line, centred on its coverage-weighted row."""
    edges = np.diff(np.concatenate(([0], (coverage >= min_cover).astype(np.int8), [0])))
    lines = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        wts = coverage[s:e]
        lines.append(Line(y=float((np.arange(s, e) * wts).sum() / wts.sum()), x0=int(x0[s:e].min()),
                          x1=int(x1[s:e].max()), thickness=int(e - s), score=float(wts.max())))
    return lines


def join_split_lines(lines: list[Line], frac: float = 0.3) -> list[Line]:
    """Two lines closer than ``frac`` of the typical gap between lines are one line the cutoff cut in two
    (a row of it dipped under); joined at their coverage-weighted centre, over the union of their extents."""
    if len(lines) < 2:
        return lines
    typical = float(np.median(np.diff([line.y for line in lines])))
    out = [lines[0]]
    for line in lines[1:]:
        prev = out[-1]
        if line.y - prev.y >= frac * typical:
            out.append(line)
            continue
        w0, w1 = prev.score * prev.thickness, line.score * line.thickness
        out[-1] = Line(y=(prev.y * w0 + line.y * w1) / (w0 + w1), x0=min(prev.x0, line.x0),
                       x1=max(prev.x1, line.x1), thickness=prev.thickness + line.thickness,
                       score=max(prev.score, line.score))
    return out


def group_staves(lines: list[Line], tol: float = 0.25) -> list[Staff]:
    """Five lines at one spacing, sharing some x-extent, are a staff. Lines need not be consecutive in the
    list: a row between two staff lines that scored (a beam, the tail of a slur) lies half a spacing off
    every target and is simply not chosen. Among the staves ``_trials`` turns up the strongest (summed line
    score) are taken first, and no line serves twice."""
    ys = np.array([line.y for line in lines])
    if len(ys) < 5:
        return []
    found, seen = [], set()
    for top, sp in _trials(ys):
        members = _staff_at(ys, top, sp, tol)
        if members is None or members in seen:
            continue
        seen.add(members)
        run = [lines[m] for m in members]
        if min(line.x1 for line in run) <= max(line.x0 for line in run):
            continue                            # five lines at one spacing that never share a column
        found.append((sum(line.score for line in run), members))
    used, staves = set(), []
    for _, members in sorted(found, key=lambda t: -t[0]):
        if used.isdisjoint(members):
            used.update(members)
            staves.append(Staff(lines=[lines[m] for m in members]))
    return sorted(staves, key=lambda s: s.top)


def _trials(ys: np.ndarray):
    """(top line, spacing) pairs worth trying: each line with each of the next eight taken as its k-th line
    below (k = 1..4). The spacing comes from a line up to four steps away because a bowed scan drifts a line's
    centre a few pixels, and the spacing measured over the whole staff is the one that fits. Only spacings
    between 0.6 and 1.5 of the page's typical gap between consecutive lines are tried (most gaps on a page
    are staff gaps): without that bound one line from each of five staves is a "staff" at the system spacing."""
    typical = float(np.median(np.diff(ys)))
    for i in range(len(ys)):
        for j in range(i + 1, min(i + 9, len(ys))):
            for k in range(1, 5):
                sp = (ys[j] - ys[i]) / k
                if 0.6 * typical <= sp <= 1.5 * typical:
                    yield i, sp


def _staff_at(ys: np.ndarray, top: int, sp: float, tol: float) -> tuple[int, ...] | None:
    """The five lines of the staff whose top line is ``top`` and whose spacing is ``sp``: below it, at each
    of the four steps, the nearest line within ``tol`` spacings of where it should be; None if one is missing."""
    members = [top]
    for k in range(1, 5):
        off = np.abs(ys - (ys[top] + k * sp))
        off[:top + 1] = np.inf
        m = int(np.argmin(off))
        if off[m] > tol * sp:
            return None
        members.append(m)
    return tuple(members) if len(set(members)) == 5 else None
