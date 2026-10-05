"""Which consecutive staves share a system: a logistic model over the evidence a pair of staves gives (a group
symbol spanning them, how their bar lines line up, the gap between them), fitted on rendered pages where the
labels say which system every bar belongs to. The front end uses it when ``links.json`` exists, else its rule.

  python -m copisteria.links --pages <rendered pages> --out src/copisteria/links.json [--limit 3000]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FEATURES = ["braced", "edges_a", "edges_b", "hit_ab", "hit_ba", "spread", "offset", "gap", "gap_rel", "dx0", "dx1",
            "same_bars", "few_bars", "n_staves", "aligned", "tight", "hit_min", "bias"]
PATH = Path(__file__).with_name("links.json")


def pair_features(syms, a, b, gaps_median: float, n_staves: int, braced: bool) -> list[float]:
    sp = (a.sp + b.sp) / 2
    lo, hi = max(a.x0, b.x0) - sp, min(a.x1, b.x1) + sp

    def edges(st):
        return [syms[k].box[2] for k in st.bars[:-1] if not syms[k].synthetic and lo <= syms[k].box[2] <= hi]

    ea, eb = edges(a), edges(b)
    tol = 2.0 * sp
    hit_ab = hit_ba = 0.0
    spread, offset = 3.0, 3.0
    if ea and eb:
        diffs = [min(eb, key=lambda y: abs(x - y)) - x for x in ea]
        good = [d for d in diffs if abs(d) <= tol]
        hit_ab = len(good) / len(ea)
        hit_ba = sum(1 for y in eb if any(abs(x - y) <= tol for x in ea)) / len(eb)
        if good:
            med = sorted(good)[len(good) // 2]
            offset = min(3.0, abs(med) / sp)
            spread = min(3.0, max(abs(d - med) for d in good) / sp) if len(good) > 1 else 1.5
    gap = (b.y0 - a.y1) / sp
    from .front import _aligned
    return [float(braced), math.log1p(len(ea)), math.log1p(len(eb)), hit_ab, hit_ba, spread, offset,
            min(gap, 30) / 10, min(gap / max(gaps_median, 1e-3), 5), min(abs(a.x0 - b.x0) / sp, 10) / 5,
            min(abs(a.x1 - b.x1) / sp, 10) / 5, float(len(a.bars) == len(b.bars)),
            float(len(a.bars) <= 2 or len(b.bars) <= 2), math.log(n_staves), float(_aligned(syms, a, b)),
            float(spread <= 0.6), min(hit_ab, hit_ba), 1.0]


def load() -> np.ndarray | None:
    if PATH.exists():
        d = json.loads(PATH.read_text())
        if d.get("features") == FEATURES:
            return np.array(d["w"])
    return None


def prob(w: np.ndarray, f: list[float]) -> float:
    return 1 / (1 + math.exp(-float(np.dot(w, f))))


UNIFORM_MARGIN = 3.0       # nats a uniform staves-per-system reading may cost over the pair-by-pair one


def decode(ps: list[float]) -> list[bool]:
    """Links between consecutive staves from their probabilities: pair by pair, unless splitting the page into
    systems of one size is nearly as likely (pages keep their staves per system)."""
    free = [p >= 0.5 for p in ps]
    n = len(ps) + 1
    lp = lambda links: sum(math.log(max(1e-9, p if l else 1 - p)) for p, l in zip(ps, links))  # noqa: E731
    best, bl = free, lp(free)
    for K in range(1, n + 1):
        if n % K:
            continue
        links = [(k + 1) % K != 0 for k in range(n - 1)]
        v = lp(links)
        if v >= bl - UNIFORM_MARGIN and (best is free or v > lp(best)):
            best = links if v >= lp(free) - UNIFORM_MARGIN else best
    return best


def _gt_systems(L, lab) -> list[int | None]:
    """GT system id per front-end staff: the majority over its detected bars matched to labelled bars."""
    from .front import iou
    gms = [e for e in lab["elements"] if e["type"] == "measure" and e.get("src")]
    boxes = [(e["bbox"][0], e["bbox"][1], e["bbox"][0] + e["bbox"][2], e["bbox"][1] + e["bbox"][3]) for e in gms]
    out = []
    for st in L.staves:
        votes: dict = {}
        for i in st.bars:
            s = L.syms[i]
            if s.synthetic:
                continue
            best, bv = None, 0.3
            for e, bb in zip(gms, boxes):
                v = iou(s.box, bb)
                if v > bv:
                    best, bv = e, v
            if best is not None:
                key = best["src"].get("system")
                votes[key] = votes.get(key, 0) + 1
        out.append(max(votes, key=votes.get) if votes else None)
    return out


RULE: list[float] = []
PAGE: list[int] = []


def collect(pages: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    from . import front
    X, Y = [], []
    n = 0
    for idx in sorted(pages.glob("t*/index.jsonl")):
        for line in idx.read_text().splitlines():
            r = json.loads(line)
            dp = Path(r["image"]).with_suffix(".dets.json")
            if not dp.exists():
                continue
            lab = json.loads(Path(r["labels"]).read_text())
            syms = front.merge(json.loads(dp.read_text()))
            staves = front._staves(syms)
            front._fill_gaps(syms, staves)
            front._snap_bars(syms, staves)
            if len(staves) < 2:
                continue
            L = front.Layout(syms=syms, staves=staves, systems=[], sp=1, width=1, height=1)
            gt = _gt_systems(L, lab)
            gaps = sorted((staves[k + 1].y0 - staves[k].y1) / staves[k].sp for k in range(len(staves) - 1))
            gm = gaps[len(gaps) // 2]
            for k in range(len(staves) - 1):
                if gt[k] is None or gt[k + 1] is None:
                    continue
                a, b = staves[k], staves[k + 1]
                X.append(pair_features(syms, a, b, gm, len(staves), front._braced(syms, a, b)))
                Y.append(float(gt[k] == gt[k + 1]))
                RULE.append(float(front._braced(syms, a, b) or front._aligned(syms, a, b)))
                PAGE.append(n)
            n += 1
            if n >= limit:
                return np.array(X), np.array(Y)
    return np.array(X), np.array(Y)


def fit(X: np.ndarray, Y: np.ndarray, l2: float = 1e-3, iters: int = 3000) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):                                      # Newton steps on the regularised log-loss
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - Y) / len(Y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X / len(Y) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", default=str(PATH))
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    X, Y = collect(Path(a.pages), a.limit)
    n = len(Y)
    k = int(n * 0.8)
    w = fit(X[:k], Y[:k])
    p = 1 / (1 + np.exp(-X[k:] @ w))
    acc = float(((p > 0.5) == (Y[k:] > 0.5)).mean())
    # the hand rule on the same held-out pairs, for comparison
    rule = np.array(RULE[k:n])
    print(f"pairs {n} (same-system {Y.mean():.3f}); held-out accuracy {acc:.4f}, the hand rule's "
          f"{float((rule > 0.5) == (Y[k:] > 0.5)).mean() if False else float(((rule > 0.5) == (Y[k:] > 0.5)).mean()):.4f}")
    pages = np.array(PAGE[:n])
    hold = pages >= pages[k] if k < n else np.zeros(n, bool)
    pr = 1 / (1 + np.exp(-X @ fit(X[~hold], Y[~hold])))
    ok = {"rule": 0, "free": 0, "uniform": 0}
    pairs_ok = {"rule": 0, "free": 0, "uniform": 0}
    ids = sorted(set(pages[hold]))
    for pid in ids:
        m = pages == pid
        y = Y[m] > 0.5
        for name, links in (("rule", np.array(RULE[:n])[m] > 0.5), ("free", pr[m] >= 0.5),
                            ("uniform", np.array(decode(list(pr[m]))))):
            ok[name] += int((links == y).all()); pairs_ok[name] += int((links == y).sum())
    tot = int(hold.sum())
    print("held-out pages", len(ids), {k_: f"pages {v / max(1, len(ids)):.3f} pairs {pairs_ok[k_] / max(1, tot):.4f}"
                                       for k_, v in ok.items()})
    w = fit(X, Y)
    Path(a.out).write_text(json.dumps({"features": FEATURES, "w": [round(float(v), 5) for v in w],
                                       "pairs": n, "heldout_acc": acc}, indent=1))
    print(dict(zip(FEATURES, np.round(w, 3))))


if __name__ == "__main__":
    main()
