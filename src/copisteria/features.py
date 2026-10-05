"""A page's symbols as arrays: what the model reads about each token.

Raw components only (class posterior, P(real), attribute distributions, geometry in staff spaces, structure
indices); the model builds its embeddings and its priors from them, so training-time corruption can edit the
raw readings and everything downstream stays consistent.
"""
from __future__ import annotations

import math

import numpy as np

from .front import Layout
from .vocab import DOTS_IN, FAM_ID, GRACE_IN, N_VOICE, STEM, VOICE_IN, family

K_CLS = 3                     # classes kept per token's posterior
GEO = ["dx_bar", "rel_bar", "y_pos", "w", "h", "bar_w", "x_sys", "y_sys", "first_bar", "last_bar", "synthetic",
       "n_merged", "bar_tokens"]
N_GEO = len(GEO)
MAX_STAVES = 16
POS_SATURATED = 22            # detector staff positions this far out are treated as no reading


def _dist(dist: dict, vocab: list[str], fold: dict | None = None, size: int | None = None) -> list[float]:
    """attr_dist over ``vocab`` -> a probability vector of ``size`` (fold maps vocab entries to slots)."""
    size = size or len(vocab)
    out = [0.0] * size
    if not dist:
        return out
    for k, v in dist.items():
        slot = fold.get(k) if fold is not None else (vocab.index(k) if k in vocab else None)
        if slot is not None:
            out[slot] += float(v)
    s = sum(out)
    return [x / s for x in out] if s > 0 else out


DOTS_FOLD = {"<na>": 0, "1": 1, "2": 2, "3": 3, "4": 3}
VOICE_FOLD = {str(v): min(N_VOICE, v) - 1 for v in range(1, 9)}
GRACE_FOLD = {"<na>": 0, "acc": 1, "unacc": 2}
STEM_FOLD = {"down": 0, "up": 1, "both": 2}


def page_arrays(L: Layout, feats: np.ndarray | None = None) -> dict:
    """Arrays over the page's assigned tokens (symbols with a staff), in reading order. ``feats``: the detector's
    image features per detection (Detector.run_array with candidates), indexed by each symbol's ``det``."""
    syms = [s for s in L.syms if s.staff >= 0]
    st_of = L.staves

    # global bar column across the page: systems in order, columns within a system
    sys_off, off = [], 0
    for sy in L.systems:
        sys_off.append(off)
        off += 1 + max((max(c) for c in sy.columns if c), default=0)

    def order(s):
        st = st_of[s.staff]
        return (st.system, st.pos, s.bar, s.fam != "measure", s.cx, s.cy)

    syms.sort(key=order)
    n = len(syms)
    a = {
        "sym": np.array([s.i for s in syms], np.int32),
        "fam": np.zeros(n, np.int16), "cls_ids": np.zeros((n, K_CLS), np.int16),
        "cls_p": np.zeros((n, K_CLS), np.float32), "p": np.zeros(n, np.float32),
        "geo": np.zeros((n, N_GEO), np.float32),
        "staffpos": np.zeros(n, np.int8), "nstaves": np.zeros(n, np.int8),
        "sys": np.zeros(n, np.int16), "col": np.zeros(n, np.int32), "staff": np.zeros(n, np.int16),
        "pos_v": np.full(n, -99, np.int8), "pos_p": np.zeros(n, np.float32),
        "dots_p": np.zeros((n, 4), np.float32), "voice_p": np.zeros((n, N_VOICE), np.float32),
        "grace_p": np.zeros((n, 3), np.float32), "stem_p": np.zeros((n, 3), np.float32),
        "cand": np.array([s.cand for s in syms], np.int8),
    }
    if feats is not None:
        a["feat"] = np.zeros((n, feats.shape[1]), np.float16)
        for t, s in enumerate(syms):
            if 0 <= s.det < len(feats):
                a["feat"][t] = feats[s.det]
    bar_count: dict[tuple, int] = {}
    for s in syms:
        bar_count[(s.staff, s.bar)] = bar_count.get((s.staff, s.bar), 0) + 1
    for t, s in enumerate(syms):
        st = st_of[s.staff]
        sy = L.systems[st.system]
        bar = L.syms[st.bars[s.bar]]
        bx0, by0, bx1, by1 = bar.box
        sp = st.sp if bar.synthetic else max(1.0, (by1 - by0) / 4)
        mid = (by0 + by1) / 2 if not bar.synthetic else st.mid
        sx0 = min(st_of[k].x0 for k in sy.staves)
        sx1 = max(st_of[k].x1 for k in sy.staves)
        sy0 = min(st_of[k].y0 for k in sy.staves)
        sy1 = max(st_of[k].y1 for k in sy.staves)
        a["fam"][t] = FAM_ID[family(s.cls)]
        top = sorted(s.post.items(), key=lambda kv: -kv[1])[:K_CLS]
        for k, (c, pr) in enumerate(top):
            a["cls_ids"][t, k] = c
            a["cls_p"][t, k] = pr
        a["p"][t] = s.p if not s.synthetic else 0.5
        x0, y0, x1, y1 = s.box
        a["geo"][t] = [
            (s.cx - bx0) / sp / 10, (s.cx - bx0) / max(1.0, bx1 - bx0), (mid - s.cy) / (sp / 2) / 10,
            (x1 - x0) / sp / 10, (y1 - y0) / sp / 10, (bx1 - bx0) / sp / 50,
            (s.cx - sx0) / max(1.0, sx1 - sx0), (s.cy - sy0) / max(1.0, sy1 - sy0) if sy1 > sy0 else 0.0,
            float(s.bar == 0), float(s.bar == len(st.bars) - 1), float(s.synthetic),
            min(s.n, 4) / 4, min(bar_count[(s.staff, s.bar)], 64) / 64,
        ]
        a["staffpos"][t] = min(st.pos, MAX_STAVES - 1)
        a["nstaves"][t] = min(len(sy.staves), MAX_STAVES)
        a["sys"][t] = st.system
        a["staff"][t] = s.staff
        a["col"][t] = sys_off[st.system] + sy.columns[st.pos][s.bar]
        if s.fam in ("note", "rest"):
            pv = s.attrs.get("staff_position")
            if pv not in (None, "<na>"):
                try:
                    v = int(pv)
                    if abs(v) < POS_SATURATED:     # the head's end values are where it gives up, not a reading
                        a["pos_v"][t] = v
                        a["pos_p"][t] = float(s.attr_p.get("staff_position", 1.0))
                except ValueError:
                    pass
            ad = s.attr_dist or {}
            a["dots_p"][t] = _dist(ad.get("dots") or {(s.attrs.get("dots") or "<na>"): 1.0}, DOTS_IN, DOTS_FOLD, 4)
            a["voice_p"][t] = _dist(ad.get("voice_slot") or {(s.attrs.get("voice_slot") or "1"): 1.0}, VOICE_IN,
                                    VOICE_FOLD, N_VOICE)
            a["grace_p"][t] = _dist(ad.get("grace") or {(s.attrs.get("grace") or "<na>"): 1.0}, GRACE_IN,
                                    GRACE_FOLD, 3)
            a["stem_p"][t] = _dist(ad.get("stem_dir") or {(s.attrs.get("stem_dir") or "<na>"): 1.0}, STEM,
                                   STEM_FOLD, 3)
    return a


def target_arrays(a: dict, T: dict) -> dict:
    """match.targets (per symbol index) -> arrays aligned with page_arrays' tokens."""
    idx = a["sym"]
    out = {}
    for k in ("real", "cls", "dots", "pos", "alter", "voice", "chord", "tie", "tup", "grace", "key", "time", "clef"):
        out["t_" + k] = np.array([T[k][i] for i in idx], np.int16)
    out["t_onset"] = np.array([math.nan if T["onset"][i] is None else T["onset"][i] for i in idx], np.float32)
    return out
