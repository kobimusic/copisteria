"""Train the evidence model on tokenised pages (copista_evidence.training.tokenize).

A sample is a window of 1-4 consecutive systems of one page. The detector's readings are corrupted on the fly
with generic, unspecific noise -- masked classes and attributes, a class swapped for a sibling, attribute values
nudged, ghost tokens, dropped tokens, a confidence calibration shift -- at rates drawn per window, so the model
sees everything from the clean detector to a badly read page. The targets never change: the model learns which
contexts make which readings likely, and how far to trust the detector against them.

The released model (evidence-2m.pt) in two stages: from scratch without the tuplet emphasis, then a fine-tune with
it (COPISTA_TUP_UNMARK / COPISTA_TUP_WEIGHT: the share of windows whose tuplets keep only their first mark, and the
sampling weight of pages with tuplets):

  COPISTA_TUP_UNMARK=0 COPISTA_TUP_WEIGHT=1 python -m copista_evidence.train --data data/tok --out runs/base \
      --steps 70000 --batch 24
  python -m copista_evidence.train --data data/tok --out runs/evidence --steps 25000 --batch 24 --lr 3e-4 \
      --warmup 200 --init runs/base/last.pt

The refinement model (evidence-refine-3m.pt) is the released model with a 2-layer second pass that reads the first
pass's conclusions (running duration sums per voice, collisions, cross-voice alignment, accidental glyphs at each
note's height, the bar's meter), fine-tuned in three stages with pages holding two voices weighted 3x
(COPISTA_MULTIVOICE_WEIGHT), the feedback growing from 29 features to 39 (zero-initialised: each stage starts out
reading as the one before), the last with 35 % of windows showing no tuplet mark at all (COPISTA_TUP_DROPALL):

  export COPISTA_MULTIVOICE_WEIGHT=3
  python -m copista_evidence.train --data data/tok --out runs/refine1 --steps 20000 --batch 8 --accum 4 --lr 3e-4 \
      --warmup 200 --init runs/evidence/last.pt --refine 2 --fb 29
  python -m copista_evidence.train --data data/tok --out runs/refine2 --steps 16000 --batch 8 --accum 4 --lr 2e-4 \
      --warmup 200 --init runs/refine1/last.pt --refine 2 --fb 39
  COPISTA_TUP_DROPALL=0.35 python -m copista_evidence.train --data data/tok --out runs/refine --steps 12000 \
      --batch 8 --accum 4 --lr 1.5e-4 --warmup 200 --init runs/refine2/last.pt --refine 2 --fb 39
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .match import IGNORE
from .model import EvidenceNet, n_params
from . import features as _features
from .vocab import CLS_FAM, CLS_ID, FAMILIES, N_CLEF, N_CLS, N_FAM, N_KEY, N_TIME, POS_MAX

TOKEN_KEYS = ("fam", "cls_ids", "cls_p", "p", "geo", "staffpos", "nstaves", "sys", "col", "pos_v", "pos_p",
              "dots_p", "voice_p", "grace_p", "stem_p")
TARGET_KEYS = ("t_real", "t_cls", "t_dots", "t_pos", "t_alter", "t_voice", "t_chord", "t_tie", "t_tup", "t_grace",
               "t_key", "t_time", "t_clef", "t_onset")
MAX_TOK = 1024
_env = os.environ.get
USE_FEATS = _env("COPISTA_FEATS", "1") == "1"          # the detector's image features, where the tokens carry them
USE_CANDS = _env("COPISTA_CANDS", "1") == "1"          # the detector's sub-threshold candidates, likewise
TUPLET_UNMARK = float(_env("COPISTA_TUP_UNMARK", "0.5"))   # windows whose tuplets are marked once only
TUPLET_WEIGHT = float(_env("COPISTA_TUP_WEIGHT", "3.0"))   # sampling weight of pages with tuplets
TUPLET_DROPALL = float(_env("COPISTA_TUP_DROPALL", "0"))    # windows with no tuplet mark at all
# (engravers mark a run of tuplets once, often pages earlier: then the page shows none, and the bar's arithmetic,
# the beaming and the spacing are what is left to read them by)
MULTIVOICE_WEIGHT = float(_env("COPISTA_MULTIVOICE_WEIGHT", "1.0"))   # sampling weight of pages with two voices
LOSS_W = {"real": 1.0, "cls": 1.0, "dots": 1.0, "pos": 0.5, "alter": 1.0, "voice": 0.5, "chord": 0.5, "tie": 0.5,
          "tup": 0.5, "grace": 0.3, "key": 1.0, "time": 1.0, "clef": 1.0, "onset_beat": 0.3, "onset_frac": 0.3}

FAM_CLASSES = [np.array([c for c in range(N_CLS) if CLS_FAM[c] == f]) for f in range(N_FAM)]
NOTE_F, REST_F = FAMILIES.index("note"), FAMILIES.index("rest")
UNPRINTED = np.array([FAMILIES.index(f) for f in ("tuplet", "tupletNum", "tupletBracket") if f in FAMILIES])
GEO_SYN = _features.GEO.index("synthetic")
CLS_MEASURE = CLS_ID["measure"]


def is_val(source: str) -> bool:
    return zlib.crc32(source.encode()) % 50 == 0          # 2 % of scores, by score


def load_pages(root, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    """``root``: a token dir, or several separated by commas; a tranche (t<k>) found in an earlier one is skipped
    in the later ones (a re-tokenised tranche replaces its old tokens)."""
    train, val = [], []
    files, seen = [], set()
    for r in str(root).split(","):
        dirs = sorted(d for d in Path(r).glob("t*") if d.is_dir() and d.name not in seen)
        seen |= {d.name for d in dirs}
        files += [f for d in dirs for f in sorted(d.glob("*.npz"))]
    if limit:
        files = files[:limit]
    for f in files:
        try:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            pg = {k: z[k] for k in TOKEN_KEYS + TARGET_KEYS}
            if "cand" in z.files:
                pg["cand"] = z["cand"]
                if USE_FEATS and "feat" in z.files:
                    pg["feat"] = z["feat"]
                if not USE_CANDS:
                    keep = pg["cand"] == 0
                    pg = {k: v[keep] for k, v in pg.items()}
        except Exception:
            continue
        pg["n_sys"] = int(pg["sys"].max()) + 1 if len(pg["sys"]) else 0
        pg["meta"] = meta
        (val if is_val(meta.get("source", "")) else train).append(pg)
    return train, val


# ------------------------------------------------------------------------------------------------ windows
def window(pg: dict, rng: random.Random, s0: int | None = None, w: int | None = None) -> dict:
    n_sys = pg["n_sys"]
    if w is None:
        w = rng.choice([1, 2, 2, 3, 3, 3, 4])
    w = min(w, n_sys)
    if s0 is None:
        s0 = rng.randrange(0, n_sys - w + 1)
    while True:
        sel = np.nonzero((pg["sys"] >= s0) & (pg["sys"] < s0 + w))[0]
        if len(sel) <= MAX_TOK or w == 1:
            break
        w -= 1
    if len(sel) > MAX_TOK:
        a = rng.randrange(0, len(sel) - MAX_TOK + 1)
        sel = sel[a:a + MAX_TOK]
    if TUPLET_DROPALL and rng.random() < TUPLET_DROPALL:
        sel = sel[~np.isin(pg["fam"][sel], UNPRINTED)]
    elif rng.random() < TUPLET_UNMARK:
        # engravers print a tuplet's number once and leave the rest of the run unmarked: keep the first tuplet
        # mark of each staff row in the window, drop the others (the targets keep the tuplets)
        fam = pg["fam"][sel]
        isup = np.isin(fam, UNPRINTED)
        if isup.any():
            first = set()
            keep = np.ones(len(sel), bool)
            for k in np.nonzero(isup)[0]:
                row = (int(pg["sys"][sel[k]]), int(pg["staffpos"][sel[k]]))
                if row in first:
                    keep[k] = False
                first.add(row)
            sel = sel[keep]
    out = {k: pg[k][sel].copy() for k in TOKEN_KEYS + TARGET_KEYS + tuple(k for k in ("feat", "cand") if k in pg)}
    out["sys"] = out["sys"] - s0
    out["col"] = out["col"] - out["col"].min() if len(sel) else out["col"]
    out["cls_mask"] = np.zeros(len(sel), np.int8)
    out["attr_mask"] = np.zeros(len(sel), np.int8)
    return out


def corrupt(x: dict, rng: random.Random, strength: float = 1.0) -> dict:
    """Generic noise on the detector's readings; targets untouched (except ghosts: not real)."""
    n = len(x["fam"])
    if n == 0:
        return x
    r = np.random.default_rng(rng.randrange(1 << 30))
    rate = lambda m: r.uniform(0, m * strength)                       # noqa: E731
    noteish = (x["fam"] == NOTE_F) | (x["fam"] == REST_F)
    # 1. confidence calibration shift (scans are not renders): p -> p^g, plus per-token jitter
    g = math.exp(r.normal(0, 0.4 * strength))
    p = np.clip(x["p"], 1e-3, 1 - 1e-3) ** g
    p = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + r.normal(0, 0.5 * strength, n))))
    x["p"] = p.astype(np.float32)
    # 2. class masked (the model must read it from context)
    m = r.random(n) < rate(0.15)
    x["cls_mask"][m] = 1
    # 3. class swapped for a sibling in its family
    sw = (r.random(n) < rate(0.08)) & ~m
    for t in np.nonzero(sw)[0]:
        sib = FAM_CLASSES[int(x["fam"][t])]
        if len(sib) < 2:
            continue
        wrong = int(r.choice(sib))
        q = r.uniform(0.5, 1.0)
        x["cls_ids"][t] = [wrong, x["cls_ids"][t, 0], 0]
        x["cls_p"][t] = [q, 1 - q, 0.0]
    # 4. attributes masked / nudged on notes and rests
    am = noteish & (r.random(n) < rate(0.10))
    x["attr_mask"][am] = 1
    x["pos_v"][am] = -99; x["pos_p"][am] = 0
    for k in ("dots_p", "voice_p", "grace_p", "stem_p"):
        x[k][am] = 0
    pn = noteish & (r.random(n) < rate(0.05)) & (x["pos_v"] > -99)
    x["pos_v"][pn] = np.clip(x["pos_v"][pn] + r.choice([-2, -1, 1, 2], pn.sum(), p=[.1, .4, .4, .1]),
                             -POS_MAX, POS_MAX)
    dn = noteish & (r.random(n) < rate(0.05)) & (x["dots_p"].sum(1) > 0)
    for t in np.nonzero(dn)[0]:
        cur = int(np.argmax(x["dots_p"][t])); new = 1 if cur == 0 else 0
        q = r.uniform(0.55, 1.0); v = np.zeros(4, np.float32); v[new] = q; v[cur] = 1 - q
        x["dots_p"][t] = v
    vn = noteish & (r.random(n) < rate(0.05)) & (x["voice_p"].sum(1) > 0)
    for t in np.nonzero(vn)[0]:
        new = int(r.integers(0, 4)); v = np.zeros(4, np.float32); v[new] = 1.0
        x["voice_p"][t] = v
    # 4b. bars the detector missed: the front end fills them from the gap (synthetic flag, no confidence)
    isbar = x["fam"] == FAMILIES.index("measure")
    sb = isbar & (r.random(n) < rate(0.3))
    if sb.any():
        x["geo"][sb, GEO_SYN] = 1.0
        x["p"][sb] = 0.5
        x["cls_ids"][sb] = [CLS_MEASURE, 0, 0]
        x["cls_p"][sb] = [1.0, 0.0, 0.0]
    # 4c. a staff position read as something else entirely, confidently (a saturated or confused reading)
    wp = noteish & (r.random(n) < rate(0.03)) & (x["pos_v"] > -99)
    x["pos_v"][wp] = r.integers(-POS_MAX, POS_MAX + 1, wp.sum())
    x["pos_p"][wp] = r.uniform(0.6, 1.0, wp.sum()).astype(np.float32)
    # 4d. image features missing (a token the detector's map says nothing about; now and then a whole window)
    if "feat" in x:
        fm = r.random(n) < (1.0 if r.random() < 0.1 else rate(0.2))
        x["feat"][fm] = 0
    # 5. dropped tokens (misses)
    keep = r.random(n) >= rate(0.05)
    # 5b. a family the page does not print at all here (unmarked tuplets: the "3" left out after the first bars)
    if r.random() < 0.4 * min(1.0, strength):
        keep &= ~np.isin(x["fam"], UNPRINTED)
    keep[x["fam"] == FAMILIES.index("measure")] = True             # bars stay: they are the structure
    # 6. ghosts: copies of tokens, moved, re-classed, low confidence, not real
    ng = int(r.binomial(n, rate(0.05)))
    parts = {k: v[keep] for k, v in x.items()}
    if ng:
        src = r.integers(0, n, ng)
        gh = {k: v[src].copy() for k, v in x.items()}
        fam = r.integers(0, N_FAM, ng)
        for i in range(ng):
            if r.random() < 0.5:
                fam[i] = gh["fam"][i]
            c = int(r.choice(FAM_CLASSES[fam[i]]))
            gh["cls_ids"][i] = [c, 0, 0]; gh["cls_p"][i] = [1.0, 0, 0]
        gh["fam"] = fam.astype(np.int16)
        if "cand" in gh:
            gh["cand"][:] = 0
        gh["p"] = r.uniform(0.05, 0.7, ng).astype(np.float32)
        gh["geo"][:, 0] += r.normal(0, 0.1, ng); gh["geo"][:, 2] += r.normal(0, 0.15, ng)
        for k in TARGET_KEYS:
            gh[k][:] = IGNORE if k != "t_onset" else np.nan
        gh["t_real"][:] = 0
        isn = (gh["fam"] == NOTE_F) | (gh["fam"] == REST_F)
        gh["pos_v"][~isn] = -99
        for k in ("dots_p", "voice_p", "grace_p", "stem_p"):
            gh[k][~isn] = 0
        parts = {k: np.concatenate([parts[k], gh[k]]) for k in parts}
        # back into reading order: by system, staff row, bar column, x
        order = np.lexsort((parts["geo"][:, 0], parts["col"], parts["staffpos"], parts["sys"]))
        parts = {k: v[order] for k, v in parts.items()}
    return parts


def collate(items: list[dict], device) -> dict:
    return to_device(collate_np(items), device)


def to_device(nb: dict, device) -> dict:
    b = {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in nb.items()}
    b["p"] = b["p"].float(); b["cls_p"] = b["cls_p"].float()
    return b


def collate_np(items: list[dict]) -> dict:
    L = max(len(x["fam"]) for x in items)
    B = len(items)
    b = {}
    keys = list(items[0]) + [k for k in ("feat", "cand") if k not in items[0] and any(k in x for x in items)]
    for k in keys:
        v0 = next(x[k] for x in items if k in x)          # pages tokenised without image features: zeros
        shape = (B, L) + v0.shape[1:]
        fill = -99 if k == "pos_v" else (IGNORE if k.startswith("t_") and k != "t_onset" else
                                         (np.nan if k == "t_onset" else 0))
        arr = np.full(shape, fill, dtype=v0.dtype)
        for i, x in enumerate(items):
            if k in x:
                arr[i, :len(x[k])] = x[k]
        b[k] = arr
    pad = np.ones((B, L), bool)
    for i, x in enumerate(items):
        pad[i, :len(x["fam"])] = False
    b["pad"] = pad
    return b


class Batches(torch.utils.data.IterableDataset):
    """Endless corrupted windows, collated (numpy) -- built in loader workers, off the training loop."""

    def __init__(self, pages: list[dict], batch: int, seed: int):
        self.pages, self.batch, self.seed = [pg for pg in pages if pg["n_sys"] > 0], batch, seed
        # pages with tuplets are drawn three times as often (they are few, and their rhythm is the hard part)
        self.weights = [(TUPLET_WEIGHT if ((pg["t_tup"] > 0) & (pg["t_tup"] != IGNORE)).any() else 1.0) *
                        (MULTIVOICE_WEIGHT if ((pg["t_voice"] > 0) & (pg["t_voice"] != IGNORE)).any() else 1.0)
                        for pg in self.pages]

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        while True:
            items = []
            for pg in rng.choices(self.pages, weights=self.weights, k=self.batch):
                strength = rng.choice([0.0, 0.0, 0.5, 1.0, 1.5])
                x = window(pg, rng)
                items.append(corrupt(x, rng, strength) if strength > 0 else x)
            yield collate_np(items)


def loader(pages, batch, seed, workers=4):
    return iter(torch.utils.data.DataLoader(Batches(pages, batch, seed), batch_size=None, num_workers=workers,
                                            prefetch_factor=4, persistent_workers=False))


# ------------------------------------------------------------------------------------------------ loss
def losses(out: dict, b: dict) -> dict:
    """Per head; with a refinement pass, the first pass's too (keys "p1_<head>", weighted half in the total)."""
    L = _losses(out, b)
    if "_pass1" in out:
        L.update({"p1_" + k: v for k, v in _losses(out["_pass1"], b).items()})
    return L


def total_loss(Ls: dict):
    return sum((0.5 if k.startswith("p1_") else 1.0) * LOSS_W[k.removeprefix("p1_")] * v for k, v in Ls.items())


def _losses(out: dict, b: dict) -> dict:
    L = {}
    for k in ("real", "chord", "tie"):
        t = b["t_" + k].long()
        m = t != IGNORE
        if m.any():
            L[k] = F.binary_cross_entropy_with_logits(out[k][..., 0][m], t[m].float())
    for k in ("cls", "dots", "pos", "alter", "voice", "tup", "grace", "key", "time", "clef"):
        t = b["t_" + k].long()
        m = t != IGNORE
        if m.any():
            L[k] = F.cross_entropy(out[k][m], t[m])
    t = b["t_onset"]
    m = ~torch.isnan(t)
    if m.any():
        beat, frac = onset_classes(t[m])
        L["onset_beat"] = F.cross_entropy(out["onset_beat"][m], beat)
        L["onset_frac"] = F.cross_entropy(out["onset_frac"][m], frac)
    return L


def onset_classes(t):
    """onset in quarters -> (whole quarters clipped to ONSET_BEATS-1, 48ths of a quarter)."""
    from .model import ONSET_BEATS, ONSET_FRAC
    k = torch.round(t.clamp(min=0) * ONSET_FRAC).long()
    return (k // ONSET_FRAC).clamp(max=ONSET_BEATS - 1), k % ONSET_FRAC


@torch.no_grad()
def evaluate(model, val: list[dict], device, n_windows=600, seed=0, strength=1.0) -> dict:
    """Accuracy per head, the model's against the detector's own argmax where the detector has one."""
    model.eval()
    rng = random.Random(seed)
    stats: dict = {}

    def add(k, a, b_):
        s = stats.setdefault(k, [0, 0]); s[0] += a; s[1] += b_

    pages = [pg for pg in val if pg["n_sys"] > 0]
    for i in range(0, n_windows, 16):
        items = []
        for _ in range(16):
            pg = pages[rng.randrange(len(pages))]
            items.append(corrupt(window(pg, rng), rng, strength) if strength > 0 else window(pg, rng))
        b = collate(items, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(b)
        pri = model.priors(b)
        for k in ("real",):
            t = b["t_real"].long(); m = t != IGNORE
            add("real", ((out["real"][..., 0] > 0).long() == t)[m].sum().item(), m.sum().item())
            add("real@det", ((pri["real"][..., 0] > 0).long() == t)[m].sum().item(), m.sum().item())
        for k in ("cls", "dots", "pos", "voice", "grace"):
            t = b["t_" + k].long(); m = t != IGNORE
            add(k, (out[k].argmax(-1) == t)[m].sum().item(), m.sum().item())
            add(k + "@det", (pri[k].argmax(-1) == t)[m].sum().item(), m.sum().item())
        for k in ("alter", "tup", "key", "time", "clef"):
            t = b["t_" + k].long(); m = t != IGNORE
            add(k, (out[k].argmax(-1) == t)[m].sum().item(), m.sum().item())
        for k, none in (("key", N_KEY - 1), ("time", N_TIME - 1), ("clef", N_CLEF - 1), ("alter", 0)):
            t = b["t_" + k].long(); m = (t != IGNORE) & (t != none)        # bars that set one / explicit alters
            add(k + "_set", (out[k].argmax(-1) == t)[m].sum().item(), m.sum().item())
        for k in ("chord", "tie"):
            t = b["t_" + k].long(); m = t != IGNORE
            add(k, ((out[k][..., 0] > 0).long() == t)[m].sum().item(), m.sum().item())
    model.train()
    return {k: round(a / max(1, n), 4) for k, (a, n) in sorted(stats.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reload-every", type=int, default=5000, help="re-scan the data dir (the corpus grows)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", help="start from this checkpoint's weights (fine-tuning)")
    ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimiser step (batch = batch x accum)")
    ap.add_argument("--refine", type=int, default=0, help="layers of the refinement pass (0: none)")
    ap.add_argument("--fb", type=int, default=17, help="feedback features the refinement pass reads (17, 29 or 39)")
    ap.add_argument("--warmup", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = load_pages(a.data, a.limit)
    print(f"pages: train {len(train)} val {len(val)}", flush=True)
    feat_dim = next((pg["feat"].shape[1] for pg in train if "feat" in pg), 0)
    model = EvidenceNet(d=a.d, layers=a.layers, heads=a.heads, ff=4 * a.d, feat_dim=feat_dim, refine=a.refine,
                        fb=a.fb).to(device)
    print("image features", feat_dim, "candidates", USE_CANDS, flush=True)
    print("params", n_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    step0 = 0
    if a.init:
        sd = torch.load(a.init, map_location=device, weights_only=False)["model"]
        w = sd.get("fb.0.weight")
        if w is not None and getattr(model, "refine", 0) and w.shape != model.fb[0].weight.shape:
            # more feedback features than the model it starts from: theirs keep their weights, the new ones
            # start at zero (the model starts out reading as the old one did)
            nw = torch.zeros_like(model.fb[0].weight)
            nw[:, :w.shape[1]] = w
            sd["fb.0.weight"] = nw
            print(f"feedback features {w.shape[1]} -> {nw.shape[1]} (new ones zero)", flush=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected or [k for k in missing if not k.startswith(("feat_", "cand_", "fb.", "rblocks.", "rnorm."))]:
            raise RuntimeError(f"--init does not fit: missing {missing}, unexpected {unexpected}")
        if any(k.startswith("rnorm.") for k in missing):
            model.init_refine_from_base()       # the second pass starts as an identity on the first one's output
        print("initialised from", a.init, "new:", missing, flush=True)
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    warm = a.warmup
    sched = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))  # noqa: E731
    log = (out / "log.jsonl").open("a")
    t0 = time.time(); run: dict = {}
    best = -1.0
    it = loader(train, a.batch, step0, a.workers)
    for step in range(step0, a.steps):
        if step and step % a.reload_every == 0:
            train2, val2 = load_pages(a.data, a.limit)
            if len(train2) > len(train):
                train, val = train2, val2
                del it
                it = loader(train, a.batch, step, a.workers)
                print(f"reloaded: train {len(train)} val {len(val)}", flush=True)
        for gp in opt.param_groups:
            gp["lr"] = a.lr * sched(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            b = to_device(next(it), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                o = model(b)
            Ls = losses(o, b)
            loss = total_loss(Ls)
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in Ls.items():
            run[k] = run.get(k, 0.0) * 0.98 + 0.02 * v.item()
        if step % 200 == 0:
            print(f"step {step} lr {a.lr * sched(step):.2e} loss {loss.item():.3f} " +
                  " ".join(f"{k} {v:.3f}" for k, v in sorted(run.items())) + f" ({time.time() - t0:.0f}s)", flush=True)
        if (step + 1) % 2000 == 0 or step + 1 == a.steps:
            ev = evaluate(model, val, device, strength=1.0)
            ev0 = evaluate(model, val, device, strength=0.0)
            rec = {"step": step + 1, "noisy": ev, "clean": ev0, "time": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("EVAL", json.dumps(rec), flush=True)
            ck = {"model": model.state_dict(), "cfg": model.cfg, "opt": opt.state_dict(), "step": step + 1}
            torch.save(ck, out / "last.pt")
            score = sum(ev[k] for k in ("real", "cls", "dots", "alter", "key", "time"))
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "cfg": model.cfg, "step": step + 1, "eval": rec},
                           out / "best.pt")
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
