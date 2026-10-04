"""Read a page: detections -> front end -> the evidence model -> every symbol's final reading.

Each system is read in a window with its neighbours (the previous and the next system), and only its own tokens
are taken from that window. A symbol's reading carries, per head, the detector's choice, the final choice, its
probability, and the evidence (in nats) the context added for the final choice over the detector's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import features, front
from .model import PRIOR_HEADS, EvidenceNet
from .train import MAX_TOK, TOKEN_KEYS, collate
from .vocab import ALTERS, CLASSES, CLEFS, KEYS, POS_MAX, TIMES, TUPLETS, GRACE

ROOT = Path(__file__).resolve().parent


OPTIONAL = ("heads.onset",)


def load_model(path: str | Path, device="cpu") -> EvidenceNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = EvidenceNet(**{k: v for k, v in ck["cfg"].items()}).to(device).eval()
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    # an older checkpoint lacks heads added since (their outputs are then not read); anything else is an error
    if [k for k in missing if not k.startswith(OPTIONAL)] or [k for k in unexpected if not k.startswith(OPTIONAL)]:
        raise RuntimeError(f"checkpoint does not fit the model: missing {missing}, unexpected {unexpected}")
    m.onset_trained = not any(k.startswith("heads.onset") for k in missing)
    return m


@dataclass
class Reading:
    layout: front.Layout
    tok: dict                                  # sym index -> per-head readings
    attn: dict = field(default_factory=dict)   # sym index -> [(other sym index, weight)] for changed symbols


def _window_items(a: dict, s0: int, s1: int):
    sel = np.nonzero((a["sys"] >= s0) & (a["sys"] <= s1))[0]
    return sel


def _decode(k: str, z: np.ndarray):
    if k == "cls":
        return CLASSES[int(z.argmax())]
    if k == "pos":
        return int(z.argmax()) - POS_MAX
    if k == "alter":
        return ALTERS[int(z.argmax())]
    if k == "voice":
        return int(z.argmax()) + 1
    if k == "dots":
        return int(z.argmax())
    if k == "grace":
        return GRACE[int(z.argmax())]
    if k == "tup":
        return TUPLETS[int(z.argmax())]
    if k == "key":
        return KEYS[int(z.argmax())]
    if k == "time":
        return TIMES[int(z.argmax())]
    if k == "clef":
        return CLEFS[int(z.argmax())]
    raise KeyError(k)


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


@torch.no_grad()
def read(dets: list[dict], width: float, height: float, model: EvidenceNet, device="cpu",
         attention: bool = True, image=None, cands=None, feats=None) -> Reading:
    """``cands`` / ``feats``: the detector's sub-threshold candidates and image features (Detector.run_array
    with ``cand_conf``), for a model trained with them."""
    if not model.feat_dim:
        cands = feats = None
    L = front.build(dets, width, height, image=image, cands=cands)
    a = features.page_arrays(L, feats)
    keys = TOKEN_KEYS + (("feat", "cand") if feats is not None else ())
    n_sys = len(L.systems)
    tok: dict = {}
    attn: dict = {}
    for s in range(n_sys):
        lo, hi = max(0, s - 1), min(n_sys - 1, s + 1)
        sel = _window_items(a, lo, hi)
        if len(sel) > MAX_TOK:
            lo = hi = s
            sel = _window_items(a, s, s)
        chunks = [sel] if len(sel) <= MAX_TOK else [sel[i:i + MAX_TOK] for i in range(0, len(sel), MAX_TOK)]
        for sel in chunks:
            x = {k: a[k][sel].copy() for k in keys}
            x["sys"] = x["sys"] - lo
            x["col"] = x["col"] - x["col"].min()
            x["cls_mask"] = np.zeros(len(sel), np.int8)
            x["attr_mask"] = np.zeros(len(sel), np.int8)
            b = collate([x], torch.device(device))
            out = model(b, need_attn=attention)
            pri = model.priors(b)
            own = np.nonzero(a["sys"][sel] == s)[0]
            syms = a["sym"][sel]
            att = None
            if attention:
                # the last two layers, heads averaged: where each token looked
                att = torch.stack([t[0].mean(0) for t in out["_attn"][-2:]]).mean(0).float().cpu().numpy()
            for j in own:
                si = int(syms[j])
                # heads of the detector's family and of the model's final class (it may move a symbol across)
                fams = {front.family(L.syms[si].cls), front.family(_decode("cls", out["cls"][0, j].float().cpu().numpy()))}
                r = {}
                pr = 1 / (1 + np.exp(-out["real"][0, j, 0].item()))
                pr_det = 1 / (1 + np.exp(-pri["real"][0, j, 0].item()))
                r["real"] = {"det": round(pr_det, 4), "p": round(pr, 4),
                             "ev": round(float(out["_evidence"]["real"][0, j, 0]), 3)}
                heads = ["cls"]
                if fams & {"note", "rest"}:
                    heads += ["dots", "pos", "voice", "tup"]
                if "note" in fams:
                    heads += ["alter", "grace"]
                if "measure" in fams:
                    heads += ["key", "time", "clef"]
                for k in heads:
                    z = out[k][0, j].float().cpu().numpy()
                    p = _softmax(z)
                    rec = {"v": _decode(k, z), "p": round(float(p.max()), 4)}
                    if k in PRIOR_HEADS:
                        zp = pri[k][0, j].float().cpu().numpy()
                        ev = out["_evidence"][k][0, j].float().cpu().numpy()
                        if np.any(zp != 0):
                            rec["det"] = _decode(k, zp)
                            # evidence for the final choice over the detector's choice, in nats
                            rec["ev"] = round(float(ev[int(z.argmax())] - ev[int(zp.argmax())]), 3)
                    r[k] = rec
                if "note" in fams:
                    for k in ("chord", "tie"):
                        r[k] = round(float(1 / (1 + np.exp(-out[k][0, j, 0].item()))), 4)
                if fams & {"note", "rest"} and getattr(model, "onset_trained", True):
                    zb = out["onset_beat"][0, j].float().cpu().numpy(); zf = out["onset_frac"][0, j].float().cpu().numpy()
                    pb, pf = _softmax(zb), _softmax(zf)
                    r["onset"] = round(int(zb.argmax()) + int(zf.argmax()) / len(zf), 4)
                    r["onset_p"] = round(float(pb.max() * pf.max()), 4)
                    # full distributions for the writer's joint decoding of a voice's rhythm
                    r["onset_beat_p"] = np.round(pb, 5).tolist()
                    r["onset_frac_p"] = np.round(pf, 5).tolist()
                    r["tup_p"] = np.round(_softmax(out["tup"][0, j].float().cpu().numpy()), 5).tolist()
                tok[si] = r
                if att is not None and _changed(r):
                    w = att[j].copy(); w[j] = 0
                    top = np.argsort(-w)[:6]
                    attn[si] = [(int(syms[t]), round(float(w[t]), 4)) for t in top if w[t] > 0.02]
    return Reading(layout=L, tok=tok, attn=attn)


def _changed(r: dict) -> bool:
    if r["cls"].get("det", r["cls"]["v"]) != "measure" and (r["real"]["p"] >= 0.5) != (r["real"]["det"] >= 0.5):
        return True
    return any(isinstance(v, dict) and "det" in v and v["det"] != v["v"] for k, v in r.items() if k != "real")


def changes(rd: Reading) -> list[dict]:
    """Every symbol whose final reading differs from the detector's, with the evidence size."""
    out = []
    for si, r in rd.tok.items():
        s = rd.layout.syms[si]
        diffs = {}
        if s.fam != "measure" and (r["real"]["p"] >= 0.5) != (r["real"]["det"] >= 0.5):   # bars are structure
            diffs["real"] = (r["real"]["det"], r["real"]["p"], r["real"]["ev"])
        for k, v in r.items():
            if k != "real" and isinstance(v, dict) and "det" in v and v["det"] != v["v"]:
                diffs[k] = (v["det"], v["v"], v.get("ev"))
        if diffs:
            out.append({"sym": si, "cls": s.cls, "box": s.box, "staff": s.staff, "bar": s.bar, "diffs": diffs,
                        "context": rd.attn.get(si, [])})
    return out
