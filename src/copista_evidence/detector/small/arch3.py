"""Round 2b: two designs that drop the centre-heatmap readout altogether.

  col-slots     set prediction per COLUMN. The stride-4 feature map is read one column at a time: S learned slot
                queries cross-attend to the column's rows (a DETR decoder whose memory is one column), each slot
                emits presence, class, a pointer over the rows (which row is the centre), sub-cell offsets, size
                and attributes. Matching is Hungarian per column -- problems of at most S x S, so the assignment
                is exact yet cheap, and unlike page-level DETR the queries do not have to discover WHERE objects
                are, only WHAT sits in this column, top to bottom. No heatmap, no anchors, no 3x3 max-pool.
  staff-canon   detection in STAFF COORDINATES. A staff-line head finds the staves (ground truth at training,
                the head's own map at inference); the features of every staff are resampled onto a canonical
                strip whose rows are half staff spaces from the middle line (rows = staff positions -24..24) and
                whose columns are the page's stride-4 cells. Objects are detected on the strips, so a note's
                staff_position is the ROW it was found in -- geometric by construction, not learned -- and the
                detector never sees a staff at a scale or tilt other than canonical.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ATTR_OFFSETS, ATTR_SIZES, ATTRS, N_ATTR, render_staff_map, render_targets
from . import models as M
from .arch2 import Base, _dets, _sincos_pos, _peaks
from .models import ConvBNAct, Encoder, FPN, attr_loss, focal, gather_at, head

POS_MAX = 24


# ----------------------------------------------------------------------------------------------------------------
# 1. col-slots

class ColSlots(Base):
    def __init__(self, n_cls, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, d=64, S=12, layers=2, heads=4):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        self.S, self.d = S, d
        self.inp = nn.Linear(fpn, d)
        self.slot_q = nn.Parameter(torch.randn(S, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.pres = nn.Linear(d, 1); self.cls = nn.Linear(d, n_cls)
        self.ptr_q = nn.Linear(d, d); self.ptr_k = nn.Linear(d, d)          # slot -> row pointer
        self.reg = nn.Linear(d, 4)                                          # x off, y off (sub-cell), log w, log h (cells)
        self.attr = nn.Linear(d, N_ATTR)
        nn.init.constant_(self.pres.bias, -3.0)

    def forward(self, x):
        f = self.fpn(self.enc(x))                                            # [B, C, H, W]
        B, C, H, W = f.shape
        rows = f.permute(0, 3, 2, 1).reshape(B * W, H, C)                    # one sequence per column: its rows
        mem = self.inp(rows) + _sincos_pos(torch.arange(H, device=f.device), torch.zeros(H, device=f.device), self.d).to(rows.dtype)[None]
        q = self.slot_q[None].expand(B * W, -1, -1).to(mem.dtype)
        h = self.norm(self.dec(q, mem))                                      # [B*W, S, d]
        ptr = torch.einsum("nsd,nhd->nsh", self.ptr_q(h), self.ptr_k(mem)) / math.sqrt(self.d)      # [B*W, S, H]
        out = {"pres": self.pres(h).reshape(B, W, self.S), "cls": self.cls(h).reshape(B, W, self.S, -1),
               "ptr": ptr.reshape(B, W, self.S, H), "reg": self.reg(h).reshape(B, W, self.S, 4),
               "attr": self.attr(h).reshape(B, W, self.S, -1)}
        return out

    def loss(self, out, t, attr_cls_mask):
        from scipy.optimize import linear_sum_assignment
        B, W, S, H = out["ptr"].shape
        dev = out["ptr"].device
        # objects grouped by (sample, column)
        m = t["mask"]
        bi = torch.arange(B, device=dev)[:, None].expand_as(t["ind"])[m]
        ind = t["ind"][m]; ci, cj = ind // W, ind % W
        key = bi * W + cj
        order = torch.argsort(key); key = key[order]
        uniq, first, counts = torch.unique_consecutive(key, return_inverse=True, return_counts=True)
        rank = torch.arange(len(key), device=dev) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        keep = rank < S
        n_dropped = int((~keep).sum())
        Ncol = len(uniq)
        if Ncol == 0:
            l = F.binary_cross_entropy_with_logits(out["pres"].float(), torch.zeros_like(out["pres"].float()))
            return l, {"pres": float(l)}, {}
        G = int(min(S, counts.max()))
        col_b, col_j = uniq // W, uniq % W
        # padded GT per column [Ncol, G]
        def pack(v, fill=0):
            o = torch.full((Ncol, G) + v.shape[1:], fill, dtype=v.dtype, device=dev)
            o[first[keep], rank[keep]] = v[order][keep]
            return o
        valid = pack(torch.ones(len(key), dtype=torch.bool, device=dev), False)
        g_cls = pack(t["cls"][m]); g_row = pack(ci); g_off = pack(t["off"][m]); g_wh = pack(t["wh"][m]); g_attrs = pack(t["attrs"][m])
        # predictions of those columns [Ncol, S, ...]
        p_cls = torch.log_softmax(out["cls"][col_b, col_j].float(), -1)          # [Ncol, S, n_cls]
        p_ptr = torch.log_softmax(out["ptr"][col_b, col_j].float(), -1)          # [Ncol, S, H]
        with torch.no_grad():
            cost = -(p_cls.exp().gather(2, g_cls[:, None, :].expand(-1, S, -1)) + p_ptr.exp().gather(2, g_row[:, None, :].expand(-1, S, -1)))
            cost = cost.masked_fill(~valid[:, None, :], 10.0).cpu().numpy()      # [Ncol, S, G]
            nv = valid.sum(1).cpu().numpy()
            ms, mg, mc = [], [], []
            for c in range(Ncol):
                r, g = linear_sum_assignment(cost[c, :, :nv[c]])
                ms.append(r); mg.append(g); mc.append(np.full(len(r), c))
            ms = torch.as_tensor(np.concatenate(ms), device=dev); mg = torch.as_tensor(np.concatenate(mg), device=dev); mc = torch.as_tensor(np.concatenate(mc), device=dev)
        losses = {}
        # presence: 1 at matched slots, 0 elsewhere (every column of every sample)
        pres_t = torch.zeros(B, W, S, device=dev)
        pres_t[col_b[mc], col_j[mc], ms] = 1.0
        losses["pres"] = F.binary_cross_entropy_with_logits(out["pres"].float(), pres_t, reduction="sum") / max(1, len(ms)) * 0.5
        losses["cls"] = F.nll_loss(p_cls[mc, ms], g_cls[mc, mg])
        losses["ptr"] = F.nll_loss(p_ptr[mc, ms], g_row[mc, mg])
        reg = out["reg"][col_b[mc], col_j[mc], ms].float()
        losses["off"] = F.l1_loss(reg[:, :2], g_off[mc, mg])
        losses["wh"] = 0.1 * F.l1_loss(reg[:, 2:], g_wh[mc, mg])
        # attributes on matched note / rest slots (round 1's masks, the matched slots laid out as one row of "cells")
        attr = out["attr"][col_b[mc], col_j[mc], ms]                                          # [n, N_ATTR]
        n = len(ms)
        tt = {"ind": torch.arange(n, device=dev)[None], "mask": torch.ones(1, n, dtype=torch.bool, device=dev),
              "cls": g_cls[mc, mg][None], "attrs": g_attrs[mc, mg][None]}
        al, parts = attr_loss(attr.t()[None, :, :, None], tt, attr_cls_mask)
        losses["attr"] = al
        parts["drop"] = n_dropped
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.5):
        B, W, S, H = out["ptr"].shape
        res = []
        for b in range(B):
            pres = torch.sigmoid(out["pres"][b].float())                       # [W, S]
            pc, cls = torch.softmax(out["cls"][b].float(), -1).max(-1)
            score = (pres * pc).reshape(-1)
            keep = score > thresh
            if keep.sum() > K:
                keep = torch.zeros_like(keep); keep[score.topk(K).indices] = True
            idx = keep.nonzero()[:, 0]
            col, slot = idx // S, idx % S
            row = out["ptr"][b].float().reshape(-1, H)[idx].argmax(-1)
            reg = out["reg"][b].float().reshape(-1, 4)[idx]
            x = (col.float() + reg[:, 0]) * self.stride; y = (row.float() + reg[:, 1]) * self.stride
            w = torch.exp(reg[:, 2]) * self.stride; h = torch.exp(reg[:, 3]) * self.stride
            boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
            attrs = out["attr"][b].float().reshape(-1, N_ATTR)[idx]
            res.append(_dets(boxes, score[idx], cls.reshape(-1)[idx], attrs, row, col, nms_iou))
        return res


# ----------------------------------------------------------------------------------------------------------------
# 2. staff-canon

def find_staves(staff_map, stride=4, step=4, min_peak=0.3, min_len=4, win=3, max_gap=25, merge=True, tol=0.2, extend=False):
    """Staves from a predicted staff-line map [Hs, Ws] (sigmoid): every ``step`` columns, the peaks of the column
    (averaged over ``2 win + 1`` columns, sampled 4x finer than the cells) are grouped into runs of five with
    near-equal spacing; groups are linked across columns into tracks (a track survives a gap of ``max_gap``
    samples, e.g. under a beam), collinear tracks are merged, and a track of at least ``min_len`` samples whose
    spacing agrees with the page's becomes a staff of five polylines (pixels, top to bottom)."""
    sm = staff_map
    Hs, Ws = sm.shape
    tracks = []                                                             # dicts: xs, mids, sps, last_j
    up = 4
    fine = np.arange(Hs * up) / up
    for j in range(1, Ws - 1, step):
        col = sm[:, max(0, j - win):j + win + 1].mean(1)
        col = np.interp(fine, np.arange(Hs), col)
        ys = np.nonzero((col[1:-1] >= min_peak) & (col[1:-1] >= col[:-2]) & (col[1:-1] > col[2:]))[0] + 1
        if len(ys) < 5:
            continue
        peaks = (ys / up + 0.5) * stride
        groups = []
        s0 = 0
        while s0 + 4 < len(peaks):
            g = np.diff(peaks[s0:s0 + 5]); med = np.median(g)
            if med > 0 and (np.abs(g - med) <= tol * med).all():
                groups.append((peaks[s0 + 2], med)); s0 += 5
            else:
                s0 += 1
        x = (j + 0.5) * stride
        for mid, sp in groups:
            best = None
            for tr in tracks:
                if j - tr["last_j"] <= max_gap * step and abs(tr["mids"][-1] - mid) < 0.6 * sp and abs(tr["sps"][-1] - sp) < 0.25 * sp:
                    if best is None or abs(tr["mids"][-1] - mid) < abs(best["mids"][-1] - mid):
                        best = tr
            if best is None:
                tracks.append({"xs": [x], "mids": [mid], "sps": [sp], "last_j": j})
            else:
                best["xs"].append(x); best["mids"].append(mid); best["sps"].append(sp); best["last_j"] = j
    if merge:
        # collinear fragments (same middle line, disjoint x ranges) -> one staff
        tracks.sort(key=lambda t: t["xs"][0])
        merged = []
        for tr in tracks:
            m0 = np.median(tr["mids"]); sp0 = np.median(tr["sps"])
            for mt in merged:
                if abs(np.median(mt["mids"]) - m0) < 0.6 * sp0 and abs(np.median(mt["sps"]) - sp0) < 0.25 * sp0 and tr["xs"][0] > mt["xs"][-1]:
                    mt["xs"] += tr["xs"]; mt["mids"] += tr["mids"]; mt["sps"] += tr["sps"]; break
            else:
                merged.append(tr)
        tracks = merged
    staves = []
    if tracks:
        all_sp = np.concatenate([tr["sps"] for tr in tracks])
        page_sp = float(np.median(all_sp))
    keep = [tr for tr in tracks if len(tr["xs"]) >= min_len and abs(float(np.median(tr["sps"])) - page_sp) <= 0.3 * page_sp]
    if extend and keep:
        # the staves of a page share the systems' width: extend every track to the page's sampled x-range (the end
        # values are held, so a straight staff is extrapolated; a strip over a margin only adds negatives)
        x_lo = min(tr["xs"][0] for tr in keep); x_hi = max(tr["xs"][-1] for tr in keep)
        for tr in keep:
            if tr["xs"][0] > x_lo:
                tr["xs"].insert(0, x_lo); tr["mids"].insert(0, tr["mids"][0]); tr["sps"].insert(0, tr["sps"][0])
            if tr["xs"][-1] < x_hi:
                tr["xs"].append(x_hi); tr["mids"].append(tr["mids"][-1]); tr["sps"].append(tr["sps"][-1])
    for tr in keep:
        xs = np.asarray(tr["xs"], np.float32); mids = np.asarray(tr["mids"], np.float32); sps = np.asarray(tr["sps"], np.float32)
        staves.append([np.stack([xs, mids + (k - 2) * sps], 1) for k in range(5)])
    return staves


def _focal_dense(logits, g, eps=1e-4):
    """Penalty-reduced focal terms of a chunk of strips, summed -- masks instead of boolean indexing, so no device
    sync per strip (96 strips x 2 syncs a step cost 8 s under a loaded GPU)."""
    p = torch.sigmoid(logits.float()).clamp(eps, 1 - eps)
    pos = g.eq(1).float()
    pos_loss = (torch.log(p) * (1 - p) ** 2 * pos).sum()
    neg_loss = (torch.log1p(-p) * p * p * (1 - g) ** 4 * (1 - pos)).sum()
    return -(pos_loss + neg_loss)


def focal_strips(pred, gt, chunk=16):
    from torch.utils.checkpoint import checkpoint
    n = gt.eq(1).sum().clamp(min=1).float()
    total = pred.new_zeros((), dtype=torch.float32)
    for s in range(0, pred.shape[0], chunk):
        total = total + checkpoint(_focal_dense, pred[s:s + chunk], gt[s:s + chunk], use_reentrant=False)
    return total / n


def _five_lines(staff):
    """The five lines of a staff group, top to bottom. The label writer's grouping occasionally hands a 6-line group
    (a 5-line staff plus a stray line: drop the line whose removal leaves the most even spacing) or a 4-line one
    (which line is missing is unknowable: skipped)."""
    lines = sorted((np.asarray(l, np.float32) for l in staff if len(l) >= 2), key=lambda l: float(l[:, 1].mean()))
    if len(lines) == 5:
        return lines
    if len(lines) == 6:
        ys = np.array([float(l[:, 1].mean()) for l in lines])
        best = None
        for drop in range(6):
            g = np.diff(np.delete(ys, drop))
            score = g.std() / max(g.mean(), 1e-3)
            if best is None or score < best[0]:
                best = (score, drop)
        if best[0] < 0.15:
            return [l for k, l in enumerate(lines) if k != best[1]]
    return None


class StaffCanon(Base):
    R = POS_MAX                     # rows -R..R = staff positions; row i <-> position R - i
    MAX_STRIPS = 64                 # per batch (memory: 96 strips took 14.4 GB a run)

    def __init__(self, n_cls, note_mask=None, widths=(24, 48, 96, 176, 256), depths=(1, 2, 3, 2), fpn=64, hd=64, attr_hd=64):
        super().__init__(n_cls)
        self.enc = Encoder(widths, depths); self.fpn = FPN(self.enc.out_channels, fpn)
        # notes take their staff_position from the strip row; rests keep the learned head (a rest's labelled position
        # is its anchor line, not its box centre)
        self.register_buffer("note_mask", torch.as_tensor(note_mask if note_mask is not None else [True] * n_cls, dtype=torch.bool))
        self.staff = head(fpn, 16, 1, prior=0.05)
        self.strip = nn.Sequential(ConvBNAct(fpn, fpn, 3), ConvBNAct(fpn, fpn, 3, d=2))
        self.hm = head(fpn, hd, n_cls, prior=0.01)
        self.reg = head(fpn, hd, 4)                                          # log w (cells), log h (rows), x off (cells), y off (rows)
        self.attr = head(fpn, attr_hd, N_ATTR, dilations=(1, 2, 4))
        self._staffs = None; self._sp = None
        self.n_rows = 2 * self.R + 1

    def set_batch(self, batch):
        self._staffs = batch.get("staffs"); self._sp = batch.get("sp")

    # strips --------------------------------------------------------------------------------------------------
    def _strip_geometry(self, staff, W_cells, sp_fallback):
        """(x0 cell, mid per column [Wc], sp per column [Wc]) for one staff (5 polylines, pixels) or None."""
        staff = _five_lines(staff)
        if staff is None:                                                    # 1-line percussion, 6-line tab, broken groups: no strip
            return None
        top, mid, bot = staff[0], staff[2], staff[4]
        if len(mid) < 2:
            return None
        xa, xb = float(max(mid[:, 0].min(), 0)), float(mid[:, 0].max())
        j0, j1 = int(math.floor(xa / self.stride)), int(math.ceil(xb / self.stride))
        j0 = max(0, j0); j1 = min(W_cells, j1)
        if j1 - j0 < 2:
            return None
        cols = (np.arange(j0, j1, dtype=np.float32) + 0.5) * self.stride
        m = np.interp(cols, mid[:, 0], mid[:, 1]).astype(np.float32)
        sp = (np.interp(cols, bot[:, 0], bot[:, 1]) - np.interp(cols, top[:, 0], top[:, 1])) / 4
        sp = np.where(sp > 1.0, sp, sp_fallback or 10.0).astype(np.float32)
        return j0, m, sp

    def _strips(self, f, staffs_per_image, sps):
        """Resample f [B, C, H, W] onto canonical strips. Returns strips [Ns, C, rows, Wmax] and the strip geometry
        dict: b [Ns], j0 [Ns] (first page column), wc [Ns] (width in columns), mid / sp [Ns, Wmax] (pixels, padded
        with the last column's value), valid [Ns, Wmax]. One host-to-device copy for the whole batch: per-strip
        copies cost 4 s a step under a loaded GPU."""
        B, C, H, W = f.shape
        geos = []
        for b, staffs in enumerate(staffs_per_image):
            for st in staffs or []:
                g = self._strip_geometry(st, W, sps[b] if sps else None)
                if g is not None:
                    geos.append((b, g[0], len(g[1]), g[1], g[2]))
        if len(geos) > self.MAX_STRIPS:
            sel = np.random.default_rng(0).choice(len(geos), self.MAX_STRIPS, replace=False)
            geos = [geos[i] for i in sorted(sel)]
        if not geos:
            return None, None
        Ns = len(geos); Wmax = max(g[2] for g in geos)
        mid = np.zeros((Ns, Wmax), np.float32); sp = np.ones((Ns, Wmax), np.float32) * 10.0
        for i, (b, j0, Wc, m_, s_) in enumerate(geos):
            mid[i, :Wc] = m_; mid[i, Wc:] = m_[-1]; sp[i, :Wc] = s_; sp[i, Wc:] = s_[-1]
        dev = f.device
        geo = {"b": torch.tensor([g[0] for g in geos], device=dev), "j0": torch.tensor([g[1] for g in geos], device=dev),
               "wc": torch.tensor([g[2] for g in geos], device=dev), "mid": torch.from_numpy(mid).to(dev), "sp": torch.from_numpy(sp).to(dev)}
        geo["valid"] = torch.arange(Wmax, device=dev)[None, :] < geo["wc"][:, None]
        rows = torch.arange(self.n_rows, device=dev).float()                                  # i: 0 = top (+R) .. 2R = bottom (-R)
        xs = (geo["j0"][:, None].float() + torch.arange(Wmax, device=dev).float()[None, :] + 0.5) * self.stride    # [Ns, Wmax] px
        ys = geo["mid"][:, None, :] + (rows[None, :, None] - self.R) * geo["sp"][:, None, :] / 2                   # [Ns, rows, Wmax] px
        grid = torch.stack([(2 * xs / (W * self.stride) - 1)[:, None, :].expand(-1, self.n_rows, -1), 2 * ys / (H * self.stride) - 1], -1)
        strips = torch.zeros(Ns, C, self.n_rows, Wmax, device=dev)
        bl = geo["b"].tolist()
        for b in range(B):
            sel = [i for i, bb in enumerate(bl) if bb == b]
            if sel:
                strips[sel] = F.grid_sample(f[b:b + 1].float().expand(len(sel), -1, -1, -1), grid[sel], mode="bilinear", padding_mode="zeros", align_corners=False)
        return strips.to(f.dtype), geo

    def forward(self, x):
        f = self.fpn(self.enc(x))
        staff_map = self.staff(f)
        if self.training and self._staffs is not None:
            staffs = self._staffs
        else:
            staffs = [find_staves(torch.sigmoid(staff_map[b, 0].float()).cpu().numpy(), stride=self.stride) for b in range(x.shape[0])]
        strips, geo = self._strips(f, staffs, self._sp if self.training else None)
        out = {"staff": staff_map, "geo": geo}
        if strips is not None:
            out["valid"] = geo["valid"]
            s = self.strip(strips)
            out.update({"hm": self.hm(s), "reg": self.reg(s), "attr": self.attr(s)})
        return out

    # training -------------------------------------------------------------------------------------------------
    def _strip_boxes(self, t, geo, W):
        """The batch's objects re-expressed per strip, [n, 10] boxes in strip units (columns, rows) -- one list per
        strip, computed over all (strip, object) pairs at once."""
        dev = t["ind"].device
        m = t["mask"]
        ob = torch.arange(t["ind"].shape[0], device=dev)[:, None].expand_as(t["ind"])[m]
        ind = t["ind"][m]; ci, cj = ind // W, ind % W
        off = t["off"][m]; wh = t["wh"][m]
        cx = (cj.float() + off[:, 0]) * self.stride; cy = (ci.float() + off[:, 1]) * self.stride
        w = torch.exp(wh[:, 0]) * self.stride; h = torch.exp(wh[:, 1]) * self.stride
        cls = t["cls"][m].float(); attrs = t["attrs"][m].float()
        Ns = geo["b"].shape[0]
        pair = (geo["b"][:, None] == ob[None, :]).nonzero()                                    # [P, 2]: strip, object
        si, oi = pair[:, 0], pair[:, 1]
        jc = cx[oi] / self.stride - geo["j0"][si].float()
        wc = geo["wc"][si]
        inside = (jc >= 0) & (jc < wc.float())
        jj = jc.floor().long().clamp(min=0); jj = torch.minimum(jj, wc - 1)
        mid = geo["mid"][si, jj]; sp = geo["sp"][si, jj]
        r = 2 * (mid - cy[oi]) / sp                                                            # staff position (continuous)
        ic = self.R - r
        ok = inside & (ic >= 0) & (ic < self.n_rows)
        # every object belongs to ONE strip. A note with a labelled staff_position goes to the strip whose row matches
        # the label (its logical staff -- cross-staff notes sit visually on the other staff); everything else goes to
        # the staff whose middle line is nearest. An object supervised in two strips would carry two positions.
        pos_id = t["attrs"][m][:, 0]
        labelled = (pos_id > 0) & self.note_mask[t["cls"][m]]
        target_r = torch.where(labelled, (pos_id - (POS_MAX + 1)).float(), torch.zeros_like(pos_id, dtype=torch.float32))
        dist = (r - target_r[oi]).abs().masked_fill(~ok, float("inf"))
        nearest = torch.full((cx.shape[0],), float("inf"), device=dev).scatter_reduce(0, oi, dist, reduce="amin")
        ok = ok & (dist <= nearest[oi]) & ~(labelled[oi] & (dist > 1.5))
        hs = h[oi] / (sp / 2); ws = w[oi] / self.stride
        bx = torch.cat([cls[oi, None], (jc - ws / 2)[:, None], (ic - hs / 2)[:, None], (jc + ws / 2)[:, None], (ic + hs / 2)[:, None], attrs[oi]], 1)
        bx = bx[ok]; si = si[ok]
        order = torch.argsort(si); bx = bx[order]; si = si[order]
        counts = torch.bincount(si, minlength=Ns).tolist()
        return list(torch.split(bx, counts))

    def loss(self, out, t, attr_cls_mask):
        losses = {}; parts = {}
        Hs, Ws = out["staff"].shape[-2:]
        staff_gt = render_staff_map([[poly for st in staffs for poly in st] for staffs in (self._staffs or [[]] * out["staff"].shape[0])],
                                    Hs * self.stride, Ws * self.stride, self.stride, out["staff"].device, sp_list=self._sp)
        losses["staff"] = F.binary_cross_entropy_with_logits(out["staff"].float(), staff_gt)
        if "hm" not in out:
            return losses["staff"] + out["staff"].sum() * 0, {k: float(v) for k, v in losses.items()}, parts
        boxes = self._strip_boxes(t, out["geo"], Ws)
        Ns, _, R_, Wmax = out["hm"].shape
        ts = render_targets(boxes, R_, Wmax, 1, self.n_cls, out["hm"].device)
        # columns beyond a strip's width are padding: no heatmap supervision there
        ts["hm"] = ts["hm"] * out["valid"][:, None, None, :].float()
        losses["hm"] = focal_strips(out["hm"].masked_fill(~out["valid"][:, None, None, :], -20.0), ts["hm"])
        reg = gather_at(out["reg"], ts["ind"])
        m = ts["mask"].unsqueeze(-1).float(); n = ts["mask"].sum().clamp(min=1)
        losses["wh"] = 0.1 * (F.l1_loss(reg[..., :2], ts["wh"], reduction="none") * m).sum() / n
        losses["off"] = (F.l1_loss(reg[..., 2:], ts["off"], reduction="none") * m).sum() / n
        al, parts = attr_loss(out["attr"], ts, attr_cls_mask)
        losses["attr"] = al
        # how often the strip row agrees with the labelled staff_position (notes / rests with a position)
        with torch.no_grad():
            pos_id = ts["attrs"][..., 0]; has = ts["mask"] & (pos_id > 0) & self.note_mask[ts["cls"]]
            if has.any():
                cy_rows = (ts["ind"] // Wmax).float() + ts["off"][..., 1]
                geo = torch.round(self.R - cy_rows).long() + POS_MAX + 1
                parts["row=pos"] = float((geo[has] == pos_id[has]).float().mean())
            parts["strips"] = Ns; parts["objs"] = int(ts["mask"].sum())
        return sum(losses.values()), {k: float(v) for k, v in losses.items()}, parts

    # inference ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decode_out(self, out, K=3000, thresh=0.03, nms_iou=0.5):
        B = out["staff"].shape[0]
        dev = out["staff"].device
        per = [[] for _ in range(B)]
        if "hm" in out:
            geo = out["geo"]
            hm = torch.sigmoid(out["hm"].float()).masked_fill(~geo["valid"][:, None, None, :], 0.0)
            Ns, C, R_, Wmax = hm.shape
            bl = geo["b"].tolist()
            for s in range(Ns):
                sc, idx = _peaks(hm[s], K, thresh)
                if len(idx) == 0:
                    continue
                b = bl[s]
                ch = idx // (R_ * Wmax); cell = idx % (R_ * Wmax); i = cell // Wmax; j = cell % Wmax
                reg = out["reg"][s].float().reshape(4, -1)[:, cell]
                attr = out["attr"][s].float().reshape(N_ATTR, -1)[:, cell].t()
                jc = j.float() + reg[2]; ic = i.float() + reg[3]                      # reg = log w, log h, x off, y off (round 1's layout)
                mid_j = geo["mid"][s, j]; sp_j = geo["sp"][s, j]
                x = (geo["j0"][s].float() + jc) * self.stride
                y = mid_j + (ic - self.R) * sp_j / 2
                w = torch.exp(reg[0]) * self.stride; h = torch.exp(reg[1]) * sp_j / 2
                boxes = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
                # staff_position by construction: the row the object was found in
                pos = torch.round(self.R - ic).long().clamp(-POS_MAX, POS_MAX) + POS_MAX + 1
                o = ATTR_OFFSETS["staff_position"]; n_ = ATTR_SIZES["staff_position"]
                isn = self.note_mask[ch]
                attr[isn, o:o + n_] = -20.0
                attr[isn.nonzero()[:, 0], o + pos[isn]] = 20.0
                per[b].append((boxes, sc, ch, attr, (y / self.stride).long(), (x / self.stride).long()))
        res = []
        for b in range(B):
            if not per[b]:
                z = torch.zeros(0, device=dev)
                res.append(_dets(torch.zeros(0, 4, device=dev), z, z.long(), torch.zeros(0, N_ATTR, device=dev), z.long(), z.long(), 0)); continue
            cat = [torch.cat([p[k] for p in per[b]]) for k in range(6)]
            res.append(_dets(*cat, nms_iou))
        return res


CONFIGS3 = {"col-slots": ColSlots, "staff-canon": StaffCanon}


def build3(name, classes, **override):
    if name == "staff-canon":
        override.setdefault("note_mask", [c.startswith("note") for c in classes])
    return CONFIGS3[name](len(classes), **override)
