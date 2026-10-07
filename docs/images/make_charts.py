"""Draw the README's benchmark charts, in a light and a dark version (GitHub picks one by the reader's theme).

    python docs/images/make_charts.py

Every system against every other, per set, as the error rate OMR-NED (musicdiff, pooled over the set's pages; lower
is better). copista-28m and copista-2m are this code's run with its two detectors (28.7M and 1.94M parameters). The
Lieder scans are 55 pages: the dataset's 64 less the 9 whose ground truth lacks the vocal staff (Legato and Legato 2
publish 64-page figures only, so they are not in that panel). Legato and Legato 2 are published figures (Legato 2 is not released: its bars are hatched); homr 0.7,
Audiveris 5.11 and Transcoda were run by KobiMusic with the same scorer.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

SETS = [  # key, name, pages
    ("sq_scan", "String quartets, scans", 252),
    ("sq_render", "String quartets, renders", 252),
    ("lieder_scan", "Lieder, scans", 55),
    ("lieder_render", "Lieder, renders", 64),
    ("polish", "Polish piano scores, scans", 112),
]

# sets whose ground truth needs a word: an asterisk on the panel, the note under the chart
NOTES = {"polish": ("* Polish: the ground truth holds notes, rests, beams and tuplets only (no slurs, pedals, dynamics, "
                    "octave lines or text),",
                    "so whatever a reader reads of those counts against it. Notes and rests only: copista-28m 28.2 %, "
                    "copista-2m 33.4 %.")}

# system -> (kind, OMR-NED per set in SETS order; None = no figure)
SYSTEMS = {
    "copista-28m": ("evidence", [8.66, 5.83, 16.91, 14.56, 34.43]),
    "copista-2m": ("small", [11.21, 8.83, 17.03, 15.90, 38.67]),
    "Legato 2 (unreleased)": ("unreleased", [31.6, 17.1, None, 27.6, None]),
    "Legato": ("published", [58.2, 32.9, None, 39.5, None]),
    "homr 0.7": ("run", [None, None, 42.3, 38.0, 48.3]),
    "Audiveris 5.11": ("run", [66.9, 33.2, 51.8, 28.8, 62.6]),
    "Transcoda": ("run", [None, None, 48.4, 42.0, 55.1]),
}
LEGEND = [("evidence", "copista-28m"), ("small", "copista-2m"), ("published", "published figure"),
          ("unreleased", "published figure, system not released"), ("run", "run by KobiMusic, same scorer")]

THEMES = {
    "light": dict(text="#1f2328", muted="#59636e", grid="#d1d9e0", accent="#b73e6a", small="#e0a3bb",
                  published="#8c959f", run="#c4cbd2"),
    "dark": dict(text="#e6edf3", muted="#9198a1", grid="#3d444d", accent="#e2729b", small="#8e4d66",
                 published="#6e7681", run="#454c55"),
}

plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10.5,  # DejaVu has the arrows
                     "hatch.linewidth": 1.6})


def bar_style(t, kind):
    if kind == "unreleased":  # hatched: a figure from a paper, for a system nobody can run
        return dict(facecolor="none", edgecolor=t["published"], hatch="////", linewidth=1.2)
    return dict(color=t["accent"] if kind == "evidence" else t[kind])


def panels(theme, keys, title, out):
    t = THEMES[theme]
    rows = []
    for key in keys:
        i = [s[0] for s in SETS].index(key)
        bars = sorted(((name, neds[i], kind) for name, (kind, neds) in SYSTEMS.items() if neds[i] is not None),
                      key=lambda b: b[1])
        rows.append((SETS[i], bars))
    extra = 0.22 * sum(len(NOTES.get(k, ())) for k in keys)
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 0.33 * sum(len(b) for _, b in rows) + 0.75 * len(rows) + 1.3 + extra),
                             sharex=True, gridspec_kw=dict(height_ratios=[len(b) + 0.6 for _, b in rows]))
    notes = [line for key in keys for line in NOTES.get(key, ())]
    for ax, ((key, name, pages), bars) in zip(axes, rows):
        name = name + ("*" if key in NOTES else "")
        for y, (system, ned, kind) in enumerate(bars):
            ax.barh(y, ned, 0.72, **bar_style(t, kind))
            ax.text(ned + 0.8, y, f"{ned:.1f} %", va="center", fontsize=9.5, color=t["text"],
                    fontweight="bold" if system.startswith("copista") else "normal")
        ax.set_ylim(len(bars) - 0.4, -0.6)  # lowest error on top
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([b[0] for b in bars], color=t["text"])
        for label, b in zip(ax.get_yticklabels(), bars):
            if b[0].startswith("copista"):
                label.set_fontweight("bold")
        ax.set_title(f"{name}  ·  {pages} pages", loc="left", color=t["text"], fontsize=11, fontweight="bold",
                     pad=4)
        ax.set_xlim(0, 80)
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.tick_params(colors=t["muted"], length=0)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(range(0, 81, 20))
    axes[-1].set_xticklabels([f"{x} %" for x in range(0, 81, 20)])
    axes[-1].set_xlabel("← Error rate (OMR-NED), lower is better", color=t["muted"], fontsize=9.5, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_style(t, kind)) for kind, _ in LEGEND]
    fig.legend(handles, [label for _, label in LEGEND], loc="upper left",
               bbox_to_anchor=(0.012, 1 - 0.78 / fig.get_figheight()), ncol=3, frameon=False, fontsize=8.5,
               labelcolor=t["text"], handlelength=1.6, columnspacing=1.2)
    top = 1 - 0.32 / fig.get_figheight()
    fig.text(0.015, top, title, ha="left", va="center", color=t["text"], fontweight="bold", fontsize=13)
    fig.text(0.015, top - 0.3 / fig.get_figheight(), "Error rate = OMR-NED, the share of the score read wrong: "
             "lower is better ↓. Best at the top of each set.", ha="left", va="center", color=t["muted"],
             fontsize=9.5)
    bottom = 0.22 * len(notes) / fig.get_figheight()
    for k, line in enumerate(notes):
        fig.text(0.015, bottom - (k + 0.7) * 0.2 / fig.get_figheight(), line, ha="left", va="center",
                 color=t["muted"], fontsize=8.5)
    fig.tight_layout(rect=(0, bottom, 1, 1 - 1.2 / fig.get_figheight()), h_pad=1.2)
    fig.savefig(OUT / f"{out}-{theme}.png", dpi=200, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    for theme in THEMES:
        panels(theme, ["sq_scan", "lieder_scan", "polish"], "Scanned pages: copista-28m and copista-2m against other OMR systems",
               "compare-scans")
        panels(theme, ["sq_render", "lieder_render"], "Rendered pages: copista-28m and copista-2m against other OMR systems",
               "compare-renders")
