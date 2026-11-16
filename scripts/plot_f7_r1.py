#!/usr/bin/env python3
"""Task 10.2: F7, s_R's per-window scores per workload (spec §6), from
results/heldout/r1_scores.json.

Two panels. (a) s_R as frozen (full declaration: decode, prefill, idle), with
its tau_alpha. (b) the decode-only s_R of T4's mis-specified declaration, with
its two thresholds (pooled calibration and decode calibration). Each panel is a
box-plot row (plotstyle.boxrow) for the declared windows of each workload and
split, the training windows of each co-location run, and development
honest-training and non-AI windows. The fraction of each row's windows above
each drawn threshold goes to the sidecar, not the figure. The unthrottled run's (`-0p5`) ROC AUC against test decode
windows, and the fraction of test decode windows that score above the run's
minimum, are recorded for the caption (exploratory, descriptive). Writes
figures/f7_r1.pdf and figures/f7_r1.json, which records the thresholds as drawn
and every annotated number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import plotstyle as ps  # noqa: E402
from plotstyle import plt  # noqa: E402

SRC = REPO / "results" / "heldout" / "r1_scores.json"
OUT = REPO / "figures" / "f7_r1.pdf"
UNPACED = "co-location panel-mixture-concurrent-0p5"
DECODE_TEST = "declared test serving-decode"
#: Panels, left to right: (model key, letter, title,
#: [(threshold key, line label, dash pattern)],
#: [(x range, x ticks, width share)] per segment of a broken score axis).
#: Panel b's non-decode rows all sit within 0.05 of its pooled threshold, so
#: its axis is broken and that range is drawn wider.
PANELS = (("r1", "a", "$s_R$, complete reference (decode, prefill, idle)",
           [("declared", r"$\tau_\alpha$", (0, (1, 1.5)))],
           [((0.37, 0.82), [0.4, 0.5, 0.6, 0.7, 0.8], 1.0)]),
          ("decode_only", "b", "$s_R$, decode-only fit",
           [("pooled calibration", r"$\tau_\alpha$ decode + idle: silent", (0, (1, 1.5))),
            ("decode calibration", r"$\tau_\alpha$ decode: loud", (0, (4, 1.5, 1, 1.5)))],
           [((0.36, 0.66), [0.4, 0.5, 0.6], 0.5),
            ((0.745, 0.812), [0.75, 0.775, 0.8], 0.5)]))
#: Text holding sub- or superscripts: mathtext sets scripts at 0.7 of this, so
#: they print at 5.0 pt (ps.MATH_PT gives 4.9).
SCRIPT_PT = 7.2
#: Height (axes fraction) of the panel letters and titles, one line for both panels.
HEAD_Y = 1.1
SPLIT = {"development": "dev", "calibration": "cal", "test": "test"}
#: Row groups, top to bottom: (header, predicate on the score-set key).
GROUPS = (("declared", lambda k: k.startswith("declared ")),
          ("co-location", lambda k: k.startswith("co-location ")),
          ("other workloads", lambda k: not k.startswith(("declared ", "co-location "))))
#: Gap (in rows) between row groups; each group's header sits 0.75 rows above its
#: first strip, so it reads with that group.
GROUP_GAP = 1.3


def label_colour(key: str) -> tuple[str, str]:
    """Display label and colour of one score set in r1_scores.json."""
    if key.startswith("declared "):
        _, split, w = key.split(" ")
        return f"{w.replace('serving-', '')} ({SPLIT[split]})", ps.SERIES["declared inference"]
    if key.startswith("co-location "):
        run = key.split(" ")[1]
        return ps.label(run).replace("co-location, ", ""), ps.SERIES[run]
    if key == "honest training":
        return "honest training (dev)", ps.SERIES["honest training"]
    return "non-AI (dev)", ps.NON_AI


def rows(keys: list[str]) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """(key, y) per strip in group order, and (header, y) per group."""
    out, heads, y = [], [], 0.0
    for name, pred in GROUPS:
        ks = [k for k in keys if pred(k)]
        heads.append((name, y))
        for k in ks:
            out.append((k, y))
            y += 1
        y += GROUP_GAP
    return out, heads


def main() -> None:
    ps.apply()
    r = json.loads(SRC.read_text())
    fig = plt.figure(figsize=(ps.PAGE, 2.6))
    outer = fig.add_gridspec(1, len(PANELS), wspace=0.1)
    annotated, first, heads_ax = {}, None, {}
    top = -0.75
    for gs, (m, letter, title, taus, segs) in zip(outer, PANELS):
        scores = r["scores"][m]
        order, heads = rows(list(scores))
        inner = gs.subgridspec(1, len(segs), wspace=0.06,
                               width_ratios=[w for _, _, w in segs])
        axs = [fig.add_subplot(inner[i], sharey=first) for i in range(len(segs))]
        first = first or axs[0]
        drawn = {tk: r["thresholds"][m][tk] for tk, _, _ in taus}
        pct = {}
        for ax, (xlim, ticks, _) in zip(axs, segs):
            for tk, tlab, ls in taus:
                if xlim[0] < drawn[tk] < xlim[1]:
                    ax.axvline(drawn[tk], color=ps.INK, lw=0.7, ls=ls, zorder=0)
                    ax.text(drawn[tk], top, tlab, ha="center", va="bottom", fontsize=SCRIPT_PT,
                            color=ps.INK)
            ps.highlight_row(ax, next(y for key, y in order if key == UNPACED))
            for key, y in order:
                ps.boxrow(ax, np.array(scores[key]), y, label_colour(key)[1])
            ax.set_xlim(*xlim)
            # The zoomed segment of a broken axis labels its end ticks only.
            short = xlim[1] - xlim[0] < 0.1
            ax.set_xticks(ticks, ["" if short and 0 < i < len(ticks) - 1 else f"{x:g}"
                                  for i, x in enumerate(ticks)])
            # A broken axis keeps its spine to the break, so the slashes sit on it.
            ax.spines["bottom"].set_bounds(*(xlim if len(segs) > 1 else ticks[::len(ticks) - 1]))
            ax.set_ylim(order[-1][1] + 0.6, top - 0.1)
            ax.tick_params(axis="y", length=0, labelleft=False)
            ax.spines["left"].set_visible(False)
        for a, b in zip(axs, axs[1:]):
            ps.axis_break(a, b)
        for key, y in order:
            s = np.array(scores[key])
            pct[label_colour(key)[0]] = [ps.percent((s > t).mean()) for t in drawn.values()]
        axs[0].text(0, HEAD_Y, title, transform=axs[0].transAxes, ha="left",
                    va="baseline", fontsize=SCRIPT_PT, color=ps.INK)
        heads_ax[letter] = axs[0]
        mid = fig.add_subplot(gs, frameon=False)
        mid.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        mid.set_xlabel("score $s_R$", labelpad=15, fontsize=SCRIPT_PT)
        entry = {"thresholds_drawn": drawn, "percent_above_thresholds": pct}
        if m == "r1":
            u, dec = np.array(scores[UNPACED]), np.array(scores[DECODE_TEST])
            auc = roc_auc_score(np.r_[np.ones(len(u)), np.zeros(len(dec))], np.r_[u, dec])
            above = (dec > u.min()).mean()
            entry |= {"auc_unpaced_vs_test_decode": round(auc, 3),
                      "frac_test_decode_above_unpaced_min": round(above, 4)}
        annotated[m] = entry
    order, heads = rows(list(r["scores"][PANELS[0][0]]))
    first.set_yticks([y for _, y in order], [label_colour(k)[0] for k, _ in order])
    first.tick_params(labelleft=True)
    for name, y in heads:
        first.text(-0.02, y - 0.75, name, transform=first.get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=ps.MATH_PT, fontweight="bold", color=ps.INK)
    # Panel letters on the titles' baseline: a at the figure's left edge, b just
    # left of its axes.
    fig.canvas.draw()
    for letter, ax in heads_ax.items():
        x = (fig.transFigure + ax.transAxes.inverted()).transform((0.004, 0))[0] \
            if letter == "a" else -0.05
        ax.text(x, HEAD_Y, letter, transform=ax.transAxes, ha="left" if letter == "a" else "right",
                va="baseline", fontsize=8, fontweight="bold", color=ps.INK)
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT, ps.PAGE)
    OUT.with_suffix(".json").write_text(json.dumps(
        {"generated_by": "scripts/plot_f7_r1.py",
         "source": str(SRC.relative_to(REPO)),
         "source_repo_commit": r["repo_commit"],
         "annotated": annotated}, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
