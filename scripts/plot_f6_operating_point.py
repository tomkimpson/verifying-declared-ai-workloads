#!/usr/bin/env python3
"""Tasks 8.4, 10.1: F6, the operating point (spec §6), from
results/heldout/operating_point.json.

Three panels, one per classifier: s_C^dev (co-location held out), the
no-evasion fit s_C^ne and the replica s_C^rep. Each panel has one box-plot row
(plotstyle.boxrow) of per-window score for each declared workload and split and
for each co-location run's training windows, with tau_d = 0.5 and the
detector's calibration-day tau_alpha marked. Most declared windows score exactly
zero and each co-location run spans a narrow range, so the windows beyond the
whiskers, drawn one by one, carry the comparison. The fraction of each row's
windows above tau_d and above tau_alpha, and the unthrottled run's (`-0p5`) ROC
AUCs against test declared windows and against test decode alone, go to the
sidecar and the caption, not the figure. Writes figures/f6_operating_point.pdf and
figures/f6_operating_point.json, which records every annotated number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import plotstyle as ps  # noqa: E402
from plotstyle import plt  # noqa: E402

SRC = REPO / "results" / "heldout" / "operating_point.json"
OUT = REPO / "figures" / "f6_operating_point.pdf"
UNPACED = "panel-mixture-concurrent-0p5"
#: Panels, left to right: (key in operating_point.json, letter, title).
PANELS = (("classifier", "a", r"$s_C^{\mathrm{dev}}$, co-location held out"),
          ("no_evasion", "b", r"$s_C^{\mathrm{ne}}$, no evasion examples"),
          ("replica", "c", r"$s_C^{\mathrm{rep}}$, replica"))
SPLIT = {"development declared": "dev", "test declared": "test"}
TAU_D = 0.5
#: Text holding sub- or superscripts: mathtext sets scripts at 0.7 of this, so
#: they print at 5.0 pt (ps.MATH_PT gives 4.9).
SCRIPT_PT = 7.2
#: Height (axes fraction) of the panel letters and titles, one line for all panels.
HEAD_Y = 1.13
#: Gap (in rows) between the declared strips and the co-location strips; each
#: group's header sits 0.75 rows above its first strip, so it reads with that group.
GROUP_GAP = 1.3


def groups(scores: dict) -> list[tuple[str, np.ndarray, str, float]]:
    """(label, scores, colour, y) per strip: declared workloads, then co-location runs."""
    out, y = [], 0.0
    for key in ("development declared", "test declared"):
        for w, s in sorted(scores[key].items()):
            out.append((f"{w.replace('serving-', '')} ({SPLIT[key]})", np.array(s),
                        ps.SERIES["declared inference"], y))
            y += 1
    y += GROUP_GAP
    for run, s in sorted(scores["co-location"].items()):
        out.append((ps.label(run).replace("co-location, ", ""), np.array(s), ps.SERIES[run], y))
        y += 1
    return out


def main() -> None:
    ps.apply()
    op = json.loads(SRC.read_text())
    fig, axes = plt.subplots(1, len(PANELS), figsize=(ps.PAGE, 2.0), sharey=True,
                             gridspec_kw={"wspace": 0.1})
    annotated = {}
    for ax, (k, letter, title) in zip(axes, PANELS):
        tau_a = op[k]["tau_alpha"]["tau"]
        auc = op[k]["roc_auc_per_run_vs_test_declared"][UNPACED]
        auc_dec = op[k]["roc_auc_per_run_vs_test_decode"][UNPACED]
        g = groups(op["scores"][k])
        ps.highlight_row(ax, next(y for lab, _, _, y in g if "unthrottled" in lab))
        ax.axvline(TAU_D, color=ps.INK, lw=0.7, ls=(0, (4, 2)), zorder=0)
        ax.axvline(tau_a, color=ps.INK, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
        pct = {}
        for lab, s, c, y in g:
            ps.boxrow(ax, s, y, c)
            pct[lab] = [ps.percent((s > TAU_D).mean()), ps.percent((s > tau_a).mean())]
        top = -0.75
        for x, name, ha in ((TAU_D, rf"$\tau_{{\mathrm{{d}}}}$ = {TAU_D:g}", "center"),
                            (tau_a, rf"$\tau_\alpha$ = {tau_a:g}", "left")):
            ax.text(x, top, name, ha=ha, va="bottom", fontsize=SCRIPT_PT, color=ps.INK)
        # The run the figure is about, judged at both thresholds.
        # The unthrottled run, judged at both thresholds by most of its windows, in
        # words, with an arrow from the empty space above it.
        y_un, s_un = next((y, s) for lab, s, _, y in g if "unthrottled" in lab)
        verdict = ["flagged" if (s_un > t).mean() > 0.5 else "missed" for t in (TAU_D, tau_a)]
        ax.annotate(rf"{verdict[0]} at $\tau_{{\mathrm{{d}}}}$," "\n"
                    rf"{verdict[1]} at $\tau_\alpha$",
                    xy=(np.median(s_un), y_un - 0.35), xytext=(0.07, g[4][3] + 0.95),
                    ha="left", va="center", fontsize=SCRIPT_PT, color=ps.INK,
                    linespacing=1.1, arrowprops={"arrowstyle": "-|>,head_length=0.3,head_width=0.15",
                                                 "color": ps.INK, "lw": 0.6,
                                                 "shrinkA": 1, "shrinkB": 0})
        ax.set_xlim(-0.04, 1.04)
        ax.spines["bottom"].set_bounds(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
        ax.set_xlabel(r"score $s_C$", fontsize=SCRIPT_PT)
        ax.set_ylim(g[-1][3] + 0.6, top - 0.1)
        ax.text(0, HEAD_Y, title, transform=ax.transAxes, ha="left", va="baseline",
                fontsize=SCRIPT_PT, color=ps.INK)
        annotated[k] = {"tau_d": TAU_D, "tau_alpha": tau_a,
                        "auc_unpaced_vs_test_declared": round(auc, 3),
                        "auc_unpaced_vs_test_decode": round(auc_dec, 3),
                        "percent_above_tau_d_tau_alpha": pct}
    g = groups(op["scores"][PANELS[0][0]])
    axes[0].set_yticks([x[3] for x in g], [x[0] for x in g])
    # The result the figure exists for: the unthrottled run's miss at tau_d.
    assert max(op["scores"]["classifier"]["co-location"][UNPACED]) <= TAU_D
    # Panel letters on the titles' baseline: a at the figure's left edge, b and c
    # just left of their axes.
    fig.canvas.draw()
    for ax, (_, letter, _) in zip(axes, PANELS):
        x = (fig.transFigure + ax.transAxes.inverted()).transform((0.004, 0))[0] \
            if letter == "a" else -0.05
        ax.text(x, HEAD_Y, letter, transform=ax.transAxes, ha="left" if letter == "a" else "right",
                va="baseline", fontsize=8, fontweight="bold", color=ps.INK)
    first_co = next(x[3] for x in g if x[0].startswith(r"$\rho$"))
    for y, name in ((g[0][3], "declared"), (first_co, "co-location")):
        axes[0].text(-0.02, y - 0.75, name, transform=axes[0].get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=ps.MATH_PT, fontweight="bold", color=ps.INK)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT, ps.PAGE)
    OUT.with_suffix(".json").write_text(json.dumps(
        {"generated_by": "scripts/plot_f6_operating_point.py",
         "source": str(SRC.relative_to(REPO)),
         "source_repo_commit": op["repo_commit"],
         "annotated": annotated}, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
