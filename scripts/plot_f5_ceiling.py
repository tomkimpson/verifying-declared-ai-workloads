#!/usr/bin/env python3
"""Task 6.1: F5, the ceiling as tested (spec §6), from results/ceiling/s2.json.

Three panels sharing one row per run: genuine training, discarded-update
training (primary), the copying control (secondary) and the time-shared 0.5
positive control. (a) The TOST, per detector, of genuine training's per-window
flag rate minus each other run's: the difference with its 90% interval, against
the +/- delta band (genuine is the reference row). (b) and (c) Per-window scores
of each run under s_R and the classifier, descriptive, with each detector's
frozen threshold, and each run's flag rate beside its strip. The verdicts go to
the caption. Writes figures/f5_ceiling.pdf and figures/f5_ceiling.json (the
source's repo_commit).
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

SRC = REPO / "results" / "ceiling" / "s2.json"
OUT = REPO / "figures" / "f5_ceiling.pdf"
DETS = ("r1", "classifier")
#: Rows, top to bottom: (run key, row label, TOST key comparing genuine with it).
ROWS = (("genuine", "genuine", None),
        ("discarded", "discarded updates", "primary"),
        ("copying", "copying control", "secondary"),
        ("control", "time-shared 0.5\n(positive control)", "positive_control"))
#: Vertical offset of each detector within a row of (a).
OFFSET = {"r1": -0.13, "classifier": 0.13}
#: Score-axis data range per detector, and the x position of the flag-rate column.
XRANGE = {"r1": (0.48, 0.71), "classifier": (0, 1)}
JITTER, SEED = 0.22, 0


def main() -> None:
    ps.apply()
    s2 = json.loads(SRC.read_text())
    delta = s2["tost"]["delta"]
    fig, axes = plt.subplots(1, 3, figsize=(ps.PAGE, 1.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.16})
    y = np.arange(len(ROWS))

    ax = axes[0]
    # The band and zero line stop short of the axes' top, below the +/-delta label.
    span = (-0.6, len(ROWS) - 0.4)
    ax.fill_betweenx(span, -delta, delta, color=ps.GRID, lw=0, zorder=0)
    ax.text(0, -0.75, rf"$\pm\delta$", ha="center", va="bottom", fontsize=ps.MATH_PT,
            color=ps.MUTED)
    ax.plot([0, 0], span, color=ps.MUTED, lw=0.5, zorder=1)
    for det in DETS:
        t = s2["tests"][det]
        rows = [(i, t[k]) for i, (_, _, k) in enumerate(ROWS) if k]
        yy = np.array([i for i, _ in rows]) + OFFSET[det]
        diff = np.array([r["difference"] for _, r in rows])
        ci = np.array([r["ci90"] for _, r in rows]).T
        ax.errorbar(diff, yy, xerr=[diff - ci[0], ci[1] - diff], fmt=ps.DETECTOR_MARKER[det],
                    ms=3.2, color=ps.DETECTOR[det], lw=0.8, capsize=1.5, capthick=0.8,
                    label=ps.DETECTOR_NAME[det], zorder=3)
    ax.text(0.07, 0, "reference", ha="left", va="center", fontsize=6, color=ps.MUTED)
    ax.set_xlim(-0.1, 0.45)
    ax.set_xticks([0, 0.2, 0.4], ["0", "0.2", "0.4"])
    ax.set_xlabel("flag rate: genuine minus this run")
    ax.xaxis.set_label_coords(0.5, -0.17)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), handletextpad=0.2,
              borderaxespad=0.1, fontsize=ps.MATH_PT)
    ax.set_title("equivalence test, 90% intervals", loc="left", pad=12)
    ps.panel(ax, "a", x=-0.03, y=1.08)

    rng = np.random.default_rng(SEED)
    for ax, det, letter in zip(axes[1:], DETS, "bc"):
        lo, hi = XRANGE[det]
        col_x = hi + 0.2 * (hi - lo)
        tau = s2["thresholds"][det]
        ax.axvline(tau, color=ps.INK, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
        ax.text(tau, -0.75, r"$\tau_\alpha$", ha="center", va="bottom", fontsize=ps.MATH_PT,
                color=ps.INK)
        ax.text(col_x, -0.75, "% above", ha="right", va="bottom", fontsize=6, color=ps.MUTED)
        for i, (arm, _, _) in enumerate(ROWS):
            d = s2["runs"][arm]["detectors"][det]
            sc = np.array(d["scores"])
            ax.scatter(sc, i + rng.uniform(-JITTER, JITTER, len(sc)), s=2.5,
                       color=ps.DETECTOR[det], lw=0, alpha=0.5, rasterized=True, zorder=2)
            ax.text(col_x, i, ps.percent(d["flag_rate"]), ha="right", va="center", fontsize=6,
                    color=ps.INK)
        pad = 0.04 * (hi - lo)
        ax.set_xlim(lo - pad, col_x + pad)
        ax.spines["bottom"].set_bounds(lo, hi)
        ticks = np.linspace(lo, hi, 3) if det == "classifier" else [0.5, 0.6, 0.7]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.set_xlabel(f"score {'$s_R$' if det == 'r1' else '$s_C$'}")
        ax.xaxis.set_label_coords(((lo + hi) / 2 - (lo - pad)) / (col_x - lo + 2 * pad), -0.17)
        ax.set_title(f"{ps.DETECTOR_NAME[det]} scores", loc="left", pad=12)
        ps.panel(ax, letter, x=-0.03, y=1.08)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y, [lab for _, lab, _ in ROWS])
    axes[0].set_ylim(len(ROWS) - 0.4, -0.85)
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT, ps.PAGE)
    OUT.with_suffix(".json").write_text(json.dumps(
        {"generated_by": "scripts/plot_f5_ceiling.py", "source": str(SRC.relative_to(REPO)),
         "source_repo_commit": s2["repo_commit"]}, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
