#!/usr/bin/env python3
"""Task 5.1: F3, s_R against the classifier under the shared calibration-day
threshold rule (spec §6), from results/heldout/h2.json.

(a) Per-window detection, with its 95% interval, per hidden-training
configuration (each co-location run; the other groups pooled; time-sharing dev
and test apart), one row per configuration and one marker per detector. (b)
The descriptive test ROC for co-location, on a symmetric-log false-positive
axis (linear below 1e-3). Each detector's achieved test false-positive rates
go to the caption. Writes figures/f3_h2.pdf and figures/f3_h2.json (the
source's repo_commit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.transforms import blended_transform_factory

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import plotstyle as ps  # noqa: E402
from plotstyle import plt  # noqa: E402

SRC = REPO / "results" / "heldout" / "h2.json"
OUT = REPO / "figures" / "f3_h2.pdf"
SPLIT = {"development": "dev", "test": "test"}
#: Vertical offset of each detector's marker within a row.
OFFSET = {"r1": -0.17, "classifier": 0.17}
#: Linear region of the ROC's symmetric-log false-positive axis.
LINTHRESH = 1e-3
#: Left end of panel (a)'s detection axis: every interval lies above it.
XMIN = 0.8


def configurations(h: dict) -> list[tuple[str, dict, dict]]:
    """(label, s_R summary, classifier summary) per row; every label names its
    split (spec task 5.1)."""
    r1, clf = h["r1"]["detection"], h["classifier"]
    out = []
    for run, e in sorted(r1["co-location"]["all"]["per_run"].items()):
        out.append((f"{ps.label(run)} ({SPLIT[e['split']]})", e,
                    clf["co-location"]["detection"]["all"]["per_run"][run]))
    for split in ("development", "test"):
        out.append((f"time-sharing ({SPLIT[split]})", r1["time-sharing"][split]["pooled"],
                    clf["time-sharing"]["detection"][split]["pooled"]))
    for g in ("jitter", "padding", "power cap"):
        splits = sorted({e["split"] for e in r1[g]["all"]["per_run"].values()})
        out.append((f"{g} ({'/'.join(SPLIT[s] for s in splits)})",
                    r1[g]["all"]["pooled"], clf[g]["detection"]["all"]["pooled"]))
    return out


def main() -> None:
    ps.apply()
    h = json.loads(SRC.read_text())
    conf = configurations(h)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(ps.HALF, 3.9),
                                  gridspec_kw={"height_ratios": [2.3, 1.4], "hspace": 0.45})
    y = np.arange(len(conf))
    for j, k in enumerate(("r1", "classifier")):
        det = np.array([c[1 + j]["detection"] for c in conf])
        ci = np.array([c[1 + j]["detection_ci95"] for c in conf]).T
        ax.errorbar(det, y + OFFSET[k], xerr=[det - ci[0], ci[1] - det],
                    fmt=ps.DETECTOR_MARKER[k], color=ps.DETECTOR[k], ms=3.2, lw=0.8,
                    capsize=1.5, capthick=0.8, label=ps.DETECTOR_NAME[k])
    ax.set_yticks(y, [c[0] for c in conf])
    ax.set_ylim(len(conf) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(XMIN, 1.006)
    ax.spines["bottom"].set_bounds(XMIN, 1)
    ax.set_xticks(np.linspace(XMIN, 1, 5), [f"{v:g}" for v in np.linspace(XMIN, 1, 5)])
    ax.set_xlabel(r"detection at $\tau_\alpha$")
    ax.grid(axis="x")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.3,
              borderaxespad=0.2, fontsize=ps.MATH_PT)
    assert min(min(c[1]["detection_ci95"][0], c[2]["detection_ci95"][0]) for c in conf) > XMIN

    # The classifier's curve (AUC 1) runs along the left and top edges, over
    # s_R's; draw it last and dashed so s_R shows through the gaps.
    for k in ("r1", "classifier"):
        r = h["roc"][k]
        axr.plot(r["fpr"], r["tpr"], color=ps.DETECTOR[k], drawstyle="steps-post",
                 ls="-" if k == "r1" else (0, (3, 1.5)), lw=1.3 if k == "r1" else 1.1,
                 zorder=2 if k == "r1" else 3)
    axr.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.4)
    axr.set_xlim(0, 1)
    axr.set_xticks([0, 1e-3, 1e-2, 1e-1, 1], ["0", r"$10^{-3}$", r"$10^{-2}$",
                                               r"$10^{-1}$", "1"])
    axr.set_ylim(0, 1.04)
    axr.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    axr.set_xlabel("false-positive rate on test declared windows")
    axr.set_ylabel("co-location detection")
    axr.grid(True)
    # Mark the change from linear to logarithmic scale on the x axis.
    axr.plot([LINTHRESH], [0], marker=(2, 0, -60), ms=6, mew=0.8, color=ps.MUTED,
             clip_on=False, transform=axr.get_xaxis_transform(), zorder=5)
    axr.text(6e-4, 0.6, ps.DETECTOR_NAME["r1"], fontsize=ps.MATH_PT,
             color=ps.DETECTOR["r1"], ha="left", va="top")
    axr.text(4e-3, 0.95, ps.DETECTOR_NAME["classifier"], fontsize=ps.MATH_PT,
             color=ps.DETECTOR["classifier"], ha="left", va="top")
    # Panel (b) has short tick labels: widen it to start under (a)'s row labels.
    fig.canvas.draw()
    left = ax.get_tightbbox().transformed(fig.transFigure.inverted()).x0 + 0.1
    pa, pb = ax.get_position(), axr.get_position()
    axr.set_position([left, pb.y0, pa.x1 - left, pb.height])
    # Both panel letters at the left edge of the row labels, on one vertical.
    for a, letter in ((ax, "a"), (axr, "b")):
        a.text(left - 0.1, 1.01, letter, ha="left", va="bottom", fontsize=8, fontweight="bold",
               color=ps.INK, transform=blended_transform_factory(fig.transFigure, a.transAxes))
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT, ps.HALF)
    OUT.with_suffix(".json").write_text(json.dumps(
        {"generated_by": "scripts/plot_f3_h2.py", "source": str(SRC.relative_to(REPO)),
         "source_repo_commit": h["repo_commit"]}, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
