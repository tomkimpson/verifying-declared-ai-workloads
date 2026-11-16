#!/usr/bin/env python3
"""Task 5.3: F4, episode detection and delay per (duration, duty) cell
(spec §6), from results/monitor/episodes.json.

(a) Each cell's headline detection (P with pre-alarmed credit under the gap
guard) with its within-run lower bound, and the onset-only rate as an open
marker, per detector, with the pre-registered level. The complete-step
restricted rate equals the headline in every cell, so it is not drawn. (b)
Delay to the window-end onset per episode that raises a new alarm, with marker
area proportional to the number of episodes at that delay, and the cell median.
Above (b), per detector, the count of episodes that start while an alarm is
already active. Writes figures/f4_episodes.pdf and figures/f4_episodes.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import plotstyle as ps  # noqa: E402
from plotstyle import plt  # noqa: E402

SRC = REPO / "results" / "monitor" / "episodes.json"
OUT = REPO / "figures" / "f4_episodes.pdf"
DETS = ("r1", "classifier")
#: Horizontal offset of each detector within a cell.
OFFSET = {"r1": -0.17, "classifier": 0.17}
#: Delays are pooled to this resolution (s) before counting, for marker area.
DELAY_RES = 0.1
#: Marker area (pt^2) in (b): BASE plus PER per episode at that delay.
BASE, PER = 3.0, 1.2
#: Episode counts shown in (b)'s size key.
KEY_N = (1, 10, 40)


def main() -> None:
    ps.apply()
    e = json.loads(SRC.read_text())
    cells = e["detectors"]["r1"]["cells"]
    order = sorted(cells, key=lambda r: (cells[r]["nominal_s"], cells[r]["duty"]))
    x = np.arange(len(order))
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(ps.HALF, 3.5), sharex=True,
                                  gridspec_kw={"hspace": 0.62, "height_ratios": [1, 1.1]})
    level = e["rule"]["lmin_level"]
    ax.axhline(level, color=ps.MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
    ax.text(len(order) - 0.5, level - 0.03, "pre-registered level", ha="right", va="top",
            fontsize=6, color=ps.MUTED)
    for k in DETS:
        c = e["detectors"][k]["cells"]
        col, mk, off = ps.DETECTOR[k], ps.DETECTOR_MARKER[k], OFFSET[k]
        rate = np.array([c[r]["headline"]["rate"] for r in order])
        lb = np.array([c[r]["headline"]["lower"] for r in order])
        ax.errorbar(x + off, rate, yerr=[rate - lb, np.zeros(len(order))], fmt=mk, color=col,
                    ms=3.2, lw=0.8, capsize=1.5, capthick=0.8, zorder=3,
                    label=ps.DETECTOR_NAME[k])
        # Onset-only rate, drawn only where it differs from the headline.
        oo = np.array([c[r]["onset_only"]["rate"] for r in order])
        diff = ~np.isclose(oo, rate)
        ax.scatter((x + off)[diff], oo[diff], marker=mk, s=11, facecolor="white",
                   edgecolor=col, lw=0.8, zorder=4)
        for i, r in enumerate(order):
            d = Counter(round(ep["delay_s"] / DELAY_RES) * DELAY_RES
                        for ep in c[r]["episodes"] if ep["class"] == "detected")
            if d:
                ys, n = zip(*sorted(d.items()))
                axd.scatter(np.full(len(ys), i + off), ys, s=BASE + PER * np.array(n),
                            marker=mk, color=col, alpha=0.75, lw=0.4, edgecolor="white",
                            zorder=3)
            if c[r]["delay"]["median_s"] is not None:
                axd.plot([i + off - 0.13, i + off + 0.13], [c[r]["delay"]["median_s"]] * 2,
                         color=col, lw=1.2, zorder=4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylabel("episode detection")
    h, lab = ax.get_legend_handles_labels()
    h.append(ax.scatter([], [], marker="o", s=11, facecolor="white", edgecolor=ps.MUTED,
                        lw=0.8))
    lab.append("onset only")
    ax.legend(h, lab, loc="lower left", ncol=3, columnspacing=0.9, handletextpad=0.2,
              bbox_to_anchor=(0.0, 0.03), fontsize=ps.MATH_PT)
    ps.panel(ax, "a", x=-0.13, y=1.0)

    # Episodes already in alarm at their start (no delay): one row per detector.
    for j, k in enumerate(DETS):
        yrow = 1.03 + 0.16 * (1 - j)
        for i, r in enumerate(order):
            n = e["detectors"][k]["cells"][r]["counts"]["pre-alarmed"]
            axd.text(i, yrow, str(n), ha="center", va="bottom", fontsize=6,
                     color=ps.DETECTOR[k], transform=axd.get_xaxis_transform())
        axd.text(-0.6, yrow, ps.DETECTOR_NAME[k], ha="right", va="bottom", fontsize=ps.MATH_PT,
                 color=ps.DETECTOR[k], transform=axd.get_xaxis_transform())
    axd.text(len(order) / 2 - 0.5, 1.03 + 0.32, "episodes starting already in alarm",
             ha="center", va="bottom", fontsize=6, color=ps.MUTED,
             transform=axd.get_xaxis_transform())
    for j, n in enumerate(KEY_N):
        yk = 5.6 - 0.75 * j
        axd.scatter([len(order) - 1.25], [yk], s=BASE + PER * n, color=ps.MUTED, lw=0)
        axd.text(len(order) - 1.0, yk, f"{n} episode{'s' if n > 1 else ''}", ha="left",
                 va="center", fontsize=6, color=ps.MUTED)
    axd.set_ylim(0, 8)
    axd.set_yticks([0, 2, 4, 6, 8])
    axd.set_ylabel("delay to alarm (s)")
    ps.panel(axd, "b", x=-0.13, y=1.35)
    labels = [f"{cells[r]['nominal_s']:g}\n{cells[r]['duty']:g}" for r in order]
    axd.set_xticks(x, labels)
    axd.set_xlim(-0.6, len(order) - 0.4)
    axd.tick_params(axis="x", length=0)
    axd.text(-0.62, -0.03, "duration (s)\nduty", ha="right", va="top", fontsize=6,
             color=ps.MUTED, transform=axd.get_xaxis_transform())
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT, ps.HALF)
    OUT.with_suffix(".json").write_text(json.dumps(
        {"generated_by": "scripts/plot_f4_episodes.py", "source": str(SRC.relative_to(REPO)),
         "source_repo_commit": e["repo_commit"]}, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
