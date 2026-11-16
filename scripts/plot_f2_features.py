#!/usr/bin/env python3
"""Task 4.3: F2, the level-2 forest's top stage-2 features per window set
(spec §6), from results/coverage/mechanism.json. One panel per feature, in
order of stage-two importance, and one row per window set: a thick bar spans
the 25-75% quantiles, a thin line the 5-95%, and the dot is the median. Writes
figures/f2_features.pdf and figures/f2_features.json (the source's
repo_commit, so a stale figure shows, and the importances).
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

SRC = REPO / "results" / "coverage" / "mechanism.json"
OUT = REPO / "figures" / "f2_features.pdf"
#: Panel title per feature: what it measures, in words. The code names are in
#: the caption and in the feature table.
NAMES = {
    "env_frac_150_250hz": "envelope:\nvariance fraction, 150–250 Hz",
    "env_p99_over_p50": "envelope:\n99th / 50th percentile",
    "env_cv": "envelope:\ncoefficient of variation",
    "env_ac_lag1": "envelope:\nlag-1 autocorrelation",
}


def main() -> None:
    ps.apply()
    m = json.loads(SRC.read_text())
    top = m["top_stage2_features"]
    imp = m["importances"]["stage2"]
    sets = list(ps.SERIES)
    y = np.arange(len(sets))
    fig, axes = plt.subplots(1, len(top), figsize=(ps.PAGE, 1.85), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, f, letter in zip(axes, top, "abcd"):
        for i, s in enumerate(sets):
            v = np.array(m["distributions"][s]["features"][f]["values"])
            p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
            c = ps.SERIES[s]
            ax.plot([p5, p95], [i, i], color=c, lw=0.8, solid_capstyle="butt")
            ax.plot([p25, p75], [i, i], color=c, lw=3.2, solid_capstyle="butt", alpha=0.55)
            ax.scatter([p50], [i], s=14, color=c, edgecolor="white", lw=0.6, zorder=3)
        ax.set_title(NAMES.get(f, f), loc="left", fontsize=6.5, pad=4)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.locator_params(axis="x", nbins=4)
        ps.panel(ax, letter, x=-0.04, y=1.02)
    axes[0].set_yticks(y, [ps.label(s) for s in sets])
    axes[0].set_ylim(len(sets) - 0.5, -0.6)
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT, ps.PAGE)
    OUT.with_suffix(".json").write_text(json.dumps(
        {"generated_by": "scripts/plot_f2_features.py",
         "source": str(SRC.relative_to(REPO)), "source_repo_commit": m["repo_commit"],
         "features": top, "importances": {f: round(imp[f], 2) for f in top}},
        indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
