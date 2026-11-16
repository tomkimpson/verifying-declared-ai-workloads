#!/usr/bin/env python3
"""F9: training in the reference blinds s_R without raising alarms on declared
work (amendment 2026-11-16b; exploratory).

(a) Training in the threshold set (E3, alpha = 0.05): the fit is fixed and k
development honest-training windows join the calibration declared windows.
Percent flagged against the share of the threshold set that is training: the
unthrottled co-location run, pooled development honest training (in-sample:
the added windows come from this pool), and test decode and idle (scores
already committed in r1_scores.json).

(b) Training in the fit (E4): k development honest-training windows join the
development declared windows of the fit; tau_alpha is recomputed on the
unchanged threshold set. Median percent flagged over draws (band: min to max)
against the share of the fit that is training: the unthrottled run, the
calibration-day full fine-tuning and LoRA runs (outside every fit), the
development declared windows (in-sample: they are in every fit) and the
calibration declared windows (about alpha by construction).

Reads results/heldout/e3_calibration_breakdown.json and
results/heldout/e4_fit_contamination.json. Writes figures/f9_contamination.pdf
and figures/f9_contamination.json.
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

E3 = REPO / "results" / "heldout" / "e3_calibration_breakdown.json"
E4 = REPO / "results" / "heldout" / "e4_fit_contamination.json"
OUT = REPO / "figures" / "f9_contamination"
HEIGHT = 3.0
UNPACED = "panel-mixture-concurrent-0p5"
FULL_FT, LORA = "panel-core-training-r4", "panel-core-lora-r4"
DECL = ps.SERIES["declared inference"]
TRAIN = ps.SERIES["honest training"]
#: E3 series: (rates key, label, colour, dash).
E3_SERIES = (
    (f"co-location {UNPACED}", "unthrottled co-location", ps.SERIES[UNPACED], "-"),
    ("honest training", "honest training (dev, in-sample)", TRAIN, "-"),
    ("declared test serving-decode", "declared decode (test)", DECL, "-"),
    ("declared test idle", "declared idle (test)", DECL, "--"),
)


def e4_series(g: list[dict]) -> list[tuple[str, str, str, list[dict]]]:
    """(label, colour, dash, per-k summaries) for panel b."""
    return [
        ("unthrottled co-location", ps.SERIES[UNPACED], "-", [x["unpaced_detection"] for x in g]),
        ("full fine-tuning (cal)", TRAIN, "-", [x["calibration_honest_by_run"][FULL_FT] for x in g]),
        ("LoRA (cal)", TRAIN, "--", [x["calibration_honest_by_run"][LORA] for x in g]),
        ("declared (dev, in fit)", DECL, "-", [x["declared_fit_flag_rate"] for x in g]),
        ("declared (cal, sets $\\tau_\\alpha$)", DECL, "--",
         [x["calibration_declared_flag_rate"] for x in g]),
    ]


def main() -> None:
    ps.apply()
    e3, e4 = json.loads(E3.read_text()), json.loads(E4.read_text())
    if "declared_fit_flag_rate" not in e4["grid"][0]:
        raise SystemExit("e4_fit_contamination.json predates amendment 2026-11-16b: rerun "
                         "scripts/e4_fit_contamination.py")
    fig, (a, b) = plt.subplots(2, 1, figsize=(ps.COL, HEIGHT), gridspec_kw={"hspace": 0.75})
    meta = {"generated_by": "scripts/plot_f9_contamination.py",
            "sources": {"e3": {"path": str(E3.relative_to(REPO)), "repo_commit": e3["repo_commit"]},
                        "e4": {"path": str(E4.relative_to(REPO)), "repo_commit": e4["repo_commit"]}},
            "alpha": e4["alpha"], "drawn": {}}

    # (a) Training in the threshold set.
    curve = e3["curve_alpha_0.05"]
    n_cal = e3["calibration_windows"]
    pts = sorted(curve.values(), key=lambda v: v["n_windows"])
    share = np.array([100 * v["n_windows"] / (v["n_windows"] + n_cal) for v in pts])
    meta["drawn"]["a"] = {"share_pct": share.round(2).tolist(), "n_calibration": n_cal}
    for key, label, colour, ls in E3_SERIES:
        y = np.array([100 * v["rates"][key] for v in pts])
        a.plot(share, y, color=colour, ls=ls, lw=0.9, marker="o", ms=1.8, label=label)
        meta["drawn"]["a"][label] = y.round(2).tolist()
    a.axvline(100 * e4["alpha"], color=ps.MUTED, lw=0.6, ls=":")
    a.text(100 * e4["alpha"] * 1.08, 25, r"$\alpha$", fontsize=ps.MATH_PT, color=ps.MUTED,
           va="center")
    a.set_xscale("symlog", linthresh=1, linscale=0.5)
    a.set_xlim(0, 60)
    a.set_xticks([0, 1, 10, 50], ["0", "1", "10", "50"])
    a.set_xlabel("training share of the threshold set (%)")
    a.set_title("Training in the threshold set", fontsize=7, loc="left", pad=3)

    # (b) Training in the fit.
    g = e4["grid"]
    x = np.array([100 * v["share_of_fit"] for v in g])
    meta["drawn"]["b"] = {"share_pct": x.round(3).tolist(), "k": [v["k"] for v in g]}
    for label, colour, ls, sm in e4_series(g):
        med = np.array([100 * s["median"] for s in sm])
        lo, hi = (np.array([100 * s[q] for s in sm]) for q in ("min", "max"))
        b.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0)
        b.plot(x, med, color=colour, ls=ls, lw=0.9, marker="o", ms=1.8, label=label)
        meta["drawn"]["b"][label] = {"median": med.round(2).tolist(),
                                     "min": lo.round(2).tolist(), "max": hi.round(2).tolist()}
    b.set_xscale("symlog", linthresh=0.1, linscale=0.4)
    b.set_xlim(0, 100)
    b.set_xticks([0, 0.1, 1, 10, 100], ["0", "0.1", "1", "10", "100"])
    b.set_xlabel("training share of the fit (%)")
    b.set_title("Training in the fit", fontsize=7, loc="left", pad=3)

    for ax, letter in ((a, "a"), (b, "b")):
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel("flagged (%)")
        ps.panel(ax, letter, x=-0.2, y=1.12)
        ax.legend(fontsize=6, frameon=False, loc="center left", handlelength=1.8,
                  bbox_to_anchor=(0.0, 0.5), borderaxespad=0.2)
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT.with_suffix(".pdf"), ps.COL)
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO)}")


if __name__ == "__main__":
    main()
