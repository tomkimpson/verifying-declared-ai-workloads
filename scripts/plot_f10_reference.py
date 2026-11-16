#!/usr/bin/env python3
"""F10: the reference of the signature check, as a schematic (amendment 2026-11-16b).

No data. Four rows, one per way the reference can be built, each a cartoon of
the score axis of s_R: declared work, hidden training, and the threshold
tau_alpha set on the threshold set. Where the work unlike the fit sits decides
whether the check works, fails loudly or fails silently (T7, F9 hold the
measured numbers).

Writes figures/f10_reference.pdf and figures/f10_reference.json.
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

OUT = REPO / "figures" / "f10_reference"
HEIGHT = 3.3
DECL = ps.SERIES["declared inference"]
TRAIN = ps.SERIES["honest training"]
X = np.linspace(0, 1, 400)
#: Rows: (title, verdict, bumps, tau, shift). A bump is (centre, label, colour,
#: style): "solid", "hatched" (declared work missing from the fit) or "ghost"
#: (where training would score with a clean reference). shift is
#: (x0, x1, label) or None: a move the reference causes, drawn as an arrow; a
#: shift of tau also draws the clean tau faintly at x0.
ROWS = (
    ("Complete reference", "works",
     [(0.30, "declared", DECL, "solid"), (0.76, "training", TRAIN, "solid")], 0.53, None),
    ("Declared workload missing from both parts", "loud",
     [(0.30, "declared", DECL, "solid"), (0.66, "missing workload", DECL, "hatched"),
      (0.88, "training", TRAIN, "solid")], 0.48, None),
    ("Declared workload in the threshold set only", "silent",
     [(0.25, "in fit", DECL, "solid"), (0.70, "training", TRAIN, "solid"),
      (0.88, "not in fit", DECL, "hatched")], 0.80, (0.48, 0.79, r"$\tau_\alpha$ rises")),
    ("Training in the reference (here, in the fit)", "silent",
     [(0.30, "declared", DECL, "solid"), (0.50, "training", TRAIN, "solid"),
      (0.82, "", TRAIN, "ghost")], 0.62, (0.80, 0.53, "training scores fall")),
)
VERDICT_COLOUR = {"works": ps.INK, "loud": "#b3261e", "silent": "#b3261e"}
WIDTH = 0.055


def bump(c: float) -> np.ndarray:
    return np.exp(-0.5 * ((X - c) / WIDTH) ** 2)


def main() -> None:
    ps.apply()
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(ps.COL, HEIGHT), sharex=True,
                             gridspec_kw={"hspace": 1.25})
    for ax, (title, verdict, bumps, tau, shift), letter in zip(axes, ROWS, "abcd"):
        for c, label, colour, style in bumps:
            y = bump(c)
            if style == "ghost":
                ax.plot(X, y, color=colour, lw=0.6, ls=":", alpha=0.8)
            else:
                ax.fill_between(X, 0, y, color=colour, lw=0,
                                alpha=0.15 if style == "hatched" else 0.35,
                                hatch="//////" if style == "hatched" else None, edgecolor=colour)
                ax.plot(X, y, color=colour, lw=0.8, ls="--" if style == "hatched" else "-")
            ax.text(c, 1.06, label, ha="center", va="bottom", fontsize=6, color=colour)
        ax.plot([tau, tau], [0, 1.3], color=ps.INK, lw=0.8, ls=":", clip_on=False)
        ax.text(tau, -0.12, r"$\tau_\alpha$", ha="center", va="top", fontsize=ps.MATH_PT,
                color=ps.INK)
        if shift:
            x0, x1, label = shift
            if "tau" in label:
                ax.plot([x0, x0], [0, 1.3], color=ps.MUTED, lw=0.6, ls=":", alpha=0.6)
            ax.annotate("", xy=(x1, 0.55), xytext=(x0, 0.55),
                        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": ps.INK})
            ax.text((x0 + x1) / 2, 0.62, label, ha="center", va="bottom", fontsize=6,
                    color=ps.INK)
        ax.text(0, 1.55, title, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.5, fontweight="bold", color=ps.INK)
        ax.text(1, 1.55, verdict, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, fontweight="bold", color=VERDICT_COLOUR[verdict])
        ps.panel(ax, letter, x=-0.06, y=1.55)
        ax.set_ylim(0, 1.3)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel(r"score $s_R$ (more unlike the fit $\rightarrow$)",
                        fontsize=ps.MATH_PT, labelpad=9)
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT.with_suffix(".pdf"), ps.COL)
    OUT.with_suffix(".json").write_text(json.dumps(
        {"generated_by": "scripts/plot_f10_reference.py", "schematic": True,
         "data": None, "rows": [{"title": r[0], "verdict": r[1]} for r in ROWS]},
        indent=1) + "\n")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO)}")


if __name__ == "__main__":
    main()
