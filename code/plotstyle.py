"""Shared figure style. Categorical slots in fixed order (validated with the
dataviz palette checker, light mode: CVD and normal-vision floors pass; three
slots are below 3:1 contrast, so every series is named in a label).

A workload is always one of SERIES, a detector always one of DETECTOR
(matplotlib's default cycle). No figure colours both at once. The two detectors
also differ in marker shape (DETECTOR_MARKER), so they never rely on colour
alone."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {
    "declared inference": "#2a78d6",
    "honest training": "#eb6834",
    "time-shared training": "#1baf7a",
    "panel-mixture-concurrent-0p25": "#eda100",
    "panel-mixture-concurrent-0p5": "#e87ba4",
    "panel-mixture-concurrent-0p75": "#008300",
}
LABELS = {
    "panel-mixture-concurrent-0p25": r"co-location, $\rho$ = 0.25",
    "panel-mixture-concurrent-0p5": r"co-location, $\rho$ = 0.5, unthrottled",
    "panel-mixture-concurrent-0p75": r"co-location, $\rho$ = 0.75",
}
NON_AI = "#8c8a83"
#: Declaration model s_R and classifier s_C: matplotlib's C0 and C1.
DETECTOR = {"r1": "#1f77b4", "classifier": "#ff7f0e"}
DETECTOR_MARKER = {"r1": "o", "classifier": "s"}
DETECTOR_NAME = {"r1": r"$s_R$", "classifier": r"$s_C^{\mathrm{dev}}$"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3de"
#: Figure widths (inches) of one ICML column, of the full text width, and of a
#: figure set at 0.5\linewidth in the one-column appendix.
COL, PAGE = 3.25, 6.75
HALF = PAGE / 2
#: Text holding sub- or superscripts is set at this size: mathtext draws scripts
#: at 0.7 of it, so they print at 4.9 pt.
MATH_PT = 7


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.0, "legend.frameon": False,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, out: Path, width: float) -> None:
    """Save with a tight bounding box whose width, padding included, is exactly
    `width` inches, so that \\includegraphics[width=\\linewidth] prints the text at
    its set size. Tight cropping alone sets the width by whatever the labels span;
    here the figure is resized until the cropped width matches."""
    target = width - 2 * plt.rcParams["savefig.pad_inches"]
    for _ in range(8):
        fig.canvas.draw()
        w = fig.get_tightbbox().width
        if abs(w - target) < 0.005:
            break
        fig.set_size_inches(fig.get_figwidth() + (target - w), fig.get_figheight())
    fig.savefig(out, metadata={"CreationDate": None})


def label(key: str) -> str:
    """Display name of a workload or co-location run."""
    return LABELS.get(key, key)


def panel(ax, letter: str, x: float = -0.02, y: float = 1.0) -> None:
    """Bold panel letter at the axes' top-left corner (axes coordinates)."""
    ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold", color=INK)


def highlight_row(ax, y: float) -> None:
    """Grey band behind one row of a box plot: the run the figure is about."""
    ax.axhspan(y - 0.5, y + 0.5, color=GRID, alpha=0.6, lw=0, zorder=-1)


def boxrow(ax, s, y: float, color: str, whis=(1, 99), h: float = 0.32) -> None:
    """One horizontal row of a box plot at height y: the 25-75% quantiles as a
    bar, whiskers to the `whis` quantiles, the median as a tick, and every value
    beyond the whiskers as its own dot. Quantiles are taken at observed values
    ("nearest"), so a whisker never ends where no window scores."""
    import numpy as np
    q = np.percentile(s, [whis[0], 25, 50, 75, whis[1]], method="nearest")
    ax.plot([q[0], q[4]], [y, y], color=color, lw=0.8, solid_capstyle="butt", zorder=2)
    ax.plot([q[1], q[3]], [y, y], color=color, lw=4, alpha=0.6, solid_capstyle="butt",
            zorder=2)
    ax.plot([q[2]] * 2, [y - h, y + h], color=INK, lw=1.0, zorder=4)
    out = s[(s < q[0]) | (s > q[4])]
    ax.scatter(out, np.full(len(out), y), s=5, facecolor="white", edgecolor=color,
               lw=0.6, zorder=3)


def axis_break(left, right, d: float = 0.012) -> None:
    """Mark a break in a shared x axis between two side-by-side axes with a pair
    of short slashes on their bottom spines."""
    for ax, x in ((left, 1), (right, 0)):
        tr = ax.transAxes
        ax.plot([x - d, x + d], [-0.02, 0.02], transform=tr, color=MUTED, lw=0.6,
                clip_on=False)


def percent(frac: float) -> str:
    """Percent label: whole numbers, but one decimal when rounding would show
    a non-zero rate as 0 or 1 or a sub-100 rate as 100 (e.g. 1 of 170 windows),
    and "<0.1" below that."""
    p = 100 * frac
    if 0 < p < 0.05:
        return "<0.1"
    return f"{p:.1f}" if 0 < p < 1.5 or 99 < p < 100 else f"{p:.0f}"
