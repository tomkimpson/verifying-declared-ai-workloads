#!/usr/bin/env python3
"""Task 14.2: F8, three configurations of s_R over the same test-day excerpts
(amendment 2026-11-13f; descriptive).

Top row: the 2 ms RMS envelope of the AC-coupled current (amperes; current,
not power) over three separate captures of the test day: idle, declared
decode, and decode with time-shared training episodes (shaded). Rows below:
the score of each configuration per 5 s window (1 s hop) against its
tau_alpha, with the 3-of-5 alarm drawn as a bar. The loud and silent rows
share one decode-only fit, so their score traces are the same; only tau_alpha
moves. Whole-day totals at each row's right are the committed H3 and T4 counts
that scripts/monitor_traces.py reproduced, not counts over the excerpt.

A narrow last column is a positive control (amendment 2026-11-16b): each
configuration's development honest-training windows (5 s hop), scored by its
fit, against its tau_alpha, from results/heldout/r1_scores.json. The silent
configuration flags almost none, so one control run exposes it.

Reads results/monitor/traces.json and results/heldout/r1_scores.json. Writes figures/f8_monitors.pdf and
figures/f8_monitors.json (what is drawn and annotated).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import plotstyle as ps  # noqa: E402
from plotstyle import plt  # noqa: E402

SRC = REPO / "results" / "monitor" / "traces.json"
CONTROL_SRC = REPO / "results" / "heldout" / "r1_scores.json"
#: Per configuration: (model key, threshold key) in r1_scores.json.
CONTROL = {"full": ("r1", "declared"), "loud": ("decode_only", "decode calibration"),
           "silent": ("decode_only", "pooled calibration")}
#: Width of the positive-control column, in the seconds units of the time columns.
CONTROL_W = 70
CONTROL_COLOUR = ps.SERIES["honest training"]
OUT = REPO / "figures" / "f8_monitors"
HEIGHT = 3.9
#: Row heights: envelope, then one row per configuration.
ROW_RATIOS = (0.8, 1.0, 1.0, 1.0)
SCORE_LIM = (0.37, 0.87)
#: The alarm bar sits in this band of the score axis, above every score.
ALARM_BAND = (0.835, 0.862)
ALARM = "#b3261e"
EPISODE_FILL = ps.SERIES["time-shared training"]
#: A break in the score trace wherever consecutive windows are further apart
#: than this (the unscored seconds between a capture's scored intervals).
BREAK_S = 1.5
COLUMN_TITLE = {"idle": "idle", "declared decode": "declared decode",
                "episodes": "declared decode with hidden training episodes (time-shared)"}
ROWS = (("full", "b", "Complete reference"),
        ("loud", "c", "Decode-only reference"),
        ("silent", "d", "Decode-only fit, idle in the threshold set"))
#: Rows that raise no alarm on declared work: the working and the silent monitor.
QUIET = ("full", "silent")
VERDICT = {"full": "works", "loud": "fails loudly", "silent": "fails silently"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def pct(x: float) -> str:
    """A fraction as a percentage, without rounding a nonzero value to 0 or
    a value below 1 to 100."""
    p = 100 * x
    if p == 0:
        return "0%"
    if p < 0.1:
        return "<0.1%"
    if p > 99.9 and p < 100:
        return f"{p:.2f}%"
    return f"{p:.0f}%" if p >= 10 else f"{p:.1f}%"


def totals(c: dict) -> tuple[str, dict]:
    """The whole-day line of one configuration, from the committed counts."""
    ep = c["episodes"]
    scored = ep["pre-alarmed"] + ep["detected"] + ep["missed"]
    fa = c["false_alarms"]
    alarm = (f"in alarm {pct(fa['declared inference']['A'])} of decode, "
             f"{pct(fa['idle']['A'])} of idle")
    if ep["missed"] == scored:
        episodes = f"0 of {scored} episodes detected"
    elif ep["pre-alarmed"] > scored / 2:
        episodes = f"{ep['pre-alarmed']} of {scored} episodes start in alarm"
    else:
        episodes = f"{scored - ep['missed']} of {scored} episodes detected"
    return f"whole test day: {episodes}; {alarm}", {"scored_episodes": scored, **ep,
                                               "A_decode": fa["declared inference"]["A"],
                                               "A_idle": fa["idle"]["A"]}


def segments(t: np.ndarray, y: np.ndarray):
    """(t, y) pieces between breaks in the window times."""
    cuts = np.flatnonzero(np.diff(t) > BREAK_S) + 1
    return zip(np.split(t, cuts), np.split(y, cuts))


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    ps.apply()
    src = json.loads(SRC.read_text())
    ctl = json.loads(CONTROL_SRC.read_text())
    ex = src["excerpts"]
    n = len(ex)
    fig, axes = plt.subplots(
        len(ROW_RATIOS), n + 1, figsize=(ps.PAGE, HEIGHT), sharex="col", sharey="row",
        gridspec_kw={"width_ratios": [e["duration_s"] for e in ex] + [CONTROL_W],
                     "height_ratios": ROW_RATIOS, "wspace": 0.035, "hspace": 0.62})
    meta = {"generated_by": "scripts/plot_f8_monitors.py", "source": str(SRC.relative_to(REPO)),
            "source_repo_commit": src["repo_commit"], "repo_commit": git("rev-parse", "HEAD"),
            "excerpts": [{k: e[k] for k in ("name", "capture_id", "t0_s", "duration_s")}
                         for e in ex],
            "thresholds_drawn": {}, "annotated": {},
            "positive_control": {"source": str(CONTROL_SRC.relative_to(REPO)),
                                 "source_repo_commit": ctl["repo_commit"],
                                 "windows": "development honest training, 5 s window, 5 s hop"}}

    for j, e in enumerate(ex):
        # Envelope.
        ax = axes[0, j]
        env = e["envelope"]
        ax.fill_between(env["t_s"], env["lo_a"], env["hi_a"], color=ps.MUTED, alpha=0.35, lw=0)
        ax.plot(env["t_s"], env["mean_a"], color=ps.INK, lw=0.5)
        ax.set_title(COLUMN_TITLE[e["name"]], fontsize=7, pad=3,
                     loc="left" if e["name"] == "episodes" else "center")
        # Scores, thresholds and alarms.
        for i, (key, _, _) in enumerate(ROWS, start=1):
            ax = axes[i, j]
            c = e["configs"][key]
            t, s = np.array(e["window_end_s"]), np.array(c["score"])
            for tt, ss in segments(t, s):
                ax.plot(tt, ss, color=ps.DETECTOR["r1"], lw=0.7)
            tau = src["thresholds"][key]
            ax.axhline(tau, color=ps.INK, lw=0.6, ls=":")
            for a, b in c["alarms_s"]:
                ax.fill_between([a, b], *ALARM_BAND, color=ALARM, lw=0, zorder=3)
            ax.set_ylim(*SCORE_LIM)
            ax.set_yticks([0.4, 0.6, 0.8])
            meta["thresholds_drawn"][key] = tau
        for ax in axes[:, j]:
            for a, b in e["episodes"]:
                ax.axvspan(a, min(b, e["duration_s"]), color=EPISODE_FILL, alpha=0.16, lw=0,
                           zorder=0)
            ax.set_xlim(0, e["duration_s"])
            # The end tick of a column would collide with the next column's 0.
            ax.set_xticks(np.arange(0, e["duration_s"] + (j == len(ex) - 1), 60))
            if j:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)
    top = max(max(e["envelope"]["hi_a"]) for e in ex)
    axes[0, 0].set_ylim(0, 1.05 * top)
    axes[0, 0].set_yticks([0, 4, 8])
    axes[-1, 1].set_xlabel("time in each recording (s)")

    # Positive control: development honest training against each tau_alpha.
    rng = np.random.default_rng(0)
    cax = axes[0, n]
    cax.set_title("positive control", fontsize=7, pad=3)
    for side in ("left", "bottom"):
        cax.spines[side].set_visible(False)
    cax.tick_params(left=False, bottom=False, labelbottom=False)
    for i, (key, _, _) in enumerate(ROWS, start=1):
        ax = axes[i, n]
        model, tk = CONTROL[key]
        tau = ctl["thresholds"][model][tk]
        assert tau == src["thresholds"][key], (key, tau, src["thresholds"][key])
        sc = np.array(ctl["scores"][model]["honest training"])
        ax.scatter(rng.uniform(0.15, 0.85, len(sc)), sc, s=0.6, lw=0, color=CONTROL_COLOUR,
                   alpha=0.5, rasterized=True)
        ax.axhline(tau, color=ps.INK, lw=0.6, ls=":")
        rate = float((sc > tau).mean())
        shown = f"{100 * rate:.2f}%" if 0 < rate < 0.01 else pct(rate)
        ax.text(0.5, 0.47, f"{shown} flagged", ha="center", va="center", fontsize=6,
                color=ps.INK, fontweight="bold" if key == "silent" else "normal")
        ax.set_xlim(0, 1)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        meta["positive_control"][key] = {"threshold": tau, "n_windows": len(sc),
                                         "flagged": rate}
    axes[0, 0].set_ylabel("RMS current (A)")
    for i in range(1, len(ROWS) + 1):
        axes[i, 0].set_ylabel(r"score $s_R$", fontsize=ps.MATH_PT)

    # Row headings and whole-day totals, set above each row across all columns.
    ps.panel(axes[0, 0], "a", x=-0.55, y=1.06)
    for i, (key, letter, name) in enumerate(ROWS, start=1):
        ps.panel(axes[i, 0], letter, x=-0.55, y=1.06)
        axes[i, 0].text(0, 1.085, f"{name}: {VERDICT[key]}", transform=axes[i, 0].transAxes,
                        ha="left", va="bottom", fontsize=7, fontweight="bold", color=ps.INK)
        line, counts = totals(src["committed_totals"][key])
        axes[i, n - 1].text(1, 1.085, line, transform=axes[i, n - 1].transAxes,
                         ha="right", va="bottom", fontsize=6, color=ps.INK)
        meta["annotated"][key] = {"line": line, **counts,
                                  "source": src["committed_totals"][key]["source"]}

    # Direct labels.
    last = ex[-1]
    a0, b0 = last["episodes"][0]
    axes[0, n - 1].text((a0 + b0) / 2, 0.93, "training",
                        transform=axes[0, n - 1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=6, color=ps.INK)
    for i, (key, _, _) in enumerate(ROWS, start=1):
        tau = src["thresholds"][key]
        # In the silent row tau_alpha lies just under the row heading.
        below = key == "silent"
        axes[i, 1].text(0.985, tau - 0.012 if below else tau + 0.008, r"$\tau_\alpha$",
                        transform=axes[i, 1].get_yaxis_transform(), ha="right",
                        va="top" if below else "bottom", fontsize=ps.MATH_PT, color=ps.INK)
    # Name the alarm bar once, on its first appearance in the full row.
    first = last["configs"]["full"]["alarms_s"][0]
    axes[1, n - 1].text(first[1] + 4, np.mean(ALARM_BAND), "alarm", ha="left", va="center",
                     fontsize=6, color=ALARM)
    # The working and the silent monitor look the same on declared work.
    for i, (key, _, _) in enumerate(ROWS, start=1):
        if key in QUIET:
            axes[i, 1].text(0.985, 0.775 if key == "full" else 0.66, "no alarm on declared work",
                            transform=axes[i, 1].get_yaxis_transform(), ha="right",
                            va="center", fontsize=6, fontweight="bold", color=ps.INK)
    axes[3, 0].text(0.5, 0.58, "idle lifts " r"$\tau_\alpha$" "\njust above the\ntraining scores",
                    transform=axes[3, 0].get_yaxis_transform(), ha="center", va="center",
                    fontsize=6, color=ps.MUTED)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.1)
    # Centre the x label under the three time columns, not under the middle one.
    lo, hi, mid = (axes[-1, k].get_position() for k in (0, n - 1, 1))
    axes[-1, 1].xaxis.label.set_x(((lo.x0 + hi.x1) / 2 - mid.x0) / mid.width)
    OUT.parent.mkdir(exist_ok=True)
    ps.save(fig, OUT.with_suffix(".pdf"), ps.PAGE)
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO)}")


if __name__ == "__main__":
    main()
