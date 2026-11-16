#!/usr/bin/env python3
"""Task 4.3: F1, 2 s envelope excerpts (spec §6, amendment 2026-10-02g).

Declared inference, honest training, time-shared training and each
co-location run: the 2 ms RMS envelope of the raw current, divided by the
excerpt's mean (amplitude-free; gain is not comparable across sessions),
with optimizer-step ends marked. Excerpts start at a rule, not a choice:
the first step start at least OFFSET_S into the first scored interval (minus
LEAD_S), or OFFSET_S into it for a run without steps. Development runs only.
Writes figures/f1_excerpts.pdf and figures/f1_excerpts.json (the excerpt
times and each excerpt's envelope coefficient of variation).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import corpus  # noqa: E402
import plotstyle as ps  # noqa: E402
from plotstyle import plt  # noqa: E402

OUT = REPO / "figures" / "f1_excerpts"
EXCERPT_S = 2.0
OFFSET_S = 30.0
LEAD_S = 0.1
PANELS = (("declared inference", "panel-serving-decode"),
          ("honest training", "panel-training-arm1"),
          ("time-shared training", "panel-mixture-timeshared-0p5"),
          ("panel-mixture-concurrent-0p25", "panel-mixture-concurrent-0p25"),
          ("panel-mixture-concurrent-0p5", "panel-mixture-concurrent-0p5"),
          ("panel-mixture-concurrent-0p75", "panel-mixture-concurrent-0p75"))


def excerpt(run: str, arc) -> dict:
    cap = arc.open(run)
    assert cap.entry["split"] == "development", run
    fs = 1.0 / cap.sample_interval_s
    s0, s1 = cap.scored_intervals()[0]
    ev = cap.events("step")
    starts = np.sort([e["scope_start_s"] for e in ev])
    ends = np.sort([e["scope_end_s"] for e in ev])
    t0 = (starts[starts >= s0 + OFFSET_S][0] - LEAD_S) if len(starts) else s0 + OFFSET_S
    assert t0 + EXCERPT_S <= s1
    i0 = int(round(t0 * fs))
    x = cap.amperes(i0, i0 + int(round(EXCERPT_S * fs)))
    env, rate = corpus.rms_envelope(x - x.mean(), fs)
    env = env / env.mean()
    t = (np.arange(len(env)) + 0.5) / rate
    return {"t0_s": float(t0), "t": t, "env": env, "env_cv": float(env.std()),
            "step_ends": (ends[(ends >= t0) & (ends <= t0 + EXCERPT_S)] - t0).tolist()}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    ps.apply()
    arc = corpus.archive()
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(ps.COL, 3.7), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.08})
    meta = {"generated_by": "scripts/plot_f1_excerpts.py", "repo_commit": git("rev-parse", "HEAD"),
            "corpus_commit": corpus.CORPUS_COMMIT,
            "excerpt_s": EXCERPT_S, "offset_s": OFFSET_S, "lead_s": LEAD_S, "panels": {}}
    for ax, (key, run) in zip(axes, PANELS):
        e = excerpt(run, arc)
        ax.plot(e["t"], e["env"], color=ps.SERIES[key], lw=0.5)
        for s in e["step_ends"]:
            ax.axvline(s, color=ps.MUTED, lw=0.5, ls=":")
        # Label inside the panel, in the headroom above the envelope.
        ax.text(0.01, 0.95, f"{ps.label(key)}   CV {e['env_cv']:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=5.5, color=ps.INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5})
        ax.set_ylim(0, 3.4)
        ax.set_yticks([0, 1, 2])
        ax.tick_params(labelsize=5.5)
        meta["panels"][key] = {"run": run, "t0_s": e["t0_s"], "env_cv": e["env_cv"],
                               "n_step_ends": len(e["step_ends"])}
    axes[-1].set_xlabel("time in excerpt (s); dotted: optimizer-step ends", fontsize=6.5)
    fig.supylabel("2 ms RMS current / excerpt mean", fontsize=6.5)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.1)
    OUT.parent.mkdir(exist_ok=True)
    # Exactly one column wide, so the text prints at its set size.
    ps.save(fig, OUT.with_suffix(".pdf"), ps.COL)
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO)}")


if __name__ == "__main__":
    main()
