#!/usr/bin/env python3
"""Task 4.2, 5 Hz item, Stage A (amendment 2026-10-02f). Removes nothing.

Per scored interval of the three co-location runs and the two dev
serving-decode runs: the envelope spectrum over the whole interval, and the
line SNR at exactly 5.000 Hz (the harness line) and at the run's micro-step
rhythm: micro-steps per step (from the step payloads) over the median `step`
duration. For the gapped `-0p25` that is the rhythm inside each step, not
4 x its step rate; the median step period is recorded beside it.

Rule: if `-0p5` has 5.000 Hz SNR < LINE_ABSENT_SNR in every interval while
serving-decode carries the line, the miss cannot depend on the harness line
and the harness candidate is rejected. Otherwise Stage B is due. Either way
the verdict goes to Tom (spec §8). Development data only.
Writes results/coverage/line5hz_stage_a.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import corpus  # noqa: E402
import mechanism as mech  # noqa: E402

OUT = REPO / "results" / "coverage" / "line5hz_stage_a.json"
MISSED = "panel-mixture-concurrent-0p5"
COLOCATION = ("panel-mixture-concurrent-0p25", MISSED, "panel-mixture-concurrent-0p75")
SERVING = ("panel-serving-decode", "panel-core-serving-decode-r1")
RUNS = (*COLOCATION, *SERVING)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def run_row(run: str, arc) -> dict:
    cap = arc.open(run)
    assert cap.entry["split"] == "development", run
    ev = cap.events("step")
    row = {"condition": cap.entry["condition"], "intervals": []}
    if ev:
        dur = np.median([e["scope_end_s"] - e["scope_start_s"] for e in ev])
        period = np.median(np.diff(sorted(e["scope_start_s"] for e in ev)))
        row.update(step_duration_s=float(dur), step_period_s=float(period),
                   micro_step_hz=mech.micro_step_hz(cap))
    for t0, s1, env, rate in mech.interval_envelopes(run, arc):
        f, p = mech.envelope_spectrum(env, rate)
        snr, fpk = mech.line_snr(f, p, mech.HARNESS_HZ)
        iv = {"t0_s": t0, "t1_s": s1, "bin_spacing_hz": float(f[1]), "record_s": s1 - t0,
              "harness_snr": snr, "harness_peak_hz": fpk}
        if ev:
            iv["micro_step_snr"], iv["micro_step_peak_hz"] = mech.line_snr(f, p, row["micro_step_hz"])
        row["intervals"].append(iv)
    row["harness_snr_max"] = max(i["harness_snr"] for i in row["intervals"])
    row["harness_snr_median"] = float(np.median([i["harness_snr"] for i in row["intervals"]]))
    if ev:
        row["micro_step_snr_median"] = float(np.median([i["micro_step_snr"] for i in row["intervals"]]))
    return row


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    arc = corpus.archive()
    rows = {}
    for run in RUNS:
        rows[run] = run_row(run, arc)
        r = rows[run]
        print(f"{run:32s} 5.000 Hz SNR median {r['harness_snr_median']:8.1f} max {r['harness_snr_max']:8.1f}"
              + (f"  micro-step {r['micro_step_hz']:.3f} Hz SNR {r['micro_step_snr_median']:.1f}"
                 if "micro_step_hz" in r else ""), flush=True)
    absent = rows[MISSED]["harness_snr_max"] < mech.LINE_ABSENT_SNR
    serving_has = all(rows[s]["harness_snr_median"] >= mech.LINE_ABSENT_SNR for s in SERVING)
    verdict = ("harness candidate rejected: the 5.000 Hz line is absent from the missed run"
               if absent and serving_has else
               "Stage B due: the rejection rule does not hold")
    OUT.write_text(json.dumps({
        "generated_by": "scripts/h1_5hz_stage_a.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "rule": {"harness_hz": mech.HARNESS_HZ, "absent_below_snr": mech.LINE_ABSENT_SNR,
                 "search_hz": mech.LINE_SEARCH_HZ, "guard_hz": mech.LINE_GUARD_HZ,
                 "noise_half_width_hz": mech.LINE_NOISE_HALF_WIDTH_HZ,
                 "missed_run": MISSED, "serving_runs": list(SERVING)},
        "missed_run_line_absent": absent,
        "serving_carries_line": serving_has,
        "verdict": verdict,
        "runs": rows,
    }, indent=1) + "\n")
    print(verdict)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
