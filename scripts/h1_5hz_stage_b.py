#!/usr/bin/env python3
"""Task 4.2, 5 Hz item, Stage B (amendment 2026-10-02f; rules as run recorded
in 2026-10-02g): remove the harness line from the co-location misses and
rescore them.

Stage A found the 5.000 Hz line in `-0p5` (results/coverage/line5hz_stage_a.json).
Here each scored interval's raw current is divided by (1 + m(t)), m the
5.000 Hz component of its whole-interval envelope (code/mechanism.py), the
25 features are recomputed on the same windows as the hop-5 window file,
and each run with a miss in T1 is rescored by the replica (level 1) and the
level-2 forest with co-location held out (the T1 fit): `-0p5` (missed at both
levels) and `-0p25` (partly missed by the replica, 36.5%).

Specificity rule, on the same removal applied to each run:
* the 5.000 Hz SNR after removal, as the MEDIAN over each serving-decode run's
  intervals, is below LINE_ABSENT_SNR. The per-interval maximum is recorded
  beside it, with whether every interval clears the level;
* `-0p5`'s 5.21 Hz and honest training's 11.65 Hz micro-step lines keep at
  least KEEP_FRACTION of their median SNR (one-sided: a rise passes).

Primary variant: the fundamental only, as specified. Secondary: the
fundamental and harmonics to 25 Hz, since the line is square-ish; reported
beside, not a replacement. Development co-location data only; the level-2
fit reads all splits through the pre-registration guard, as T1.
Writes results/coverage/line5hz_stage_b.json.
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
import coverage as cv  # noqa: E402
import declaration as d  # noqa: E402
import mechanism as mech  # noqa: E402
import prereg  # noqa: E402
import windows  # noqa: E402
from detectors import fit_replica  # noqa: E402

OUT = REPO / "results" / "coverage" / "line5hz_stage_b.json"
MISSED = "panel-mixture-concurrent-0p5"
RESCORED = (MISSED, "panel-mixture-concurrent-0p25")
SERVING = ("panel-serving-decode", "panel-core-serving-decode-r1")
HONEST = "panel-training-arm1"
VARIANTS = {"fundamental": (mech.HARNESS_HZ,),
            "harmonics_to_25hz": tuple(k * mech.HARNESS_HZ for k in range(1, 6))}
#: Feature agreement of the unmodified recompute with the window file (§3 check 2).
FEATURE_RTOL = 1e-6
#: Each kept line must retain at least this fraction of its SNR.
KEEP_FRACTION = 0.8


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def snrs(x: np.ndarray, fs: float, own_hz: float | None) -> dict:
    env, rate = corpus.rms_envelope(x - x.mean(), fs)
    f, p = mech.envelope_spectrum(env, rate)
    out = {"harness": mech.line_snr(f, p, mech.HARNESS_HZ)[0]}
    if own_hz:
        out["own"] = mech.line_snr(f, p, own_hz)[0]
    return out


def process(run: str, arc, rows: list[dict] | None, names: list[str]) -> dict:
    """Line SNRs before and after each variant, per interval; and, if `rows`,
    the run's windows refeaturised after each variant."""
    cap = arc.open(run)
    fs = 1.0 / cap.sample_interval_s
    n_win = int(round(windows.WINDOW_S * fs))
    own = mech.micro_step_hz(cap)
    out = {"own_line_hz": own, "intervals": [],
           "features": {v: [] for v in VARIANTS} if rows else None,
           "unmodified_max_rel_err": 0.0}
    for k, (s0, s1) in enumerate(cap.scored_intervals()):
        i0 = int(round(s0 * fs))
        x = cap.amperes(i0, int(round(s1 * fs)))
        iv = {"before": snrs(x, fs, own)}
        wins = [r for r in rows if r["interval"] == k] if rows else []
        for v, freqs in VARIANTS.items():
            y = mech.demodulate(x, fs, freqs)
            iv[v] = snrs(y, fs, own)
            for r in wins:
                j = int(round(r["t0_s"] * fs)) - i0
                feats, _ = windows.featurise(y[j:j + n_win], fs)
                out["features"][v].append([feats[n] for n in names])
        for r in wins[:2]:
            j = int(round(r["t0_s"] * fs)) - i0
            feats, _ = windows.featurise(x[j:j + n_win], fs)
            a = np.array([feats[n] for n in names])
            b = np.array([r["features"][n] for n in names])
            err = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
            out["unmodified_max_rel_err"] = max(out["unmodified_max_rel_err"], err)
        out["intervals"].append(iv)
        print(f"  {run} interval {k}: " + "  ".join(
            f"{v} " + "/".join(f"{s:.1f}" for s in iv[v].values()) for v in ("before", *VARIANTS)),
            flush=True)
    return out


def median(rows: list[dict], variant: str, key: str) -> float:
    return float(np.median([iv[variant][key] for iv in rows]))


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    names = json.loads((REPO / "results/declaration/choice.json").read_text())["frozen"]["features"]
    all_rows = []
    for split in ("development", "calibration", "test"):
        all_rows += d.load_windows(split, 5)[1]
    arc = corpus.archive()
    targets = {run: sorted((r for r in all_rows if r["run_id"] == run),
                           key=lambda r: (r["interval"], r["t0_s"])) for run in RESCORED}
    runs = {run: process(run, arc, targets[run], names) for run in RESCORED}
    for run in (*SERVING, HONEST):
        runs[run] = process(run, arc, None, names)
    for run in RESCORED:
        assert runs[run]["unmodified_max_rel_err"] <= FEATURE_RTOL, runs[run]["unmodified_max_rel_err"]

    rep_names, replica = fit_replica()
    level2 = cv.fit_level2(cv.fit_rows(all_rows, "co-location"), names)
    rescored = {}
    for run in RESCORED:
        x0 = d.matrix(targets[run], names)
        rescored[run] = {}
        for v in ("original", *VARIANTS):
            x = x0 if v == "original" else np.array(runs[run]["features"][v])
            xr = x[:, [names.index(n) for n in rep_names]]
            rescored[run][v] = {
                lvl: cv.summarise(targets[run], model.predict(xm))["per_run"][run]
                for lvl, model, xm in (("level1_replica", replica, xr),
                                       ("level2_held_out", level2, x))}
            print(run, v, {k: round(e["detection"], 3) for k, e in rescored[run][v].items()},
                  flush=True)

    spec = {}
    for v in VARIANTS:
        spec[v] = {
            "serving_harness_snr_after": {s: median(runs[s]["intervals"], v, "harness") for s in SERVING},
            "serving_harness_snr_after_max": {s: max(iv[v]["harness"] for iv in runs[s]["intervals"])
                                              for s in SERVING},
            "missed_harness_snr_after": median(runs[MISSED]["intervals"], v, "harness"),
            "missed_own_line_kept": median(runs[MISSED]["intervals"], v, "own")
            / median(runs[MISSED]["intervals"], "before", "own"),
            "honest_own_line_kept": median(runs[HONEST]["intervals"], v, "own")
            / median(runs[HONEST]["intervals"], "before", "own"),
        }
        s = spec[v]
        s["passes"] = bool(all(x < mech.LINE_ABSENT_SNR for x in s["serving_harness_snr_after"].values())
                           and s["missed_own_line_kept"] >= KEEP_FRACTION
                           and s["honest_own_line_kept"] >= KEEP_FRACTION)
        s["serving_every_interval_below"] = bool(
            all(x < mech.LINE_ABSENT_SNR for x in s["serving_harness_snr_after_max"].values()))
        print(v, s, flush=True)
    for r in runs.values():
        r.pop("features", None)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/h1_5hz_stage_b.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "removal": {"half_width_hz": mech.REMOVE_HALF_WIDTH_HZ, "variants": VARIANTS,
                    "primary": "fundamental"},
        "specificity_rule": {"serving_harness_snr_below": mech.LINE_ABSENT_SNR,
                             "serving_aggregate": "median over each run's intervals",
                             "own_lines_keep_fraction": KEEP_FRACTION,
                             "own_lines_aggregate": "median over intervals; one-sided"},
        "specificity": spec,
        "rescored": rescored,
        "line_snrs": runs,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
