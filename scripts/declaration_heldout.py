#!/usr/bin/env python3
"""Phase 2a (spec tasks 2.1-2.3): R1 and R1a on declared data only.

For each family (Mahalanobis, isolation forest), variant (R1, R1a) and alpha
in the grid:

* §3 check 6, positive control and self-scored rate (`checks.check6_positive_control`);
* §3 check 4, held out by run: R1 refitted on the dev declared runs minus one,
  thresholded at the (1 - alpha) quantile of calibration declared windows,
  scored on the held-out run. Per-run false-positive rate with a circular
  block-bootstrap interval and the positions (0 = the run's first window) of flagged windows.
  Check 4 and the §8 3-alpha clause pool the like-for-like
  hold-outs only, those whose workload the fit has seen (amendment
  2026-10-02b); the all-run pooling is reported beside them as `literal`;
* achieved calibration false-positive rate per declared workload at the
  full-fit threshold (the pooled quantile need not be alpha for each);
* threshold stability: the full-fit threshold under a run-level bootstrap of
  the calibration declared runs, as the 95% range of threshold ratios and of
  the rate it implies on the pooled calibration windows;
* the median score of each declared run under the full dev fit: dev runs
  in-sample, calibration runs out of session (the §10 idle drift risk).

No hidden-training window is read into any scored set (`assert_not_hidden`).
Writes results/declaration/heldout.json.
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

import checks  # noqa: E402
import corpus  # noqa: E402
import declaration as d  # noqa: E402
import detectors  # noqa: E402
import groups  # noqa: E402

OUT = REPO / "results" / "declaration" / "heldout.json"
#: §8: held-out dev declared false positives above KILL_FP_MULT * alpha kill R1.
KILL_FP_MULT = 3.0
#: D3: a calibration threshold moving by more than this factor under a
#: run-level bootstrap triggers the cross-validated fallback.
STABILITY_FACTOR = 2.0


def repo_commit() -> str:
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"]).returncode:
        raise SystemExit("working tree is dirty: commit first so results cite one commit")
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def heldout(fit_rows, cal_rows, names, family, amplitude) -> dict:
    """Leave-one-dev-declared-run-out scores, per held-out run."""
    out = {}
    for rid in sorted({r["run_id"] for r in fit_rows}):
        tr = [r for r in fit_rows if r["run_id"] != rid]
        te = sorted((r for r in fit_rows if r["run_id"] == rid), key=lambda r: r["t0_s"])
        m = d.OneClass(family).fit(d.matrix(tr, names, amplitude))
        out[rid] = {"condition": te[0]["condition"], "session": te[0]["session"],
                    "cal": m.score(d.matrix(cal_rows, names, amplitude)),
                    "scores": m.score(d.matrix(te, names, amplitude)),
                    "unseen_workload": te[0]["condition"] not in {r["condition"] for r in tr}}
    return out


def stability(cal_scores_by_run: dict[str, np.ndarray], alpha: float) -> dict:
    pooled = np.concatenate(list(cal_scores_by_run.values()))
    thr = d.threshold(pooled, alpha)
    runs = list(cal_scores_by_run)
    rng = np.random.default_rng(d.SEED)
    ratios, rates = [], []
    for _ in range(d.N_BOOT):
        pick = rng.integers(0, len(runs), len(runs))
        t = d.threshold(np.concatenate([cal_scores_by_run[runs[i]] for i in pick]), alpha)
        ratios.append(t / thr)
        rates.append(float((pooled > t).mean()))
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    moved = max(hi, 1.0 / lo)
    return {"threshold": thr, "ratio_ci95": [float(lo), float(hi)],
            "implied_cal_rate_ci95": [float(np.quantile(rates, 0.025)),
                                      float(np.quantile(rates, 0.975))],
            "max_factor": float(moved), "stable": bool(moved <= STABILITY_FACTOR)}


def check4(flags: dict[str, np.ndarray], unseen: dict[str, bool], alpha: float) -> dict:
    """§3 check 4 and the §8 3-alpha clause from held-out flags per run.

    Pools the like-for-like runs (``unseen[run]`` false) for the verdict, and
    all runs as `literal` (amendment 2026-10-02b).
    """
    like = {k: v for k, v in flags.items() if not unseen[k]}
    like_rate = float(np.concatenate(list(like.values())).mean())
    llo, lhi = d.run_bootstrap_ci(like)
    pooled = float(np.concatenate(list(flags.values())).mean())
    plo, phi = d.run_bootstrap_ci(flags)
    return {"like_for_like_runs": sorted(like),
            "fp_rate": like_rate, "ci95": [llo, lhi],
            "ok": llo <= alpha <= lhi,
            "direction": ("conservative" if lhi < alpha else
                          "liberal" if llo > alpha else "within"),
            "kill_3alpha_fires": like_rate > KILL_FP_MULT * alpha,
            "literal": {"fp_rate": pooled, "ci95": [plo, phi],
                        "ok": plo <= alpha <= phi,
                        "kill_3alpha_fires": pooled > KILL_FP_MULT * alpha}}


def main() -> None:
    commit = repo_commit()
    dev_header, dev = d.load_windows("development")
    cal_header, cal = d.load_windows("calibration")
    names = detectors.replication_traces()[0]
    fit_rows = d.select(dev, groups.DECLARED)
    cal_rows = sorted(d.select(cal, groups.DECLARED), key=lambda r: (r["run_id"], r["t0_s"]))
    d.assert_not_hidden(fit_rows + cal_rows)

    results = []
    for amplitude in (False, True):
        for family in d.FAMILIES:
            ho = heldout(fit_rows, cal_rows, names, family, amplitude)
            full = d.OneClass(family).fit(d.matrix(fit_rows, names, amplitude))
            cal_full = full.score(d.matrix(cal_rows, names, amplitude))
            cal_by_run = d.by_run(cal_rows, cal_full)
            for alpha in d.ALPHAS:
                per_run, flags = {}, {}
                for rid, h in ho.items():
                    f = h["scores"] > d.threshold(h["cal"], alpha)
                    flags[rid] = f
                    lo, hi = d.block_bootstrap_ci(f)
                    per_run[rid] = {"condition": h["condition"], "session": h["session"],
                                    "unseen_workload": h["unseen_workload"],
                                    "n_windows": int(len(f)), "fp_rate": float(f.mean()),
                                    "fp_ci95": [lo, hi], "alpha_in_ci": lo <= alpha <= hi,
                                    "flagged_windows": np.nonzero(f)[0].tolist()}
                thr = d.threshold(cal_full, alpha)
                c6 = checks.check6_positive_control(family, alpha, amplitude)
                results.append({
                    "family": family, "variant": "R1a" if amplitude else "R1",
                    "alpha": alpha,
                    "check6": {k: c6[k] for k in ("threshold", "n_positive", "positive_rate",
                                                  "self_rate", "self_ci95", "ok_positive",
                                                  "ok_self", "ok")},
                    "check4": {"per_run": per_run, **check4(
                        flags, {k: h["unseen_workload"] for k, h in ho.items()}, alpha)},
                    "cal_fp_by_workload": {
                        c: float((cal_full[[r["condition"] == c for r in cal_rows]] > thr).mean())
                        for c in sorted({r["condition"] for r in cal_rows})},
                    "stability": stability(cal_by_run, alpha),
                    "median_score_by_run": {
                        **{f"dev:{k}": float(np.median(v)) for k, v in d.by_run(
                            fit_rows, full.score(d.matrix(fit_rows, names, amplitude))).items()},
                        **{f"cal:{k}": float(np.median(v)) for k, v in cal_by_run.items()}},
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/declaration_heldout.py",
        "spec_tasks": ["2.1", "2.2", "2.3"],
        "repo_commit": commit, "corpus_commit": corpus.CORPUS_COMMIT,
        "windows_repo_commit": {"development": dev_header["repo_commit"],
                                "calibration": cal_header["repo_commit"]},
        "hop_s": dev_header["hop_s"],
        "fit_runs": sorted({r["run_id"] for r in fit_rows}),
        "calibration_runs": sorted({r["run_id"] for r in cal_rows}),
        "positive_control_runs": c6["positive_runs"],
        "kill_fp_mult": KILL_FP_MULT, "stability_factor": STABILITY_FACTOR,
        "block_windows": d.BLOCK_WINDOWS, "n_boot": d.N_BOOT, "seed": d.SEED,
        "results": results,
    }, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
