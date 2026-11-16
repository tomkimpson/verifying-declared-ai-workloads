#!/usr/bin/env python3
"""Task 12.1: E3, threshold-set contamination, on committed scores (exploratory).

Promoted from notes/2026-11-07-framing-checks/calibration_breakdown.py
(amendment 2026-11-08). The breakdown arithmetic and its random draws are
unchanged, so k per alpha reproduces the notes JSON. Not pre-registered.

Reads committed artefacts only: no windows, no archive.
- results/heldout/r1_scores.json: R1's per-window scores (full fit and the
  decode-only fit of T4) and its thresholds;
- results/heldout/thresholds.json: the run-level bootstrap interval of R1's
  tau_alpha;
- results/windows/manifest.json: runs, sessions and window counts per run.

Outputs, in results/heldout/e3_calibration_breakdown.json:
1. which decode windows R1 flags, per split, and whether each is its run's
   first window (5 s hop; spec §6 story 4, stability over time);
2. the score band (min, median, max) of each workload under both fits, the
   in-fit development prefill flag rate, and the unpaced run's margin above
   the upper end of tau_alpha's bootstrap interval (story 4, specificity);
3. E3 (spec §1): k development honest-training windows mixed into the 340
   calibration declared windows, R1's fit held fixed. At each alpha, k is
   the smallest number that takes the unpaced run to 0% at the new
   threshold (median over 200 draws). Contamination of the fit itself is
   E4, not this.
E3 reads committed test declared scores, which the pre-registered runs
already read (amendment 2026-11-08).

    python3 scripts/e3_calibration_breakdown.py   # refuses a dirty tree
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCORES = REPO / "results" / "heldout" / "r1_scores.json"
THRESHOLDS = REPO / "results" / "heldout" / "thresholds.json"
MANIFEST = REPO / "results" / "windows" / "manifest.json"
OUT = REPO / "results" / "heldout" / "e3_calibration_breakdown.json"

ALPHAS = (0.01, 0.02, 0.05, 0.10)
RATE_ALPHA = 0.05
SHARES_PCT = (0, 1, 2, 3, 4, 5, 6, 8, 10, 20, 50)
N_DRAWS = 200
SEED = 0
HOP_S = 5  # results/windows/manifest.json: hops_s.0
RUN = "co-location panel-mixture-concurrent-0p5"
DECODE = "serving-decode"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def rate(scores, tau):
    return float((np.asarray(scores) > tau).mean())


def contaminated_tau(base, pool, n, alpha, rng):
    if n == 0:
        return float(np.quantile(base, 1 - alpha))
    taus = [np.quantile(np.concatenate([base, rng.choice(pool, n, replace=False)]), 1 - alpha)
            for _ in range(N_DRAWS)]
    return float(np.median(taus))


def rates_at(fit, tau):
    return {k: rate(v, tau) for k, v in fit.items()
            if k.startswith("co-location") or k in ("honest training", "non-AI",
                                                    "declared test serving-decode",
                                                    "declared test idle")}


def decode_runs(manifest: dict, split: str) -> list[dict]:
    """The split's decode runs at the 5 s hop, from the window manifest."""
    return [{"run_id": c["run_id"], "session": c["session"], "n": c["windows"][str(HOP_S)]}
            for c in manifest["captures"]
            if c["condition"] == DECODE and c["split"] == split]


def first_window_check(scores: list[float], runs: list[dict], tau: float) -> dict:
    """Flagged windows in one pooled decode set, against its run starts.

    r1_scores.json pools a split's runs in load order and does not record the
    order. Every run here has the same window count, so the run starts are
    the same whatever the order; the assert makes that a precondition.
    """
    counts = [r["n"] for r in runs]
    assert sum(counts) == len(scores), "manifest counts do not match the pooled set"
    assert len(set(counts)) == 1, "unequal runs: start indices depend on load order"
    starts = [i * counts[0] for i in range(len(runs))]
    flagged = np.where(np.asarray(scores) > tau)[0].tolist()
    return {"runs": runs, "run_start_index": starts, "n": len(scores),
            "flagged_index": flagged,
            "every_flag_is_a_run_start": set(flagged) <= set(starts),
            "runs_with_first_window_flagged": sum(s in flagged for s in starts)}


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    d = json.loads(SCORES.read_text())
    thr = json.loads(THRESHOLDS.read_text())
    manifest = json.loads(MANIFEST.read_text())
    full, dec = d["scores"]["r1"], d["scores"]["decode_only"]
    tau = d["thresholds"]["r1"]["declared"]
    assert tau == thr["r1"]["threshold"], "r1_scores.json and thresholds.json disagree"
    tau_pooled = d["thresholds"]["decode_only"]["pooled calibration"]
    tau_decode = d["thresholds"]["decode_only"]["decode calibration"]
    rng = np.random.default_rng(SEED)
    out = {"generated_by": "scripts/e3_calibration_breakdown.py", "spec_task": "12.1",
           "preregistered": False, "exploratory": True,
           "promoted_from": "notes/2026-11-07-framing-checks/calibration_breakdown.py",
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "repo_commit": git("rev-parse", "HEAD"),
           "sources": {"r1_scores.json": d["repo_commit"],
                       "thresholds.json": thr["repo_commit"],
                       "windows/manifest.json": manifest["repo_commit"]},
           "numpy": np.__version__,
           "source": "results/heldout/r1_scores.json", "repo_commit_of_source": d["repo_commit"],
           "tau_full": tau, "tau_decode_only": {"pooled": tau_pooled, "decode": tau_decode}}

    # 1. Stability over time: which decode windows the full fit flags.
    checks = {f"declared {s} {DECODE}": first_window_check(
        full[f"declared {s} {DECODE}"], decode_runs(manifest, s), tau)
        for s in ("development", "calibration", "test")}
    runs = [r for c in checks.values() for r in c["runs"]]
    out["decode_windows_flagged_full_fit"] = checks
    out["decode_summary"] = {
        "n_runs": len(runs), "n_sessions": len({r["session"] for r in runs}),
        "n_windows": sum(c["n"] for c in checks.values()),
        "n_flagged": sum(len(c["flagged_index"]) for c in checks.values()),
        "runs_with_first_window_flagged": sum(c["runs_with_first_window_flagged"]
                                              for c in checks.values()),
        "every_flag_is_a_run_start": all(c["every_flag_is_a_run_start"]
                                         for c in checks.values())}

    # 2. Specificity: score bands, the in-fit prefill rate, the unpaced margin.
    band = lambda fit: {k: [float(np.min(v)), float(np.median(v)), float(np.max(v))] for k, v in fit.items()}
    out["score_band_min_median_max"] = {"full": band(full), "decode_only": band(dec)}
    out["decode_only_flag_rates"] = {k: {"pooled": rate(v, tau_pooled), "decode": rate(v, tau_decode)}
                                     for k, v in dec.items()}
    prefill = np.asarray(full["declared development serving-prefill"])
    out["dev_prefill_in_fit"] = {"n": int(prefill.size), "n_flagged": int((prefill > tau).sum()),
                                 "rate": rate(prefill, tau)}
    tau_hi = thr["r1"]["stability"]["ci95"][1]
    run_min = float(np.min(full[RUN]))
    out["unpaced_margin"] = {"run_min": run_min, "tau": tau, "tau_ci95_hi": tau_hi,
                             "margin_over_tau": run_min - tau,
                             "margin_over_tau_ci95_hi": run_min - tau_hi,
                             "tau_ci95_source": "thresholds.json r1.stability.ci95"}
    # Where the declared band reaches the hidden band: declared windows above the
    # unpaced run's lowest score, and whether each is its decode run's first window.
    above = {k: np.where(np.asarray(v) > run_min)[0].tolist()
             for k, v in full.items() if k.startswith("declared")}
    starts = {k: c["run_start_index"] for k, c in checks.items()}
    out["declared_above_unpaced_min"] = {
        "windows": above,
        "n": sum(len(v) for v in above.values()),
        "all_are_decode_run_starts": all(set(v) <= set(starts.get(k, [])) for k, v in above.items())}

    # 3. E3: the threshold set's breakdown point (draw order as in the notes).
    base = np.concatenate([full["declared calibration serving-decode"], full["declared calibration idle"]])
    hon = np.asarray(full["honest training"])
    out["calibration_windows"] = len(base)

    out["breakdown_by_alpha"] = {}
    for a in ALPHAS:
        for n in range(0, len(hon)):
            t = contaminated_tau(base, hon, n, a, rng)
            if rate(full[RUN], t) == 0:
                out["breakdown_by_alpha"][str(a)] = {
                    "tau_clean": float(np.quantile(base, 1 - a)), "n_windows": n,
                    "share_of_calibration": n / (n + len(base)), "seconds_at_hop": n * HOP_S,
                    "tau": t, "rates": rates_at(full, t)}
                break
        else:
            raise SystemExit(f"no breakdown at alpha {a}: the unpaced run never reaches 0%")
    out["breakdown_honest_min"] = min(b["rates"]["honest training"]
                                      for b in out["breakdown_by_alpha"].values())

    out["curve_alpha_0.05"] = {}
    for pct in SHARES_PCT:
        n = int(round(len(base) * pct / (100 - pct)))
        t = contaminated_tau(base, hon, n, RATE_ALPHA, rng)
        out["curve_alpha_0.05"][str(pct)] = {"n_windows": n, "tau": t, "rates": rates_at(full, t)}

    OUT.write_text(json.dumps(out, indent=1) + "\n")

    for a, b in out["breakdown_by_alpha"].items():
        r = b["rates"]
        print(f"alpha {a}: {b['n_windows']} windows ({100 * b['share_of_calibration']:.1f}%, {b['seconds_at_hop']} s), "
              f"tau {b['tau']:.3f}; run {100 * r[RUN]:.0f}%, honest {100 * r['honest training']:.1f}%, "
              f"test decode {100 * r['declared test serving-decode']:.2f}%")
    print("decode:", out["decode_summary"])
    print("prefill in fit:", out["dev_prefill_in_fit"], "margin:", out["unpaced_margin"])
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
