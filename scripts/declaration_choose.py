#!/usr/bin/env python3
"""Phase 2a, task 2.4: freeze R1's family, variant and alpha from declared data.

Reads results/declaration/heldout.json only (declared and honest-training
windows; no hidden-training run). The rule, applied in order:

1. alpha <= 0.05, the bound of the §8 co-location clause;
2. required: the check-6 positive control passes and the like-for-like 3-alpha
   clause does not fire (§8, amendment 2026-10-02b);
3. alpha: the largest remaining, so the calibration quantile rests on the
   most windows (17 of 340 at 0.05, against 3 at 0.01);
4. variant: R1 unless no R1 candidate passes step 2 (parsimony: amplitude
   enters only if it earns its place);
5. family, among that variant's candidates: prefer one whose calibration
   threshold passes D3's <= 2x stability test, so the primary threshold rule
   applies rather than the fallback.

Variant is fixed before family. That is the order the code applies and the
order that froze choice.json (2.4); an earlier docstring listed them the other
way round (PR #2 review). The two orders pick the same candidate on these data,
where all 16 candidates pass step 2.

Checks 4 and 6 (self-scored) fail for every candidate in the conservative
direction, so they cannot discriminate and are not in the rule; they are
recorded. Writes results/declaration/choice.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import declaration as d  # noqa: E402
import detectors  # noqa: E402

HELDOUT = REPO / "results" / "declaration" / "heldout.json"
OUT = REPO / "results" / "declaration" / "choice.json"
ALPHA_MAX = 0.05


def required(r: dict) -> bool:
    return r["check6"]["ok_positive"] and not r["check4"]["kill_3alpha_fires"]


def choose(results: list[dict]) -> dict:
    cand = [r for r in results if r["alpha"] <= ALPHA_MAX]
    alpha = max(r["alpha"] for r in cand if required(r))
    at = [r for r in cand if r["alpha"] == alpha]
    r1 = [r for r in at if r["variant"] == "R1" and required(r)]
    pool = r1 or [r for r in at if r["variant"] == "R1a" and required(r)]
    stable = [r for r in pool if r["stability"]["stable"]]
    pick = (stable or pool)
    if len(pick) != 1:
        raise SystemExit(f"rule does not single out one candidate: "
                         f"{[(r['family'], r['variant']) for r in pick]}")
    return pick[0]


def main() -> None:
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"]).returncode:
        raise SystemExit("working tree is dirty: commit first")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    h = json.loads(HELDOUT.read_text())
    c = choose(h["results"])
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/declaration_choose.py",
        "spec_task": "2.4", "repo_commit": commit,
        "heldout_repo_commit": h["repo_commit"], "corpus_commit": h["corpus_commit"],
        "frozen": {
            "family": c["family"], "variant": c["variant"], "alpha": c["alpha"],
            "amplitude": c["variant"] == "R1a",
            "threshold_rule": "(1 - alpha) quantile of calibration declared windows (D3 primary)",
            "threshold": c["stability"]["threshold"],
            "iforest_kwargs": d.IFOREST_KWARGS if c["family"] == "iforest" else None,
            "features": detectors.replication_traces()[0],
            "fit_runs": h["fit_runs"], "calibration_runs": h["calibration_runs"],
            "hop_s": h["hop_s"],
        },
        "evidence": c,
        "notes": [
            "Checks 4 and 6 (self-scored) fail conservative for every candidate (amendment 2026-10-02b).",
            "The isolation forest passes D3's <= 2x ratio test while Mahalanobis does not, but its "
            "scores are compressed in (0, 1]: the implied calibration rate under the run bootstrap "
            "spans the same range for both families (stability.implied_cal_rate_ci95).",
        ],
    }, indent=1) + "\n")
    print(f"wrote {OUT}: {c['family']} {c['variant']} alpha={c['alpha']}")


if __name__ == "__main__":
    main()
