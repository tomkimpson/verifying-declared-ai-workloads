#!/usr/bin/env python3
"""Task 9.3: R1's per-window scores per workload, behind F7 (exploratory).

Scores 5 s-hop windows with two one-class models:

- R1 as frozen (heldout.fit_r1: fitted on development declared windows), with
  its tau_alpha from results/heldout/thresholds.json;
- the decode-only R1 of T4's mis-specified declaration (fitted on development
  `serving-decode` only, as scripts/t4_robustness.py fits it), with its two
  thresholds: pooled calibration and decode-only calibration.

Sets: declared windows per workload in development, calibration and test;
co-location training windows per run; development honest training windows;
development non-AI windows. Test windows are scored for declared workloads
only, which the pre-registered H2 runs already scored. The thresholds are
pre-registered or T4's; the per-window scores are descriptive (amendment
2026-10-14). Writes results/heldout/r1_scores.json.
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
import declaration as d  # noqa: E402
import groups  # noqa: E402
import heldout as h  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "heldout" / "r1_scores.json"
T4 = REPO / "results" / "robustness" / "t4.json"
SPLITS = ("development", "calibration", "test")
HOP_S = 5
#: The decode-only declaration of T4 (pre-registration §10).
MISSPEC_DECLARED = ("serving-decode",)
#: Decimals kept for the per-window scores F7 plots.
SCORE_DECIMALS = 4


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def score_sets(rows: list[dict]) -> dict[str, list[dict]]:
    """Named window sets, in plotting order."""
    out: dict[str, list[dict]] = {}
    for split in SPLITS:
        for w in groups.DECLARED:
            rr = [r for r in rows if r["split"] == split and r["condition"] == w]
            if rr:
                out[f"declared {split} {w}"] = rr
    co = [r for r in rows if groups.group(r["condition"]) == "co-location"
          and r["label"] == "training"]
    for run in sorted({r["run_id"] for r in co}):
        out[f"co-location {run}"] = [r for r in co if r["run_id"] == run]
    dev = [r for r in rows if r["split"] == "development"]
    out["honest training"] = [r for r in dev if groups.group(r["condition"]) == "honest"
                              and r["label"] == "training"]
    out["non-AI"] = [r for r in dev if groups.group(r["condition"]) == "negatives"
                     and r["condition"] not in groups.DECLARED]
    return out


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    thr = h.load_thresholds()
    assert thr["preregistration_sha256"] == prereg.PREREG_SHA256
    t4 = json.loads(T4.read_text())["misspecified_declaration"]
    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    dev = [r for r in rows if r["split"] == "development"]
    cal = h.calibration_rows([r for r in rows if r["split"] == "calibration"])
    co = [r for r in rows if groups.group(r["condition"]) == "co-location"]

    r1, f = h.fit_r1(dev)
    names = f["features"]
    # Matrices below omit the amplitude column, as T4 and h2_compare do.
    assert not f["amplitude"], "frozen R1 uses amplitude; score with it"

    ms = d.OneClass(f["family"]).fit(d.matrix(d.select(dev, MISSPEC_DECLARED), names))
    models = {"r1": r1, "decode_only": ms}

    tau = {"r1": {"declared": d.threshold(r1.score(d.matrix(cal, names)), h.ALPHA)},
           "decode_only": {
               "pooled calibration": d.threshold(ms.score(d.matrix(cal, names)), h.ALPHA),
               "decode calibration": d.threshold(ms.score(d.matrix(
                   [r for r in cal if r["condition"] in MISSPEC_DECLARED], names)), h.ALPHA)}}
    # The verification clause: thresholds and co-location detection reproduce
    # thresholds.json and T4's mis-specified-declaration rows.
    assert tau["r1"]["declared"] == thr["r1"]["threshold"], "R1 tau_alpha differs"
    assert tau["decode_only"]["pooled calibration"] == t4["pooled"]["threshold"]
    assert tau["decode_only"]["decode calibration"] == t4["declared"]["threshold"]
    detection = {}
    for k, model in models.items():
        s_co = model.score(d.matrix(co, names))
        detection[k] = {cal_name: h.detection(co, h.flags(s_co, t))["all"]["pooled"]["detection"]
                        for cal_name, t in tau[k].items()}
    for cal_name, t4_key in (("pooled calibration", "pooled"), ("decode calibration", "declared")):
        assert detection["decode_only"][cal_name] == \
            t4[t4_key]["h2"]["r1"]["co-location"]["pooled"], cal_name
    h2 = json.loads((REPO / "results" / "heldout" / "h2.json").read_text())
    assert detection["r1"]["declared"] == h2["headline"]["Z_r1"]["detection"]

    sets = score_sets(rows)
    scores = {k: {name: np.round(model.score(d.matrix(rr, names)), SCORE_DECIMALS).tolist()
                  for name, rr in sets.items()} for k, model in models.items()}
    for k in models:
        print(f"{k}: tau {tau[k]}; co-location detection {detection[k]}", flush=True)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/e3_r1_scores.py", "spec_task": "9.3",
        "preregistered": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "t4_repo_commit": json.loads(T4.read_text())["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "windows": {s: {k: headers[s][k] for k in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "alpha": h.ALPHA,
        "flag": "score > tau; co-location detection over windows labelled training",
        "notes": ["exploratory per-window scores (amendment 2026-10-14); thresholds as "
                  "frozen in thresholds.json (r1) and as in t4.json (decode_only)",
                  "r1 is fitted on development declared windows (decode, prefill, idle); "
                  "decode_only on development serving-decode only",
                  "development declared windows are in each model's fit data where the "
                  "workload is declared (in-sample scores)",
                  "honest training and non-AI: development windows only; test "
                  "windows are scored for declared workloads only"],
        "fit_runs": {"r1": f["fit_runs"],
                     "decode_only": sorted({r["run_id"] for r in d.select(dev, MISSPEC_DECLARED)})},
        "calibration_runs": sorted({r["run_id"] for r in cal}),
        "thresholds": tau, "co_location_detection": detection,
        "n_windows": {name: len(rr) for name, rr in sets.items()},
        "scores": scores,
    }) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
