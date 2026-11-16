#!/usr/bin/env python3
"""Task 7.5: T5, the G&K-trained forests scored on our windows.

Two forests, both `detectors.TwoStageForest` with unchanged settings, fitted on
task 7.3's features of every G&K trace:
- genuine: the 930 genuine traces (training / inference / non-ai);
- hardened (T5's headline): the same plus the 680 adversarial traces, labelled
  training.

Each scores our 5 s windows from all three splits (test through the
pre-registration guard). The transfer gate (pre-registration §9) is applied per
forest on point estimates:
- condition `training`, windows labelled `training`, flagged >= 0.80 (the full
  honest group reported beside it; amendment 2026-10-06);
- `serving-decode` + `serving-prefill`, all windows, flagged <= 0.20.
Co-location is scored, per run on its `training` windows as in T1, only for a
forest that passes. Refuses to run unless task 7.4's negative control failed
on these features (the own-data score is reported, not gated; amendment
2026-10-09).

Writes results/gk/t5.json.
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
import groups  # noqa: E402
import prereg  # noqa: E402
from detectors import FOREST_KWARGS, TwoStageForest  # noqa: E402

FEATURES = REPO / "results" / "gk" / "features.json"
OWN_CHECK = REPO / "results" / "gk" / "own_check.json"
OUT = REPO / "results" / "gk" / "t5.json"
SPLITS = ("development", "calibration", "test")
HOP_S = 5
#: Pre-registration §9; `check_gate_levels` asserts the frozen text says so.
GATE_TRAINING_MIN = 0.80
GATE_INFERENCE_MAX = 0.20
GATE_TRAINING_CONDITION = "training"
GATE_INFERENCE_CONDITIONS = ("serving-decode", "serving-prefill")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.strip()


def check_gate_levels() -> None:
    text = prereg.PREREG_PATH.read_text()
    for needle in (f"flagged as training\n    at ≥ **{GATE_TRAINING_MIN:.2f}**",
                   f"flagged as training at ≤ **{GATE_INFERENCE_MAX:.2f}**"):
        if needle not in text:
            raise SystemExit(f"pre-registration §9 does not contain {needle!r}")


def gk_training_sets(feat: dict) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    names = feat["feature_names"]
    out = {}
    for forest, keep in (("genuine", lambda t: t["kind"] != "adversarial"),
                         ("hardened", lambda t: True)):
        tr = [t for t in feat["traces"] if keep(t)]
        x = np.array([[t["features"][n] for n in names] for t in tr])
        y = np.array([t["cls"] for t in tr])
        out[forest] = (x, y, sorted({t["label"] for t in tr}))
    return out


def flag_rate(rows: list[dict], pred: np.ndarray) -> dict:
    flags = pred == "training"
    by_cond = {}
    for c in sorted({r["condition"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["condition"] == c]
        by_cond[c] = {"n_windows": len(idx), "flag_rate": float(flags[idx].mean()),
                      "predicted": {k: int((pred[idx] == k).sum()) for k in sorted(set(pred[idx]))}}
    return {"n_windows": len(rows), "flag_rate": float(flags.mean()), "by_condition": by_cond}


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    feat = json.loads(FEATURES.read_text())
    own = json.loads(OWN_CHECK.read_text())
    if not own["negative_control"]["failed"] or own["features_generated_at"] != feat["generated_at"]:
        raise SystemExit("task 7.4's own-data check has not run cleanly on these features")
    check_gate_levels()

    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    names = feat["feature_names"]
    frozen = json.loads((REPO / "results/declaration/choice.json").read_text())["frozen"]["features"]
    assert sorted(names) == sorted(frozen), "G&K and corpus features differ"

    sets = {
        "gate_training": [r for r in rows if r["condition"] == GATE_TRAINING_CONDITION],
        "honest": [r for r in rows if groups.group(r["condition"]) == "honest"],
        "gate_inference": [r for r in rows if r["condition"] in GATE_INFERENCE_CONDITIONS],
        "co-location": [r for r in rows if groups.group(r["condition"]) == "co-location"],
    }
    results, manifests = {}, {}
    for forest, (x, y, labels) in gk_training_sets(feat).items():
        manifests[forest] = labels
        model = TwoStageForest().fit(x, y)
        pred = {k: model.predict(d.matrix(v, names)) for k, v in sets.items() if k != "co-location"}
        train = cv.summarise(sets["gate_training"], pred["gate_training"])
        infer = flag_rate(sets["gate_inference"], pred["gate_inference"])
        t_rate, i_rate = train["pooled"]["detection"], infer["flag_rate"]
        passed = bool(t_rate >= GATE_TRAINING_MIN and i_rate <= GATE_INFERENCE_MAX)
        res = {
            "n_fit_traces": int(len(y)),
            "fit_class_counts": {c: int((y == c).sum()) for c in sorted(set(y))},
            "gate": {"training_detection": t_rate, "training_min": GATE_TRAINING_MIN,
                     "inference_flag_rate": i_rate, "inference_max": GATE_INFERENCE_MAX,
                     "passed": passed},
            "gate_training": train,
            "honest_group": cv.summarise(sets["honest"], pred["honest"]),
            "gate_inference": infer,
        }
        if passed:
            res["co-location"] = cv.summarise(
                sets["co-location"], model.predict(d.matrix(sets["co-location"], names)))
        else:
            res["co-location"] = "not scored: the transfer gate failed for this forest"
        results[forest] = res
        print(f"{forest}: honest `training` {t_rate:.3f} (>= {GATE_TRAINING_MIN}), "
              f"declared inference {i_rate:.3f} (<= {GATE_INFERENCE_MAX}): "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
        if passed:
            for run, v in res["co-location"]["per_run"].items():
                print(f"  {run}: {v['detection']}", flush=True)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/t5_gk_transfer.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_task": "7.5",
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "dataset": feat["dataset"], "revision": feat["revision"],
        "features_generated_at": feat["generated_at"],
        "own_check": {**{k: own[k] for k in ("macro_f1_mean", "macro_f1_range", "accuracy_mean",
                                             "negative_control", "gk_reported")},
                      "n_runs": len(own["runs"])},
        "windows": {s: {k: headers[s][k] for k in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "features": names,
        "forest": FOREST_KWARGS,
        "headline_forest": "hardened",
        "flag": "predict == 'training'; detection over windows labelled training, partial "
                "excluded; inference flag rate over all windows",
        "forests": results,
        "fit_manifests": manifests,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
