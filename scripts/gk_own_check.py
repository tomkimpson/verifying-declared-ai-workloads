#!/usr/bin/env python3
"""Task 7.4: the own-data check, before any of our windows is scored.

The two-stage forest (`detectors.TwoStageForest`, unchanged settings) on the
930 genuine G&K traces (task 7.3's features), under G&K's protocol for unseen
families: five-fold group k-fold, shuffled, grouped by family (each non-AI
kernel its own group), over four fold seeds and two forest seeds. The mean
macro-F1 and its range are reported, not gated (amendment 2026-10-09). G&K's
97% / 0.955, for 30 features at 10 MHz, is reported beside it.

A shuffled-label negative control (labels permuted, first fold and forest
seed) must fail: its macro-F1 must sit below every genuine run. If it doesn't,
test-fold labels are reaching the predictions and the script exits non-zero.
A row-wise permutation does not test the grouping (ungrouped CV would score
near 1.0 while the control still fails); `n_groups` in the output does.

Writes results/gk/own_check.json. Task 7.5 refuses to run unless the negative
control failed.
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
import detectors  # noqa: E402

FEATURES = REPO / "results" / "gk" / "features.json"
OUT = REPO / "results" / "gk" / "own_check.json"
FOLD_SEEDS = (0, 1, 2, 3)
FOREST_SEEDS = (0, 1)
CONTROL_SEED = 0                          # permutation of the labels
GK_REPORTED = {"accuracy": 0.97, "macro_f1": 0.955,
               "protocol": "five-fold CV grouped by family, shuffled, averaged over four "
                           "splits and two seeds; 30 features at 10 MHz (dataset README)"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.strip()


def genuine(feat: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = feat["feature_names"]
    tr = [t for t in feat["traces"] if t["kind"] != "adversarial"]
    x = np.array([[t["features"][n] for n in names] for t in tr])
    y = np.array([t["cls"] for t in tr])
    g = np.array([t["family"] for t in tr])   # the kernel, for non-AI
    return names, x, y, g


def main() -> int:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    feat = json.loads(FEATURES.read_text())
    names, x, y, g = genuine(feat)
    assert len(y) == 930 and np.isfinite(x).all()
    runs = []
    for fs in FOLD_SEEDS:
        for rs in FOREST_SEEDS:
            f1, acc = detectors.grouped_kfold_scores(x, y, g, fold_seed=fs, random_state=rs)
            runs.append({"fold_seed": fs, "forest_seed": rs, "macro_f1": f1, "accuracy": acc})
            print(f"fold seed {fs}, forest seed {rs}: macro-F1 {f1:.4f}, accuracy {acc:.4f}",
                  flush=True)
    f1s = np.array([r["macro_f1"] for r in runs])
    accs = np.array([r["accuracy"] for r in runs])
    y_perm = np.random.default_rng(CONTROL_SEED).permutation(y)
    c_f1, c_acc = detectors.grouped_kfold_scores(x, y_perm, g, fold_seed=FOLD_SEEDS[0],
                                                 random_state=FOREST_SEEDS[0])
    control_failed = bool(c_f1 < f1s.min())
    print(f"shuffled-label control: macro-F1 {c_f1:.4f}, accuracy {c_acc:.4f}", flush=True)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/gk_own_check.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_task": "7.4",
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "features_generated_at": feat["generated_at"],
        "n_traces": int(len(y)),
        "class_counts": {c: int((y == c).sum()) for c in sorted(set(y))},
        "n_groups": int(len(set(g))),
        "forest": {**detectors.FOREST_KWARGS, "random_state": list(FOREST_SEEDS)},
        "runs": runs,
        "macro_f1_mean": float(f1s.mean()), "macro_f1_range": [float(f1s.min()), float(f1s.max())],
        "accuracy_mean": float(accs.mean()), "accuracy_range": [float(accs.min()), float(accs.max())],
        "negative_control": {"label_permutation_seed": CONTROL_SEED,
                             "fold_seed": FOLD_SEEDS[0], "forest_seed": FOREST_SEEDS[0],
                             "macro_f1": c_f1, "accuracy": c_acc,
                             "failed": control_failed},
        "gk_reported": GK_REPORTED,
    }, indent=1) + "\n")
    print(f"mean macro-F1 {f1s.mean():.4f} [{f1s.min():.4f}, {f1s.max():.4f}]; "
          f"negative control {'failed as required' if control_failed else 'DID NOT FAIL'}"
          f" -> {OUT.relative_to(REPO)}")
    return 0 if control_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
