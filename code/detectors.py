"""Detectors. Phase 1 holds the published-recipe replica (spec D2, level 1).

The two-stage forest of Gargiulo & Kulp as reimplemented in amodo
`scripts/classify_replication.py`: stage 1 separates AI from non-AI, stage 2
(fitted on AI rows only) separates training from inference, and a row stage 1
calls non-AI is non-AI whatever stage 2 would say. Hyperparameters are the
replication's, fixed a priori.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import corpus

FOREST_KWARGS = {"n_estimators": 500, "random_state": 0, "n_jobs": -1}
NON_AI = "non-ai"
AI_CLASSES = ("training", "inference")


def replication_traces() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(feature names, X, class, group, run ids) of the 300 replication traces."""
    rep = json.loads((corpus.CORPUS_REPO / "results/replication/features.json").read_text())
    names = list(rep["feature_names"])
    tr = rep["traces"]
    x = np.array([[t["features"][n] for n in names] for t in tr])
    y = np.array([t["class"] for t in tr])
    g = np.array([t["group"] for t in tr])
    return names, x, y, g, [t["run_id"] for t in tr]


class TwoStageForest:
    def __init__(self, **kwargs):
        self.kwargs = {**FOREST_KWARGS, **kwargs}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TwoStageForest":
        self.stage1 = RandomForestClassifier(**self.kwargs).fit(
            x, np.where(y == NON_AI, NON_AI, "ai"))
        ai = np.isin(y, AI_CLASSES)
        self.stage2 = RandomForestClassifier(**self.kwargs).fit(x[ai], y[ai])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.full(len(x), NON_AI, dtype=object)
        ai = self.stage1.predict(x) == "ai"
        if ai.any():
            out[ai] = self.stage2.predict(x[ai])
        return out

    def p_training(self, x: np.ndarray) -> np.ndarray:
        """Stage-2 P(training) on rows stage 1 calls AI, 0 elsewhere."""
        p = np.zeros(len(x))
        ai = self.stage1.predict(x) == "ai"
        if ai.any():
            col = list(self.stage2.classes_).index("training")
            p[ai] = self.stage2.predict_proba(x[ai])[:, col]
        return p


def fit_replica() -> tuple[list[str], TwoStageForest]:
    """Level 1: the two-stage forest fitted on all 300 replication traces."""
    names, x, y, _, _ = replication_traces()
    return names, TwoStageForest().fit(x, y)


def leave_one_group_out_macro_f1(x: np.ndarray, y: np.ndarray,
                                 groups: np.ndarray) -> float:
    """Pooled macro-F1 of the two-stage forest under leave-one-group-out,
    as amodo `classify_replication.cross_validate` + `score` compute it."""
    pooled = np.empty(len(y), dtype=object)
    for g in sorted(set(groups)):
        test = groups == g
        pooled[test] = TwoStageForest().fit(x[~test], y[~test]).predict(x[test])
    return float(f1_score(y, pooled.astype(str), average="macro", zero_division=0))


def grouped_kfold_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                         n_splits: int = 5, fold_seed: int = 0,
                         **forest_kwargs) -> tuple[float, float]:
    """(pooled macro-F1, accuracy) of the two-stage forest under shuffled
    group k-fold: G&K's protocol for unseen families (task 7.4)."""
    pooled = np.empty(len(y), dtype=object)
    folds = GroupKFold(n_splits, shuffle=True, random_state=fold_seed)
    for tr, te in folds.split(x, y, groups):
        pooled[te] = TwoStageForest(**forest_kwargs).fit(x[tr], y[tr]).predict(x[te])
    pred = pooled.astype(str)
    return (float(f1_score(y, pred, average="macro", zero_division=0)),
            float((pred == y).mean()))
