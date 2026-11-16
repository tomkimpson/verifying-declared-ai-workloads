#!/usr/bin/env python3
"""Task 13.1: E4, fit contamination (exploratory, amendments 2026-11-08 and 2026-11-13).

The case Scher & Thiergart flag: the operator's declared reference already
contains some training. R1 (frozen family, features and amplitude setting) is
refitted on its development declared windows plus k development honest-training
windows drawn at random from the pooled honest set. The calibration windows and
the threshold rule are unchanged; tau_alpha is recomputed for each refit, since
the model changes.

For each k: the unpaced co-location run's detection at tau_alpha, honest
detection on the development honest windows not drawn (not independent: they
share runs with the drawn ones), and honest detection on calibration honest
training (another session, outside every fit).

Also, for each k (amendment 2026-11-16b): the flag rate on the development declared
windows, which are in the fit (in-sample), overall and per condition, and on the
calibration declared windows, which set tau_alpha (about alpha by construction).
These ask whether contamination raises alarms on declared work.

Reads development and calibration windows only; no test window. Writes
results/heldout/e4_fit_contamination.json.
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

OUT = REPO / "results" / "heldout" / "e4_fit_contamination.json"
SPLITS = ("development", "calibration")
HOP_S = 5
#: The unpaced (unthrottled) co-location run R1 detects at 1.000 (h2.json).
UNPACED = "panel-mixture-concurrent-0p5"
#: Geometric grid of k, up to the whole development honest pool (amendment 2026-11-13).
K_GRID = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 2905)
N_DRAWS = 20
#: Detection levels whose smallest k is reported (spec §1 E4).
LEVELS = (0.5, 0.05)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def summary(v: list) -> dict | None:
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    thr = h.load_thresholds()
    assert thr["preregistration_sha256"] == prereg.PREREG_SHA256
    f = h.frozen_r1()
    names, amp = f["features"], f["amplitude"]

    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    assert all(r["split"] in SPLITS for r in rows), "a test window was read"
    dev = [r for r in rows if r["split"] == "development"]
    cal = h.calibration_rows([r for r in rows if r["split"] == "calibration"])

    declared = d.select(dev, groups.DECLARED)
    assert sorted({r["run_id"] for r in declared}) == f["fit_runs"]
    pool = [r for r in dev if groups.group(r["condition"]) == "honest" and r["label"] == "training"]
    assert len(pool) == K_GRID[-1], len(pool)
    unpaced = [r for r in dev if r["run_id"] == UNPACED and r["label"] == "training"]
    cal_honest = [r for r in rows if r["split"] == "calibration"
                  and groups.group(r["condition"]) == "honest" and r["label"] == "training"]

    x_decl, x_pool = d.matrix(declared, names, amp), d.matrix(pool, names, amp)
    x_cal, x_unp = d.matrix(cal, names, amp), d.matrix(unpaced, names, amp)
    x_calh = d.matrix(cal_honest, names, amp)
    calh_run = np.array([r["run_id"] for r in cal_honest])
    pool_run = np.array([r["run_id"] for r in pool])
    decl_cond = np.array([r["condition"] for r in declared])

    # The verification clause: k = 0 is R1 as frozen.
    r1, _ = h.fit_r1(dev)
    tau_r1 = d.threshold(r1.score(x_cal), h.ALPHA)
    assert tau_r1 == thr["r1"]["threshold"], (tau_r1, thr["r1"]["threshold"])
    det_r1 = h.detection(unpaced, h.flags(r1.score(x_unp), tau_r1))["all"]["pooled"]["detection"]
    assert det_r1 == 1.0, det_r1

    grid = []
    for k in K_GRID:
        draws = []
        for seed in range(1 if k in (0, len(pool)) else N_DRAWS):
            idx = np.sort(np.random.default_rng(seed).choice(len(pool), k, replace=False))
            rest = np.setdiff1d(np.arange(len(pool)), idx)
            m = d.OneClass(f["family"]).fit(np.vstack([x_decl, x_pool[idx]]))
            s_cal, s_decl = m.score(x_cal), m.score(x_decl)
            tau = d.threshold(s_cal, h.ALPHA)
            draws.append({
                "seed": seed, "tau": tau,
                "unpaced_detection": float(h.flags(m.score(x_unp), tau).mean()),
                "honest_not_drawn_detection":
                    float(h.flags(m.score(x_pool[rest]), tau).mean()) if len(rest) else None,
                "calibration_honest_detection": float(h.flags(m.score(x_calh), tau).mean()),
                "drawn_runs": len({pool[i]["run_id"] for i in idx}),
                "calibration_honest_by_run": {
                    run: float(h.flags(m.score(x_calh[calh_run == run]), tau).mean())
                    for run in sorted(set(calh_run))},
                "honest_not_drawn_by_run": {
                    run: float(h.flags(m.score(x_pool[rest][pool_run[rest] == run]), tau).mean())
                    for run in sorted(set(pool_run[rest]))},
                "declared_fit_flag_rate": float(h.flags(s_decl, tau).mean()),
                "declared_fit_flag_rate_by_condition": {
                    c: float(h.flags(s_decl[decl_cond == c], tau).mean())
                    for c in sorted(set(decl_cond))},
                "calibration_declared_flag_rate": float(h.flags(s_cal, tau).mean()),
            })
        if k == 0:
            assert draws[0]["tau"] == tau_r1 and draws[0]["unpaced_detection"] == det_r1
        entry = {"k": k, "share_of_fit": k / (len(declared) + k), "n_draws": len(draws),
                 **{key: summary([x[key] for x in draws]) for key in
                    ("tau", "unpaced_detection", "honest_not_drawn_detection",
                     "calibration_honest_detection")},
                 "calibration_honest_by_run": {
                     run: summary([x["calibration_honest_by_run"][run] for x in draws])
                     for run in sorted(set(calh_run))},
                 "honest_not_drawn_by_run": {
                     run: summary([x["honest_not_drawn_by_run"].get(run) for x in draws])
                     for run in sorted(set(pool_run))},
                 **{key: summary([x[key] for x in draws]) for key in
                    ("declared_fit_flag_rate", "calibration_declared_flag_rate")},
                 "declared_fit_flag_rate_by_condition": {
                     c: summary([x["declared_fit_flag_rate_by_condition"][c] for x in draws])
                     for c in sorted(set(decl_cond))},
                 "draws": draws}
        grid.append(entry)
        print(f"k={k:5d} share={entry['share_of_fit']:.3f} tau={entry['tau']['median']:.4f} "
              f"unpaced={entry['unpaced_detection']} cal_honest="
              f"{entry['calibration_honest_detection']['median']:.3f}", flush=True)

    smallest = {}
    for lv in LEVELS:
        hit = [g["k"] for g in grid if g["unpaced_detection"]["median"] < lv]
        smallest[str(lv)] = ({"k": hit[0], "share_of_fit": hit[0] / (len(declared) + hit[0])}
                             if hit else f"not reached by k = {K_GRID[-1]}")
    print("smallest k (median unpaced detection below level):", smallest)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/e4_fit_contamination.py", "spec_task": "13.1",
        "preregistered": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "windows": {s: {key: headers[s][key] for key in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "splits_read": list(SPLITS),
        "alpha": h.ALPHA, "family": f["family"], "amplitude": amp,
        "flag": "score > tau; tau the (1 - alpha) quantile of calibration declared scores",
        "notes": ["exploratory, not pre-registered (amendments 2026-11-08, 2026-11-13)",
                  "fit: development declared windows plus k development honest-training "
                  "windows drawn at random (numpy default_rng(seed), without replacement)",
                  "calibration windows and threshold rule unchanged; tau recomputed per refit",
                  "honest_not_drawn shares runs with the drawn windows, so it is not an "
                  "independent test; calibration_honest is another session, outside every fit",
                  "median/min/max are over draws; one draw at k = 0 and k = all",
                  "declared_fit_flag_rate is in-sample (those windows are in every fit); "
                  "calibration_declared_flag_rate is about alpha by construction "
                  "(amendment 2026-11-16b)"],
        "fit_runs": f["fit_runs"], "n_declared_fit": len(declared),
        "honest_pool": {"n_windows": len(pool), "runs": sorted({r["run_id"] for r in pool})},
        "calibration_runs": sorted({r["run_id"] for r in cal}),
        "calibration_honest_runs": sorted({r["run_id"] for r in cal_honest}),
        "unpaced_run": UNPACED, "n_unpaced": len(unpaced),
        "k0": {"tau": tau_r1, "unpaced_detection": det_r1},
        "smallest_k": smallest,
        "grid": grid,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
