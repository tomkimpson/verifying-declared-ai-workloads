#!/usr/bin/env python3
"""Task 7.3: our 25 features on every decimated G&K trace.

Reads the traces task 7.2 wrote to the SSD (each checked against its
manifest sha256), featurises each 5 s trace whole through
`windows.featurise(x, gk.FS_OUT)` (amodo `extract_features` at the pinned
commit, spec D5), and writes results/gk/features.json.

    python3 scripts/gk_features.py [--workers 6] [--root DIR] [--manifest FILE]

--root reads the traces from DIR instead of the SSD's gk-200kSs/, and
--manifest reads a fetch manifest other than the committed one. Output paths
are relative and the decimation is deterministic, so `--root` alone checks a
fresh fetch against the committed manifest's sha256: a bit-for-bit
reproduction test. If your scipy decimates to different bits, that check
fails; pass the fetch's own manifest too (fetch_gk.py --out DIR --manifest
FILE) to featurise it anyway, and say so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "scripts"))

import corpus  # noqa: E402
import gk  # noqa: E402
import windows  # noqa: E402
from fetch_gk import MANIFEST, default_out  # noqa: E402

OUT = REPO / "results" / "gk" / "features.json"
N_TRACES = 1610


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.strip()


def featurise_one(job: tuple[str, str, str]) -> dict:
    root, rel, sha = job
    raw = (Path(root) / rel).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise IOError(f"{rel}: sha256 differs from the fetch manifest")
    x = np.load(Path(root) / rel).astype(np.float64)
    if len(x) != gk.N_OUT:
        raise ValueError(f"{rel}: {len(x)} samples, expected {gk.N_OUT}")
    feats, _ = windows.featurise(x, gk.FS_OUT)
    return {k: float(v) for k, v in feats.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    a = ap.parse_args()
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    lines = a.manifest.read_text().splitlines()
    header, rows = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    assert header["revision"] == gk.REVISION
    if len(rows) != N_TRACES or len({r["path"] for r in rows}) != N_TRACES:
        raise SystemExit(f"fetch manifest has {len(rows)} traces, expected {N_TRACES}: finish 7.2")
    rows.sort(key=lambda r: r["path"])
    root = a.root or default_out()
    with ProcessPoolExecutor(a.workers) as pool:
        feats = list(pool.map(featurise_one, [(str(root), r["out"], r["out_sha256"]) for r in rows],
                              chunksize=8))

    names = list(corpus.FEATURE_NAMES)
    traces = []
    for r, f in zip(rows, feats):
        lab = gk.parse_label(r["label"])
        traces.append({"label": r["label"], "subset": r["subset"], "capture": r["capture"],
                       "family": lab.family, "kind": lab.kind, "strategy": lab.strategy,
                       "cls": lab.cls, "features": {n: f[n] for n in names}})
    non_finite = {n: sum(not np.isfinite(t["features"][n]) for t in traces) for n in names}
    counts = Counter("adversarial" if t["kind"] == "adversarial" else t["cls"] for t in traces)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/gk_features.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_task": "7.3",
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "dataset": gk.DATASET, "revision": gk.REVISION,
        "fs_hz": gk.FS_OUT, "n_samples": gk.N_OUT,
        "feature_names": names,
        "n_traces": len(traces),
        "class_counts": dict(sorted(counts.items())),
        "non_finite": {k: v for k, v in non_finite.items() if v},
        "traces": traces,
    }, indent=1) + "\n")
    print(f"{len(traces)} traces, {dict(counts)}, non-finite {sum(non_finite.values())} "
          f"-> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
