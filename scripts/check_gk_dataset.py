#!/usr/bin/env python3
"""Size and format of the Gargiulo & Kulp dataset on HuggingFace (spec task 1.6).

Queries the HuggingFace API (no download), totals the bytes per subset and
per file kind, and records the go/no-go against spec D6's 200 GB budget,
beside the free space actually available on the laptop and the SSD.

    python3 scripts/check_gk_dataset.py     # writes results/gk_dataset/size.json
"""

from __future__ import annotations

import glob
import json
import re
import shutil
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "gk_dataset" / "size.json"

DATASET = "simgar/h200-power-workload-traces-10MHz_5s"
API = f"https://huggingface.co/api/datasets/{DATASET}"
BUDGET_BYTES = 200e9                       # spec D6
#: The SSD mounts as "Extreme SSD" or, after a reconnect, "Extreme SSD 1".
VOLUMES = {"laptop": Path.home(),
           "ssd": next(map(Path, sorted(glob.glob("/Volumes/Extreme SSD*"))), None)}
#: Their probe channel at 10 MHz against ours at 200 kS/s (spec D6 decimation).
DECIMATION = 10e6 / 200e3


def list_files() -> list[dict]:
    url, files = f"{API}/tree/main?recursive=true", []
    while url:
        with urllib.request.urlopen(url) as r:
            files += [f for f in json.load(r) if f["type"] == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
            url = m.group(1) if m else None
    return files


def kind(path: str) -> str:
    """`channel:<name>` for per-capture channel files, else the extension."""
    p = path.split("/")
    if "channels" in p[:-1]:
        return f"channel:{p[p.index('channels') + 1]}"
    return p[-1].rsplit(".", 1)[-1]


def main() -> int:
    with urllib.request.urlopen(API) as r:
        meta = json.load(r)
    files = list_files()
    size = {f["path"]: f.get("lfs", {}).get("size", f["size"]) for f in files}
    total = sum(size.values())
    by_subset, by_kind, n_kind = Counter(), Counter(), Counter()
    for path, b in size.items():
        by_subset[path.split("/")[0]] += b
        k = kind(path)
        by_kind[k] += b
        n_kind[k] += 1
    labels = {tuple(p.split("/")[:3]) for p in size if "channels" in p.split("/")}
    traces = max(by_kind.items(), key=lambda kv: kv[1])
    missing = [k for k, v in VOLUMES.items() if v is None or not v.exists()]
    if missing:
        raise SystemExit(f"volumes not found: {missing}; mount the SSD and rerun")
    free = {k: shutil.disk_usage(v).free for k, v in VOLUMES.items()}
    decimated = traces[1] / DECIMATION

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/check_gk_dataset.py",
        "spec_task": "1.6",
        "dataset": DATASET, "revision_sha": meta["sha"],
        "last_modified": meta["lastModified"],
        "n_files": len(files), "total_bytes": total,
        "bytes_by_subset": dict(by_subset),
        "bytes_by_kind": dict(by_kind), "files_by_kind": dict(n_kind),
        "n_labels": len(labels),
        "largest_kind": traces[0],
        "decimated_variant_published": False,
        "decimated_variant_note": (
            "no decimated waveform is published; the only small artefacts are "
            "features_930.parquet and features_adv_680.parquet, their own "
            "30 features at 10 MHz, which are not comparable with our 25 "
            "features at 200 kS/s"),
        "estimated_bytes_after_decimation_to_200kSs": decimated,
        "budget_bytes": BUDGET_BYTES,
        "within_budget": total <= BUDGET_BYTES,
        "free_bytes_now": free,
        "fits_free_space_whole": {k: total <= v for k, v in free.items()},
        "verdict": "go" if total <= BUDGET_BYTES else "no-go",
        "verdict_note": (
            "the whole download does not fit the free space on any volume "
            "today, so Phase 7 must stream: fetch, decimate, keep the "
            "200 kS/s copy, delete the 10 MHz file"
            if not any(total <= v for v in free.values()) else
            "the whole download fits the free space on at least one volume"),
    }, indent=1))
    print(f"{len(files)} files, {total / 1e9:.2f} GB; budget {BUDGET_BYTES / 1e9:.0f} GB; "
          f"free {({k: round(v / 1e9, 1) for k, v in free.items()})} GB; "
          f"decimated ~{decimated / 1e9:.1f} GB -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
