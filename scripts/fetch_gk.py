#!/usr/bin/env python3
"""Task 7.2: fetch the G&K power traces and decimate them to 200 kS/s.

Streams every power-channel file of the dataset at the pinned revision
(`gk.REVISION`). For each one it checks the LFS sha256, decimates by
`gk.decimate` (exact 64/3125), and writes the float32 trace together with the
capture's annotation and record JSON to the SSD under `gk-200kSs/`. The 10 MHz
file is deleted as soon as the trace is decimated: the whole corpus (157 GB)
fits no local volume.

One manifest line per trace is appended to results/gk/fetch_manifest.jsonl,
so the script resumes where it stopped. Line 1 is a header.

    python3 scripts/fetch_gk.py [--limit N] [--workers 3] [--out DIR] [--manifest FILE]

--manifest starts (or resumes) a separate manifest, e.g. for a fresh fetch to
another volume; the default is the committed results/gk/fetch_manifest.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import gk  # noqa: E402

MANIFEST = REPO / "results" / "gk" / "fetch_manifest.jsonl"
API = f"https://huggingface.co/api/datasets/{gk.DATASET}"
RESOLVE = f"https://huggingface.co/datasets/{gk.DATASET}/resolve/{gk.REVISION}/"
RETRIES = 5
CHUNK = 1 << 20


def default_out() -> Path:
    ssd = sorted(glob.glob("/Volumes/Extreme SSD*"))
    if not ssd:
        raise SystemExit("SSD not mounted (spec §5: the G&K data live on the SSD); or pass --out")
    return Path(ssd[0]) / "gk-200kSs"


def list_power_files() -> list[dict]:
    url, files = f"{API}/tree/{gk.REVISION}?recursive=true", []
    while url:
        with urllib.request.urlopen(url) as r:
            files += [f for f in json.load(r) if f["type"] == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
            url = m.group(1) if m else None
    return sorted((f for f in files if "/channels/power/" in f["path"]), key=lambda f: f["path"])


def download(path: str, dest: Path, sha: str | None = None) -> str:
    """Stream to dest; return its sha256. Retries; raises on a sha mismatch."""
    for attempt in range(RETRIES):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(RESOLVE + path, timeout=120) as r, open(dest, "wb") as o:
                while chunk := r.read(CHUNK):
                    h.update(chunk)
                    o.write(chunk)
            digest = h.hexdigest()
            if sha is not None and digest != sha:
                raise IOError(f"sha256 mismatch for {path}: {digest} != {sha}")
            return digest
        except Exception:
            dest.unlink(missing_ok=True)
            if attempt == RETRIES - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_one(f: dict, out: Path) -> dict:
    subset, label, stem = gk.capture_key(f["path"])
    d = out / subset / label
    d.mkdir(parents=True, exist_ok=True)
    raw = d / f"{stem}.10MHz.npy.part"
    sha = download(f["path"], raw, f["lfs"]["oid"])
    x = np.load(raw)
    n_in, dtype_in = len(x), str(x.dtype)
    if n_in != gk.N_IN or x.dtype != np.float16:
        raise ValueError(f"{f['path']}: {dtype_in} x {n_in}, expected float16 x {gk.N_IN}")
    y = gk.decimate(x)
    del x
    raw.unlink()
    target = d / f"{stem}.npy"
    np.save(target, y)
    for kind, p in gk.sidecar_paths(f["path"]).items():
        download(p, d / f"{stem}.{kind}.json")
    return {
        "path": f["path"], "subset": subset, "label": label, "capture": stem,
        "source_sha256": sha, "source_bytes": f["lfs"]["size"],
        "dtype_in": dtype_in, "n_in": n_in, "n_out": len(y),
        "finite": bool(np.isfinite(y).all()),
        "out": str(target.relative_to(out)),
        "out_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N new traces")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    a = ap.parse_args()
    out = a.out or default_out()
    out.mkdir(parents=True, exist_ok=True)

    manifest = a.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines()[1:]:
            done.add(json.loads(line)["path"])
    else:
        manifest.write_text(json.dumps({
            "header": True, "generated_by": "scripts/fetch_gk.py", "spec_task": "7.2",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": gk.DATASET, "revision": gk.REVISION,
            "fs_in": gk.FS_IN, "fs_out": gk.FS_OUT, "up": gk.UP, "down": gk.DOWN,
            "decimation": "scipy.signal.resample_poly, default Kaiser window (beta 5.0), float64",
            "out_root": "gk-200kSs/ on the Extreme SSD" if a.out is None else str(out),
        }) + "\n")

    todo = [f for f in list_power_files() if f["path"] not in done]
    if a.limit is not None:
        todo = todo[:a.limit]
    print(f"{len(done)} done, {len(todo)} to fetch -> {out}", flush=True)
    lock, t0, failed = threading.Lock(), time.time(), []
    with ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(fetch_one, f, out): f for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:      # left out of the manifest, so a rerun retries it
                failed.append(futures[fut]["path"])
                print(f"FAILED {futures[fut]['path']}: {e}", flush=True)
                continue
            with lock, open(manifest, "a") as m:
                m.write(json.dumps(row) + "\n")
            rate = i / (time.time() - t0)
            print(f"[{i}/{len(todo)}] {row['label']}/{row['capture']} "
                  f"({rate * 3600:.0f}/h, ~{(len(todo) - i) / rate / 3600:.1f} h left)", flush=True)
    if failed:
        print(f"{len(failed)} failed; rerun to retry", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
