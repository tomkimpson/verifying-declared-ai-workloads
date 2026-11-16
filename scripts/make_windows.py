#!/usr/bin/env python3
"""Window the whole corpus for the paper (spec task 1.4).

Two passes over every usable capture of the families in `windows.FAMILIES`,
in every split, failures excluded by `capture_dir`:

* 5 s windows at a 5 s hop, for the snapshot claims (H1, H2, S2);
* 5 s windows at a 1 s hop, for the monitor (H3).

Writes one JSONL per (hop, split) under results/windows/ (gitignored,
regenerable), each starting with a header line, and a committed
results/windows/manifest.json listing every capture used and its window
counts. Exits non-zero unless every capture's window count equals the sum,
over its scored intervals, of floor(duration / hop) - (window / hop - 1).

Refuses to run from a working tree with uncommitted changes to tracked
files, and reads the repo commit once, so every output names the commit that
produced it.

Test-split windows are written to their own files and nothing here reads,
fits or scores them; summaries below are counts and labels only. Phase 3.2
puts a guard on loading them.

    python3 scripts/make_windows.py [--workers 6]

Units are amperes in; nothing is reported in watts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import corpus  # noqa: E402
import windows  # noqa: E402

OUT = REPO / "results" / "windows"
HOPS_S = (5.0, 1.0)


def jsonl_path(hop_s: float, split: str) -> Path:
    return OUT / f"windows-hop{hop_s:g}s-{split}.jsonl"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if dirty := git("status", "--porcelain", "--untracked-files=no"):
        sys.exit(f"uncommitted changes to tracked files; commit first:\n{dirty}")
    commit = git("rev-parse", "HEAD")

    entries = windows.usable_captures()
    arc = corpus.archive()
    caps = {}
    for e in entries:
        iv = arc.open(e["capture_id"]).scored_intervals()
        caps[e["capture_id"]] = {
            "capture_id": e["capture_id"], "run_id": e["run_id"],
            "capture_dir": corpus.capture_dir(e), "family": e["family"],
            "condition": e["condition"], "split": e["split"],
            "session": e["session"], "n_intervals": len(iv),
            "scored_s": sum(b - a for a, b in iv),
            "windows": {}, "expected": {}, "labels": {}}
        for hop in HOPS_S:
            caps[e["capture_id"]]["expected"][f"{hop:g}"] = sum(
                windows.expected_count(a, b, windows.WINDOW_S, hop) for a, b in iv)

    OUT.mkdir(parents=True, exist_ok=True)
    splits = sorted({c["split"] for c in caps.values()})
    for hop in HOPS_S:
        key = f"{hop:g}"
        handles = {s: jsonl_path(hop, s).open("w") for s in splits}
        for s, fh in handles.items():
            fh.write(json.dumps({
                "header": True, "split": s, "window_s": windows.WINDOW_S,
                "hop_s": hop, "corpus_commit": corpus.CORPUS_COMMIT,
                "archive": str(corpus.ARCHIVE), "repo_commit": commit,
                "families": windows.FAMILIES}) + "\n")
        counts, labels = Counter(), defaultdict(Counter)
        ids = [e["capture_id"] for e in entries]
        for row in windows.window_many(ids, windows.WINDOW_S, hop, args.workers):
            handles[row["split"]].write(json.dumps(row) + "\n")
            counts[row["capture_id"]] += 1
            labels[row["capture_id"]][row["label"]] += 1
        for fh in handles.values():
            fh.close()
        for cid, c in caps.items():
            c["windows"][key] = counts[cid]
            c["labels"][key] = dict(labels[cid])
        print(f"hop {hop:g} s: {sum(counts.values())} windows over {len(counts)} captures")

    bad = [(cid, h, c["windows"][h], c["expected"][h])
           for cid, c in caps.items() for h in c["windows"]
           if c["windows"][h] != c["expected"][h]]
    totals = {f"{h:g}": {s: sum(c["windows"][f"{h:g}"] for c in caps.values()
                                if c["split"] == s) for s in splits} for h in HOPS_S}
    (OUT / "manifest.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/make_windows.py", "spec_task": "1.4",
        "corpus_commit": corpus.CORPUS_COMMIT, "repo_commit": commit,
        "window_s": windows.WINDOW_S, "hops_s": list(HOPS_S),
        "families": windows.FAMILIES,
        "excluded_capture_dirs": sorted(corpus.failed_capture_dirs()),
        "n_captures": len(caps), "totals": totals,
        "count_check": {"rule": "floor(scored interval / hop) - (window/hop - 1), "
                                "summed over scored intervals",
                        "n_mismatched": len(bad), "mismatched": bad},
        "files": {f"{h:g}": {s: jsonl_path(h, s).name for s in splits} for h in HOPS_S},
        "captures": sorted(caps.values(), key=lambda c: c["capture_id"]),
    }, indent=1))
    print(json.dumps(totals))
    print(f"count check: {len(bad)} mismatched of {len(caps) * len(HOPS_S)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
