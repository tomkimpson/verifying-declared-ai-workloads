"""Read-only access to the amodo current corpus, pinned to one corpus commit.

Spec D5: the feature definition comes from amodo-power-experiments, so the
replication and this paper share one definition. The files we execute or read
from that repo are vendored, unmodified, in `vendor/amodo/` at `CORPUS_COMMIT`.
"Pinned" is enforced, not just recorded: on import, every vendored file must
match its SHA-256 in `vendor/amodo/SHA256SUMS`, which were taken from
`git show <commit>:<path>` in the corpus repo.

The archive loader (`dataset/loader.py`) ships inside the archive itself and
is covered by the archive's own SHA256SUMS.

Units are amperes. Nothing here, or downstream, is reported in watts.
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys
from pathlib import Path

#: Vendored copy of the amodo-power-experiments files this repo uses, laid out
#: as in that repo (code/..., results/...).
CORPUS_REPO = Path(__file__).resolve().parents[1] / "vendor" / "amodo"


def _find_archive() -> Path:
    """`$AMODO_ARCHIVE`, else the SSD's archive under whichever name macOS
    mounted it ("Extreme SSD", or "Extreme SSD 1" after a reconnect)."""
    if env := os.environ.get("AMODO_ARCHIVE"):
        return Path(env)
    hits = sorted(glob.glob("/Volumes/Extreme SSD*/amodo-raw-export/dataset/index.json"))
    if not hits:
        raise FileNotFoundError("amodo archive not found: mount the SSD or set AMODO_ARCHIVE")
    return Path(hits[0]).parents[1]


ARCHIVE = _find_archive()
CORPUS_COMMIT = "73ffebeb8c8055cfa85b5edb59e5d26bb94f41c6"

#: Corpus paths whose code we execute. `features` imports `channel.aliasing`
#: and `channel.envelope`, so the whole package is pinned.
PINNED_PATHS = ("code/features.py", "code/channel")

#: Corpus results this repo reads, also checked against the pinned commit.
PINNED_RESULTS = ("results/replication/features.json",
                  "results/replication/classification.json",
                  "results/panel/episodes-recovery.json")


class CorpusNotPinned(RuntimeError):
    pass


def check_pinned(paths=PINNED_PATHS + PINNED_RESULTS) -> None:
    """Raise unless the vendored files under `paths` match their pinned hashes.

    Three conditions: each pinned path has at least one entry in SHA256SUMS (a
    typo would otherwise pass silently), every listed file under it exists with
    its recorded hash, and no unlisted .py or .json file sits under a pinned
    path (where it could be imported or read).
    """
    sums = {}
    for line in (CORPUS_REPO / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        sums[rel] = digest

    def under(rel: str, p: str) -> bool:
        return rel == p or rel.startswith(p.rstrip("/") + "/")

    absent = [p for p in paths if not any(under(rel, p) for rel in sums)]
    bad = [rel for rel, digest in sums.items()
           if any(under(rel, p) for p in paths)
           and (not (CORPUS_REPO / rel).is_file()
                or hashlib.sha256((CORPUS_REPO / rel).read_bytes()).hexdigest() != digest)]
    extra = [str(f.relative_to(CORPUS_REPO)) for p in paths
             for f in ([CORPUS_REPO / p] if (CORPUS_REPO / p).is_file()
                       else (CORPUS_REPO / p).rglob("*"))
             if f.suffix in (".py", ".json") and str(f.relative_to(CORPUS_REPO)) not in sums]
    if absent or bad or extra:
        raise CorpusNotPinned(
            f"{CORPUS_REPO} does not match pinned commit {CORPUS_COMMIT[:7]}: "
            f"not in SHA256SUMS {absent}, missing or changed {bad}, unlisted {extra}")

check_pinned()
for _p in (CORPUS_REPO / "code", ARCHIVE / "dataset"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from channel.envelope import rms_envelope  # noqa: E402,F401
from features import FEATURE_NAMES, FRAME_S, extract_features  # noqa: E402,F401
from loader import Archive  # noqa: E402,F401


def archive() -> "Archive":
    return Archive(ARCHIVE)


def failed_capture_dirs() -> set[str]:
    """`capture_dir`s of every failed or superseded capture (dataset/failures.json).

    Exclusion is by directory, never by run_id: 12 failed run ids have good
    recaptures under the same id.
    """
    return {e["capture_dir"] for e in archive().failures()["entries"]}


def capture_dir(entry: dict) -> str:
    """The capture directory an index entry's sidecar lives in."""
    return str(Path(entry["sidecar_path"]).parent)
