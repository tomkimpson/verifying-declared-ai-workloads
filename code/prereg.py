"""The pre-registration guard (spec task 3.2).

Test-split windows may be read only once `spec/preregistration.md` is frozen.
`require()` raises unless the file exists and its SHA-256 equals
`PREREG_SHA256`, the hash of the signed-off file. Changing the
pre-registration therefore breaks every test-split read until the constant is
updated, which is a visible, reviewable amendment (spec §12).

Scope: the lock covers `declaration.load_windows`, the only sanctioned reader
of window files. It does not cover the window files opened directly, nor the
raw test captures via `corpus.archive().captures(split="test")` or
`windows.usable_captures()`. Phase 4-7 scripts read test current data through
`load_windows` only. The one sanctioned exception is task 5.3's episode
labels: test `step` events read from the captures (as `checks.py` does for
§3 check 5), which are labels, not current.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PREREG_PATH = Path(__file__).resolve().parents[1] / "spec" / "preregistration.md"

#: SHA-256 of the signed-off `spec/preregistration.md` at freeze commit 0a4dcf7 (earlier freezes 912eb3b, 0c2e058, 01d50bd superseded).
PREREG_SHA256 = "b6b0a2c968a1fe86b6a9672806422061f361f9be5d1c71ceb6fbfd24dd20e725"


def sha256(path: Path = PREREG_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(path: Path = PREREG_PATH, expected: str | None = None) -> None:
    """Raise unless the pre-registration exists and matches its frozen hash."""
    expected = PREREG_SHA256 if expected is None else expected
    if expected is None:
        raise RuntimeError("test split is locked: the pre-registration is not frozen")
    if not path.exists():
        raise RuntimeError(f"test split is locked: {path} is missing")
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"test split is locked: {path.name} sha256 {got} "
                           f"does not match the frozen {expected}")
