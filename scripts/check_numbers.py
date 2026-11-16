#!/usr/bin/env python3
"""Task 8.3: fail if any number in the paper's prose does not trace to JSON.

Checks paper/main.tex and paper/sections/*.tex: every digit string must be a
macro from paper/numbers.tex or on numbers.toml's [check] allow list, and
numbers.tex must match its JSON sources. Exit 1 with a list of problems.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import papernums  # noqa: E402


def main() -> int:
    problems = papernums.check(REPO / "paper", REPO)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
