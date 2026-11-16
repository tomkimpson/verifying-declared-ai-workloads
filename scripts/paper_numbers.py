#!/usr/bin/env python3
"""Task 8.3: write paper/numbers.tex from paper/numbers.toml.

Each macro's value is read from the committed JSON artefact named in
numbers.toml. Run after any results change; scripts/check_numbers.py fails
if numbers.tex is stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import papernums  # noqa: E402

PAPER = REPO / "paper"


def main() -> None:
    spec = papernums.load_spec(PAPER / "numbers.toml")
    (PAPER / "numbers.tex").write_text(papernums.render(spec, REPO))
    print(f"wrote paper/numbers.tex ({len(spec.get('n', []))} macros)")


if __name__ == "__main__":
    main()
