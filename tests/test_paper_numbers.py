"""Task 8.3: the manuscript's number check can fail, and passes on the repo."""

import json
from pathlib import Path

import pytest

import papernums

REPO = Path(__file__).resolve().parents[1]


def _toy(tmp_path: Path, value: float = 0.25, prose: str = r"We detect \Rate\%.") -> Path:
    (tmp_path / "r.json").write_text(json.dumps({"a": {"b": value}}))
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "numbers.toml").write_text(
        '[check]\nallow = ["3-of-5"]\n'
        '[[n]]\nname = "Rate"\nsrc = "r.json"\npath = ["a", "b"]\nfmt = "{:.1f}"\nscale = 100\n')
    (paper / "numbers.tex").write_text(papernums.render(papernums.load_spec(paper / "numbers.toml"), tmp_path))
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\usepackage[scale=0.5]{x}\n\\begin{document}\n"
        "\\input{sections/s}\n\\end{document}\n")
    (paper / "sections" / "s.tex").write_text(prose + "\n")
    return paper


def test_render_formats_and_scales(tmp_path):
    paper = _toy(tmp_path)
    assert "\\newcommand{\\Rate}{25.0}" in (paper / "numbers.tex").read_text()


def test_clean_toy_passes(tmp_path):
    paper = _toy(tmp_path, prose=r"We detect \Rate\% under a 3-of-5 rule, see \cref{f:x2} and "
                                 r"\includegraphics[width=0.5\linewidth]{f6.pdf}. % 42 in a comment")
    assert papernums.check(paper, tmp_path) == []


def test_planted_literal_fails(tmp_path):
    paper = _toy(tmp_path, prose=r"We detect \Rate\%, and 97.3\% elsewhere.")
    problems = papernums.check(paper, tmp_path)
    assert len(problems) == 1 and "'97.3'" in problems[0]


def test_changed_json_makes_numbers_stale(tmp_path):
    paper = _toy(tmp_path)
    (tmp_path / "r.json").write_text(json.dumps({"a": {"b": 0.5}}))
    problems = papernums.check(paper, tmp_path)
    assert any("stale" in p for p in problems)


def test_bad_macro_name_rejected(tmp_path):
    spec = {"n": [{"name": "X2", "src": "r.json", "path": ["a"]}]}
    (tmp_path / "r.json").write_text('{"a": 1}')
    with pytest.raises(ValueError):
        papernums.render(spec, tmp_path)


def test_repo_manuscript_passes():
    assert papernums.check(REPO / "paper", REPO) == []


def test_reduce_and_minus(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps(
        {"rows": [{"s": 3600}, {"s": 1800}], "d": {"x": {"v": 1}, "y": {"v": 4}, "z": {"v": 9}},
         "a": 5, "b": 2}))
    v = lambda **e: papernums.value({"src": "r.json", **e}, tmp_path)  # noqa: E731
    assert v(path=["rows"], each=["s"], reduce="sum", scale=1 / 3600, fmt="{:.1f}") == "1.5"
    assert v(path=["d"], each=["v"], reduce="max") == "9"
    assert v(path=["d"], keys=["x", "y"], each=["v"], reduce="max") == "4"
    assert v(path=["a"], minus=["b"]) == "3"
    assert v(path=["d"], each=["v"], reduce="sum", minus={"path": ["a"]}) == "9"
