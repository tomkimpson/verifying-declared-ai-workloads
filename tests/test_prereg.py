"""The test-split lock (spec task 3.2)."""

import pytest

import declaration as d
import prereg


def test_require_passes_on_matching_hash(tmp_path):
    f = tmp_path / "prereg.md"
    f.write_text("frozen\n")
    prereg.require(f, expected=prereg.sha256(f))


def test_require_fails_when_file_changes(tmp_path):
    f = tmp_path / "prereg.md"
    f.write_text("frozen\n")
    h = prereg.sha256(f)
    f.write_text("frozen, then edited\n")
    with pytest.raises(RuntimeError, match="does not match"):
        prereg.require(f, expected=h)


def test_require_fails_when_file_missing(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        prereg.require(tmp_path / "absent.md", expected="0" * 64)


def test_loading_test_fails_with_hash_changed(monkeypatch):
    """The 3.2 verification: a changed constant locks the test split."""
    monkeypatch.setattr(prereg, "PREREG_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="test split is locked"):
        d.load_windows("test")


def test_loading_test_fails_when_unfrozen(monkeypatch):
    monkeypatch.setattr(prereg, "PREREG_SHA256", None)
    with pytest.raises(RuntimeError, match="not frozen"):
        d.load_windows("test")


def test_other_splits_are_not_locked(monkeypatch):
    monkeypatch.setattr(prereg, "PREREG_SHA256", "0" * 64)
    header, _ = d.load_windows("calibration")
    assert header["split"] == "calibration"


def test_frozen_hash_matches_the_preregistration():
    """The committed constant unlocks the committed file, and only it."""
    prereg.require()
