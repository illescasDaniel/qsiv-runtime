"""The release workflow prints the CHANGELOG section for the tag; the first entry must exist."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load() -> object:
    spec = importlib.util.spec_from_file_location(
        "release_notes", ROOT / "scripts/release_notes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_given_the_changelog_when_asking_for_the_first_release_then_its_notes_are_returned() -> (
    None
):
    # given
    module = _load()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    # when
    notes = module.section(changelog, "0.1.0")  # type: ignore[attr-defined]
    # then
    assert "QSIVClient.decide" in notes
    assert "## [0.1.0]" in changelog


def test_given_an_unknown_version_when_asking_for_notes_then_lookup_error() -> None:
    # given / when / then
    with pytest.raises(LookupError):
        _load().section((ROOT / "CHANGELOG.md").read_text(), "9.9.9")  # type: ignore[attr-defined]
