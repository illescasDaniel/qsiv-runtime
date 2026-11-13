from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from qsiv_runtime.paths import INSTALL_MARKER
from qsiv_runtime.setup import QSIV_REVISION, QSIV_SHA256


@dataclass
class Install:
    model: Path
    python: Path


def make_install(root: Path, *, marker: str | None = INSTALL_MARKER) -> Install:
    """A fake installed model directory and runtime interpreter (no real weights)."""
    model = root / "model"
    (model / "tokenizer").mkdir(parents=True)
    for name in ("config.json", "calibration.json", "qsiv_fp8.safetensors"):
        (model / name).write_text("{}")
    if marker is not None:
        (model / marker).write_text(
            json.dumps({"revision": QSIV_REVISION, "weights_sha256": QSIV_SHA256})
        )
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    return Install(model, python)


@pytest.fixture
def install(tmp_path: Path) -> Install:
    return make_install(tmp_path)


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    # AF_UNIX paths are limited to ~108 bytes, so not pytest's tmp_path; mkdtemp creates mode 0700.
    path = Path(tempfile.mkdtemp(prefix="qsv-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)
