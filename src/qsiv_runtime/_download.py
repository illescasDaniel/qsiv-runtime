"""Runs inside the GPU virtualenv (where huggingface_hub is installed): fetch the pinned snapshot.

This is the only code that downloads, and only `qsiv-runtime setup` runs it."""

from __future__ import annotations

import sys
from pathlib import Path

from ._pins import QSIV_REVISION

ALLOW_PATTERNS = [
    "qsiv/*",
    "tokenizer/*",
    "pyproject.toml",
    "LICENSE",
    "NOTICE",
    "config.json",
    "calibration.json",
    "qsiv_fp8.safetensors",
    "qsiv_fp8.safetensors.sha256",
]


def download(directory: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        "djinn-san/QSIV",
        revision=QSIV_REVISION,
        local_dir=str(directory),
        allow_patterns=ALLOW_PATTERNS,
    )


if __name__ == "__main__":
    download(Path(sys.argv[1]))
