"""Explicit installation: the only code that downloads. Inference never does."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _preflight
from ._launch import stage_package
from ._pins import QSIV_REVISION, QSIV_SHA256, WEIGHTS_FP8
from ._ports import CommandRunner, Machine, SubprocessRunner, SystemMachine
from .errors import Cause, QSIVDiagnostic, QSIVError, QSIVUnavailable
from .paths import ENV_MODEL_DIRECTORY, INSTALL_MARKER, data_directory

__all__ = ["QSIV_REVISION", "QSIV_SHA256", "SetupError", "SetupResult", "setup"]

LOCK_FILE = Path(__file__).with_name("data") / "qsiv-linux-py312.lock"
_NUMPY_PINS = ("numpy==2.4.6", "numpy==2.5.4")


class SetupError(QSIVError):
    """Setup could not finish (missing uv, unexpected download); nothing was marked installed."""


@dataclass(frozen=True)
class SetupResult:
    model_directory: Path
    runtime_python: Path
    revision: str
    weights_sha256: str
    dry_run: bool = False
    commands: tuple[tuple[str, ...], ...] = ()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_mismatch(expected: str, actual: str) -> QSIVUnavailable:
    return QSIVUnavailable(
        f"Checksum mismatch for {WEIGHTS_FP8} (expected {expected}, got {actual}); "
        "the runtime is NOT installed",
        QSIVDiagnostic(Cause.BAD_CHECKSUM, _preflight.SETUP_HINT),
    )


def setup(
    model_directory: str | os.PathLike[str] | None = None,
    venv_directory: str | os.PathLike[str] | None = None,
    *,
    dry_run: bool = False,
    machine: Machine | None = None,
    runner: CommandRunner | None = None,
) -> SetupResult:
    """Create the pinned GPU environment and download the pinned weights.

    Refuses before any command on an unsupported machine (F1); a checksum mismatch discards the
    file and writes no install marker (F8). Re-running reuses the environment; nothing is
    deleted."""
    machine = machine or SystemMachine()
    _preflight.check_platform(machine, "fp8")
    data = data_directory() / "qsiv-runtime"
    explicit = model_directory or os.environ.get(ENV_MODEL_DIRECTORY)
    model = Path(os.path.abspath(os.path.expanduser(explicit))) if explicit else data / "model"
    venv = (
        Path(os.path.abspath(os.path.expanduser(venv_directory)))
        if venv_directory
        else data / "venv"
    )
    python = venv / "bin" / "python"
    before: list[tuple[str, ...]] = []
    if not python.exists():
        before.append(("uv", "venv", "--python", "3.12", str(venv)))
    before.append(("uv", "pip", "install", "--python", str(python), "-r", str(LOCK_FILE)))
    before.append((str(python), "-m", "qsiv_runtime._download", str(model)))
    after = [
        (
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-build-isolation",
            "--no-deps",
            str(model),
        ),
        ("uv", "pip", "check", "--python", str(python)),
    ]
    if dry_run:
        return SetupResult(model, python, QSIV_REVISION, QSIV_SHA256, True, tuple(before + after))
    _preflight.check_gpu(machine, "fp8")
    if machine.which("uv") is None:
        raise SetupError("QSIV setup needs uv; install uv and retry")
    runner = runner or SubprocessRunner()
    # A failed re-run must not leave a stale "installed" marker next to partly replaced files.
    (model / INSTALL_MARKER).unlink(missing_ok=True)
    venv.parent.mkdir(parents=True, exist_ok=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    # The venv has no qsiv_runtime installed: the download module runs through PYTHONPATH.
    with tempfile.TemporaryDirectory(prefix="qsiv-runtime-") as scratch:
        download_env = dict(os.environ)
        staged = str(stage_package(Path(scratch)))
        download_env["PYTHONPATH"] = os.pathsep.join(
            [staged, download_env["PYTHONPATH"]] if download_env.get("PYTHONPATH") else [staged]
        )
        for command in before:
            runner.run(command, download_env if "qsiv_runtime._download" in command else None)
    _verify_download(model)
    for command in after:
        runner.run(command)
    marker = {
        "provider": "qsiv",
        "directory": str(model),
        "runtime_python": str(python),
        "revision": QSIV_REVISION,
        "weights_sha256": QSIV_SHA256,
    }
    (model / INSTALL_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    return SetupResult(model, python, QSIV_REVISION, QSIV_SHA256, False, tuple(before + after))


def _verify_download(model: Path) -> None:
    """Checksum the weights (discard them on a mismatch), then adjust the NumPy pin."""
    weights = model / WEIGHTS_FP8
    if not weights.is_file():
        raise SetupError(
            f"the download did not produce {WEIGHTS_FP8}; the runtime is NOT installed"
        )
    actual = sha256_file(weights)
    if actual != QSIV_SHA256:
        weights.unlink()
        raise checksum_mismatch(QSIV_SHA256, actual)
    metadata = model / "pyproject.toml"
    text = metadata.read_text() if metadata.is_file() else ""
    if not any(pin in text for pin in _NUMPY_PINS):
        raise SetupError("unexpected QSIV metadata; inspect it before installing")
    metadata.write_text(text.replace("numpy==2.4.6", "numpy==2.5.4"))
