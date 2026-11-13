"""Checks shared by the client and doctor: platform, GPU, install, marker, socket directory."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ._pins import QSIV_REVISION, QSIV_SHA256, WEIGHTS_FP8, WEIGHTS_REFERENCE
from ._ports import Machine
from .errors import Cause, QSIVDiagnostic, QSIVUnavailable
from .paths import INSTALL_MARKER, LEGACY_INSTALL_MARKER, ResolvedPaths

SETUP_HINT = "Run: qsiv-runtime setup"
_MINIMUM = {"fp8": (8, 9), "reference": (8, 0)}


def weights_name(runtime: str) -> str:
    return WEIGHTS_FP8 if runtime == "fp8" else WEIGHTS_REFERENCE


DOCTOR_HINT = "Run `qsiv-runtime doctor`, or use another provider"


def tool_hint(diagnostic: QSIVDiagnostic) -> str | None:
    """The hint for doctor/setup output: no self-referential "run doctor" advice."""
    if diagnostic.hint == DOCTOR_HINT:
        return "Or use another provider"
    return diagnostic.hint


def _no_gpu(runtime: str, found: str) -> QSIVUnavailable:
    capability = "8.9 or higher (FP8)" if runtime == "fp8" else "8.0 or higher"
    return QSIVUnavailable(
        "QSIV needs Linux x86_64 and an NVIDIA GPU with compute capability "
        f"{capability}; found: {found}",
        QSIVDiagnostic(Cause.NO_GPU, DOCTOR_HINT),
    )


def check_platform(machine: Machine, runtime: str) -> None:
    if machine.system() != "Linux" or machine.machine().lower() not in ("x86_64", "amd64"):
        raise _no_gpu(runtime, f"{machine.system()} {machine.machine()}")


def check_gpu(machine: Machine, runtime: str) -> str:
    """Raise F1 unless a GPU meets the runtime's minimum; return a one-line description."""
    gpus = machine.gpus()
    if not gpus:
        raise _no_gpu(runtime, "no NVIDIA GPU")
    best = max(gpus, key=lambda gpu: gpu.compute_capability)
    major, minor = best.compute_capability
    if best.compute_capability < _MINIMUM[runtime]:
        raise _no_gpu(runtime, f"{best.name} ({major}.{minor})")
    return f"{best.name} ({major}.{minor})"


def check_machine(machine: Machine, runtime: str) -> None:
    check_platform(machine, runtime)
    check_gpu(machine, runtime)


def _not_installed(paths: ResolvedPaths) -> QSIVUnavailable:
    checked = ", ".join(str(path) for path in paths.checked or (paths.model_directory,))
    return QSIVUnavailable(
        "QSIV is not installed",
        QSIVDiagnostic(
            Cause.NOT_INSTALLED, f"{SETUP_HINT} (checked: {checked})", None, {"checked": checked}
        ),
    )


def check_artifacts(paths: ResolvedPaths) -> None:
    model = paths.model_directory
    required = ("config.json", "calibration.json", "tokenizer", weights_name(paths.runtime))
    if not model.is_dir() or any(not (model / name).exists() for name in required):
        raise _not_installed(paths)


def check_python(paths: ResolvedPaths) -> None:
    if not paths.python.is_file():
        raise _not_installed(paths)


def _marker_matches(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and data.get("revision") == QSIV_REVISION
        and data.get("weights_sha256") == QSIV_SHA256
    )


def bad_checksum() -> QSIVUnavailable:
    return QSIVUnavailable(
        "QSIV runtime is not installed (setup did not complete)",
        QSIVDiagnostic(Cause.BAD_CHECKSUM, SETUP_HINT),
    )


def check_marker(paths: ResolvedPaths) -> None:
    """The install marker must carry the pinned revision and checksum (F8). The agemem marker
    counts too. Only the FP8 weights are pinned, so the reference runtime has no marker check."""
    if paths.runtime != "fp8":
        return
    model = paths.model_directory
    if not any(_marker_matches(model / name) for name in (INSTALL_MARKER, LEGACY_INSTALL_MARKER)):
        raise bad_checksum()


def check_install(paths: ResolvedPaths) -> None:
    check_artifacts(paths)
    check_python(paths)
    check_marker(paths)


def _problem(path: Path, uid: int) -> str | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if not stat.S_ISDIR(info.st_mode):
        return "not a directory"
    if info.st_uid != uid:
        return "owned by another user"
    if info.st_mode & 0o077:
        return f"mode {info.st_mode & 0o777:04o}"
    return None


def check_private_directory(path: Path, uid: int | None = None) -> None:
    """Raise F7 when the directory exists and is a symlink, foreign or not mode 0700."""
    problem = _problem(path, getattr(os, "getuid", lambda: 0)() if uid is None else uid)
    if problem is not None:
        raise QSIVUnavailable(
            f"QSIV socket directory {path} is not private ({problem}, expected 0700)",
            QSIVDiagnostic(Cause.UNSAFE_DIRECTORY, "Remove or fix the directory"),
        )


def ensure_private_directory(path: Path, uid: int | None = None) -> Path:
    """Create the directory with mode 0700 if missing; refuse an unsafe one (F7)."""
    check_private_directory(path, uid)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_private_directory(path, uid)
    return path
