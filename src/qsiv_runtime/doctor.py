"""Read-only diagnostics: no downloads, no model load, no inference."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from . import _preflight
from ._pins import QSIV_SHA256, WEIGHTS_FP8
from ._ports import Machine, SystemMachine
from .errors import Cause, QSIVUnavailable
from .paths import Runtime, resolve_paths


@dataclass(frozen=True)
class DoctorCheck:
    name: str  # platform, gpu, model directory, runtime, socket directory
    ok: bool
    detail: str
    cause: Cause | None = None


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _run(name: str, check: Callable[[], str]) -> DoctorCheck:
    try:
        return DoctorCheck(name, True, check())
    except QSIVUnavailable as exc:
        hint = _preflight.tool_hint(exc.diagnostic)
        detail = exc.message + (f". {hint}" if hint else "")
        return DoctorCheck(name, False, detail, exc.diagnostic.cause)


def doctor(
    model_directory: str | os.PathLike[str] | None = None,
    python: str | os.PathLike[str] | None = None,
    runtime: Runtime | None = None,
    *,
    verify: bool = False,
    machine: Machine | None = None,
    environ: Mapping[str, str] | None = None,
) -> DoctorReport:
    """One entry per check. `verify` also hashes the FP8 weights against the pinned SHA-256."""
    machine = machine or SystemMachine()
    try:
        paths = resolve_paths(model_directory, python, runtime, environ=environ)
    except ValueError as exc:
        invalid = DoctorCheck(
            "runtime", False, f"invalid QSIV runtime setting ({exc})", Cause.STARTUP_FAILED
        )
        return DoctorReport((invalid,))

    def platform_check() -> str:
        _preflight.check_platform(machine, paths.runtime)
        return f"{machine.system()} {machine.machine()}"

    def model_check() -> str:
        _preflight.check_artifacts(paths)
        _preflight.check_marker(paths)
        if verify and paths.runtime == "fp8":
            from .setup import checksum_mismatch, sha256_file

            actual = sha256_file(paths.model_directory / WEIGHTS_FP8)
            if actual != QSIV_SHA256:
                raise checksum_mismatch(QSIV_SHA256, actual)
        suffix = " (legacy agemem location)" if paths.legacy else ""
        return f"{paths.model_directory}{suffix}"

    def runtime_check() -> str:
        _preflight.check_python(paths)
        return f"{paths.python} ({paths.runtime})"

    def socket_check() -> str:
        _preflight.check_private_directory(paths.socket_directory)
        return str(paths.socket_directory)

    return DoctorReport(
        (
            _run("platform", platform_check),
            _run("gpu", lambda: _preflight.check_gpu(machine, paths.runtime)),
            _run("model directory", model_check),
            _run("runtime", runtime_check),
            _run("socket directory", socket_check),
        )
    )
