"""Failure types: every failure of the client is a QSIVUnavailable with a cause (fail closed)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

EXIT_MODEL_IN_USE = 3  # exit code of `python -m qsiv_runtime.service` when the GPU slot is taken


class Cause(str, Enum):
    NO_GPU = "no_gpu"  # F1
    NOT_INSTALLED = "not_installed"  # F2
    STARTUP_FAILED = "startup_failed"  # F3
    SERVICE_EXITED = "service_exited"  # F4
    MODEL_IN_USE = "model_in_use"  # F5
    INVALID_ANSWER = "invalid_answer"  # F6
    UNSAFE_DIRECTORY = "unsafe_directory"  # F7
    BAD_CHECKSUM = "bad_checksum"  # F8
    TOO_LARGE = "too_large"  # F9
    TIMEOUT = "timeout"  # F10


@dataclass(frozen=True)
class QSIVDiagnostic:
    cause: Cause
    hint: str | None = None  # "Run: qsiv-runtime setup" and similar
    log_path: str | None = None  # private service.log, when relevant
    detail: Mapping[str, str] = field(default_factory=dict)  # gpu, capability, resident, requested


class QSIVError(Exception):
    """Base class of every error this package raises on purpose."""


class QSIVUnavailable(QSIVError):
    """Fail closed: the review did not happen.

    `message` is the text after "Reviewer unavailable: "."""

    def __init__(self, message: str, diagnostic: QSIVDiagnostic) -> None:
        super().__init__(message)
        self.message = message
        self.diagnostic = diagnostic
