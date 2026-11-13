"""JSON lines, one object per line, UTF-8, and the strict validation of a service answer."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypedDict

from .errors import Cause, QSIVDiagnostic, QSIVError, QSIVUnavailable
from .paths import Runtime

PROTOCOL_VERSION: Final = 1

# client -> service   {"protocol": 1, "config": [python, model_directory, runtime]}
# service -> client   {"ready": true}  |  {"error": str, "diagnostic": {"cause": ..., ...}}
# client -> service   {"request": {"state": str, "questions": [WireQuestion, ...]}}
# service -> client   {"answers": [{"p_true": float}, ...]}  |  {"error": str, "diagnostic": {...}}


@dataclass(frozen=True)
class ServiceConfig:
    python: str  # not resolved: its path selects the virtualenv
    model_directory: str  # resolved
    runtime: Runtime

    def as_wire(self) -> list[str]:
        return [self.python, self.model_directory, self.runtime]


class WireQuestion(TypedDict):
    type: Literal["noul"]
    text: str


class ProtocolError(QSIVError):
    """A line that is not one JSON object."""


def encode(message: Mapping[str, object]) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode()


def decode(line: bytes) -> dict[str, object]:
    try:
        value = json.loads(line)
    except ValueError as exc:  # includes UnicodeDecodeError
        raise ProtocolError("malformed message") from exc
    if not isinstance(value, dict):
        raise ProtocolError("message is not an object")
    return value


def invalid_answer(expected: int, got: str) -> QSIVUnavailable:
    return QSIVUnavailable(
        f"QSIV returned an invalid answer (expected {expected} probabilities in [0, 1], got {got})",
        QSIVDiagnostic(Cause.INVALID_ANSWER, "Retry; `qsiv-runtime doctor` if it repeats"),
    )


def validate_probabilities(answers: object, expected: int) -> list[float]:
    """Whole-answer check: a list of the expected length; each p_true an int or float, not bool,
    finite, within [0, 1]. One bad value rejects everything (S12)."""
    if not isinstance(answers, list):
        raise invalid_answer(expected, "no list")
    if len(answers) != expected:
        raise invalid_answer(expected, str(len(answers)))
    result: list[float] = []
    for answer in answers:
        value = answer.get("p_true") if isinstance(answer, dict) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise invalid_answer(expected, "an invalid value")
        result.append(float(value))
    return result
