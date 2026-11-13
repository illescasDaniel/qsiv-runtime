"""S11, S12, S16: model_in_use, strict answer validation and the too-large refusal."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qsiv_runtime import Cause, QSIVUnavailable
from qsiv_runtime._protocol import validate_probabilities
from tests.conftest import Install, make_install
from tests.fakes.fake_service import FakeEngine, ThreadServiceLauncher
from tests.helpers import QUESTIONS, make_client


def test_given_a_different_model_is_resident_when_deciding_then_model_in_use_and_service_untouched(
    install: Install, tmp_path: Path, sock_dir: Path
) -> None:
    """S11 / F5"""
    # given
    launcher = ThreadServiceLauncher(FakeEngine([0.1] * 4))
    resident = make_client(install, sock_dir, launcher)
    resident.decide("x", QUESTIONS)
    other = make_install(tmp_path / "other")
    (other.model / "model.safetensors").write_text("{}")
    requester = make_client(other, sock_dir, launcher, runtime="reference")
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        requester.decide("x", QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.MODEL_IN_USE
    assert f"fp8 at {install.model}" in exc.message  # resident
    assert f"reference at {other.model}" in exc.message  # requested
    assert exc.diagnostic.hint == "Use the same settings, or wait for idle exit (300 s)"
    assert resident.decide("y", QUESTIONS) == [0.1] * 4  # not stopped
    assert (launcher.starts, launcher.loads) == (1, 1)
    resident.close()
    launcher.stop()


def _decide_with(install: Install, sock_dir: Path, probabilities: list[object]) -> QSIVUnavailable:
    launcher = ThreadServiceLauncher(FakeEngine(probabilities))
    client = make_client(install, sock_dir, launcher)
    try:
        with pytest.raises(QSIVUnavailable) as caught:
            client.decide("x", QUESTIONS)
        return caught.value
    finally:
        client.close()
        launcher.stop()


def test_given_three_values_when_deciding_then_invalid_answer_naming_the_count(
    install: Install, sock_dir: Path
) -> None:
    """S12 / F6 (count)"""
    # given / when
    exc = _decide_with(install, sock_dir, [0.1, 0.2, 0.3])
    # then
    assert exc.diagnostic.cause is Cause.INVALID_ANSWER
    assert (
        exc.message == "QSIV returned an invalid answer (expected 4 probabilities in [0, 1], got 3)"
    )


@pytest.mark.parametrize(
    "bad",
    [1.2, -0.1, math.nan, math.inf, "0.5", True, None, [0.5]],
    ids=["above", "below", "nan", "inf", "string", "bool", "none", "list"],
)
def test_given_one_bad_value_among_valid_ones_when_deciding_then_the_whole_answer_is_rejected(
    install: Install, sock_dir: Path, bad: object
) -> None:
    """S12 / F6 (range and type), even though the other values are valid."""
    # given / when
    exc = _decide_with(install, sock_dir, [0.1, bad, 0.2, 0.3])
    # then
    assert exc.diagnostic.cause is Cause.INVALID_ANSWER
    assert exc.message.startswith("QSIV returned an invalid answer (expected 4 probabilities")


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, 0.5, 1e-9])
def test_given_a_valid_number_when_validating_then_it_is_accepted_as_a_float(
    value: float,
) -> None:
    """S12: int and float in [0, 1] are fine (and become floats)."""
    # given / when
    result = validate_probabilities([{"p_true": value}], 1)
    # then
    assert result == [float(value)]
    assert isinstance(result[0], float)


@pytest.mark.parametrize(
    "answers",
    [None, {"p_true": 0.5}, "x", [0.5], [{"p": 0.5}], [{"p_true": 0.5}, {"p_true": 0.5}], []],
    ids=["none", "dict", "str", "bare-number", "wrong-key", "too-many", "empty"],
)
def test_given_a_malformed_answer_shape_when_validating_then_invalid_answer(
    answers: object,
) -> None:
    """S12: the shape of the whole answer is checked, not only the numbers."""
    # given / when
    with pytest.raises(QSIVUnavailable) as caught:
        validate_probabilities(answers, 1)
    # then
    assert caught.value.diagnostic.cause is Cause.INVALID_ANSWER


def test_given_a_too_long_script_when_deciding_then_too_large_and_never_truncated(
    install: Install, sock_dir: Path
) -> None:
    """S16 / F9: the limit lives in the service; the client never shortens the input."""
    # given
    engine = FakeEngine([0.1] * 4, limit=20)
    launcher = ThreadServiceLauncher(engine)
    client = make_client(install, sock_dir, launcher)
    script = "x" * 21
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide(script, QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.TOO_LARGE
    assert exc.message == (
        "this script is too long for the QSIV model, so it was not reviewed (it is never truncated)"
    )
    assert [r["state"] for r in engine.requests] == [script]  # full text, nothing shortened
    assert client.decide("x" * 20, QUESTIONS) == [0.1] * 4  # the connection is still usable
    client.close()
    launcher.stop()


@pytest.mark.parametrize("count", [0, 9])
def test_given_zero_or_too_many_questions_when_deciding_then_value_error(
    install: Install, sock_dir: Path, count: int
) -> None:
    """The engine takes 1 to 8 questions per call; anything else is a caller bug, not F1..F10."""
    # given
    launcher = ThreadServiceLauncher()
    client = make_client(install, sock_dir, launcher)
    # when / then
    with pytest.raises(ValueError, match="1 to 8"):
        client.decide("x", ["q"] * count)
    assert launcher.starts == 0
