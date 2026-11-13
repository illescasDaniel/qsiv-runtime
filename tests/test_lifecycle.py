"""S1, S2, S8, S9, S10, S17, S18, S28: starting, attaching, restarting and timing out."""

from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path

import pytest

from qsiv_runtime import Cause, QSIVClient, QSIVUnavailable
from qsiv_runtime.errors import EXIT_MODEL_IN_USE
from tests.conftest import Install
from tests.fakes.fake_clock import FakeClock
from tests.fakes.fake_machine import FakeMachine
from tests.fakes.fake_service import FakeEngine, Generation, ScriptedLauncher, ThreadServiceLauncher
from tests.helpers import QUESTIONS, make_client

FOUR_LOW = [0.01, 0.01, 0.01, 0.01]


def test_given_no_service_when_deciding_then_it_starts_once_and_returns_validated_answers(
    install: Install, sock_dir: Path
) -> None:
    """S1"""
    # given
    engine = FakeEngine([0.02, 0.11, 0.03, 0.01])
    launcher = ThreadServiceLauncher(engine)
    client = make_client(install, sock_dir, launcher)
    # when
    answers = client.decide("state", QUESTIONS)
    # then
    assert answers == [0.02, 0.11, 0.03, 0.01]
    assert (launcher.starts, launcher.loads) == (1, 1)
    request = engine.requests[0]
    assert request["state"] == "state"
    assert request["questions"] == [{"type": "noul", "text": q} for q in QUESTIONS]
    with open(sock_dir / "startup.lock", "wb") as handle:  # the startup lock was released
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    client.close()
    launcher.stop()


def test_given_a_new_client_is_created_when_nothing_is_called_then_nothing_happens(
    install: Install, sock_dir: Path
) -> None:
    """Lazy attach: no socket, lock or process until the first decide()."""
    # given / when
    launcher = ThreadServiceLauncher()
    make_client(install, sock_dir, launcher)
    # then
    assert launcher.starts == 0
    assert list(sock_dir.iterdir()) == []


def test_given_a_resident_service_when_deciding_with_a_new_client_then_no_second_load(
    install: Install, sock_dir: Path
) -> None:
    """S2 (attach) and S17 (two clients share one service)."""
    # given
    launcher = ThreadServiceLauncher(FakeEngine([0.04, 0.62, 0.01, 0.02]))
    first = make_client(install, sock_dir, launcher)
    second = make_client(install, sock_dir, launcher)
    first.decide("a", QUESTIONS)
    # when
    answers = second.decide("b", QUESTIONS)
    first_again = first.decide("c", QUESTIONS)
    # then
    assert answers == first_again == [0.04, 0.62, 0.01, 0.02]
    assert (launcher.starts, launcher.loads) == (1, 1)
    first.close()
    second.close()
    launcher.stop()


def test_given_requests_spaced_inside_the_idle_window_when_deciding_then_the_timer_restarts(
    install: Install, sock_dir: Path
) -> None:
    """S2: each request restarts the idle timer (three requests span more than one idle period)."""
    # given
    launcher = ThreadServiceLauncher(idle_seconds=1.0)
    client = make_client(install, sock_dir, launcher)
    # when
    for _ in range(3):
        client.decide("x", QUESTIONS)
        threading.Event().wait(0.6)
    # then
    assert (launcher.starts, client.restarts) == (1, 0)
    client.close()
    launcher.stop()


def test_given_two_clients_starting_together_when_both_decide_then_one_service_one_load(
    install: Install, sock_dir: Path
) -> None:
    """S18"""
    # given
    launcher = ThreadServiceLauncher(load_delay=0.4)
    clients = [make_client(install, sock_dir, launcher) for _ in range(2)]
    barrier = threading.Barrier(2)
    results: list[list[float]] = []

    def run(index: int) -> None:
        barrier.wait()
        results.append(clients[index].decide("x", QUESTIONS))

    # when
    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    # then
    assert len(results) == 2
    assert (launcher.starts, launcher.loads) == (1, 1)
    for client in clients:
        client.close()
    launcher.stop()


def test_given_a_service_that_never_becomes_ready_when_120_s_pass_then_startup_failed(
    install: Install, sock_dir: Path
) -> None:
    """S8: fake clock, lock released, a second attempt starts fresh."""
    # given
    clock = FakeClock()
    launcher = ScriptedLauncher([Generation(startup="never"), Generation(replies=[FOUR_LOW])])
    client = make_client(install, sock_dir, launcher, clock=clock)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.STARTUP_FAILED
    assert exc.message.startswith("the QSIV service did not become ready within 120 s")
    assert exc.diagnostic.log_path == str(sock_dir / "service.log")
    assert str(sock_dir / "service.log") in exc.message
    assert clock.slept >= 119.9
    with open(sock_dir / "startup.lock", "wb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert client.decide("x", QUESTIONS) == FOUR_LOW  # a fresh start
    assert launcher.starts == 2
    client.close()
    launcher.stop()


def test_given_a_service_that_crashes_during_start_when_deciding_then_startup_failed(
    install: Install, sock_dir: Path
) -> None:
    """S8 (variant: exits during start)."""
    # given
    launcher = ScriptedLauncher([Generation(startup="crash")])
    client = make_client(install, sock_dir, launcher, clock=FakeClock())
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.STARTUP_FAILED
    assert launcher.starts == 1


def test_given_the_gpu_slot_is_taken_when_the_service_exits_during_start_then_model_in_use(
    install: Install, sock_dir: Path
) -> None:
    """S11 / F5 (agemem's model.lock, Q13): the service exits with its dedicated code."""
    # given
    launcher = ScriptedLauncher([Generation(startup="crash", exit_code=EXIT_MODEL_IN_USE)])
    client = make_client(install, sock_dir, launcher, clock=FakeClock())
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.MODEL_IN_USE
    assert "another QSIV model is already loaded" in caught.value.message


def test_given_a_lock_file_that_cannot_be_opened_when_deciding_then_a_bounded_error_not_oserror(
    install: Install, sock_dir: Path
) -> None:
    """Contract: never a raw traceback (a symlinked startup.lock fails with ELOOP)."""
    # given
    (sock_dir / "startup.lock").symlink_to(sock_dir / "elsewhere")
    launcher = ThreadServiceLauncher()
    client = make_client(install, sock_dir, launcher)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.STARTUP_FAILED
    assert launcher.starts == 0


def test_given_a_service_that_accepts_but_never_answers_the_handshake_then_startup_failed(
    install: Install, sock_dir: Path
) -> None:
    """S8 (variant: the ready line never arrives); a real, short startup timeout."""
    # given
    launcher = ScriptedLauncher([Generation(startup="silent")])
    client = make_client(install, sock_dir, launcher, startup_timeout=0.5)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.STARTUP_FAILED
    launcher.stop()


def test_given_the_service_exits_during_a_request_when_deciding_then_one_restart_succeeds(
    install: Install, sock_dir: Path
) -> None:
    """S9 / F4"""
    # given
    launcher = ScriptedLauncher([Generation(replies=["close"]), Generation(replies=[FOUR_LOW])])
    client = make_client(install, sock_dir, launcher)
    # when
    answers = client.decide("x", QUESTIONS)
    # then
    assert answers == FOUR_LOW
    assert client.restarts == 1
    assert launcher.starts == 2
    assert launcher.requests[0] == launcher.requests[1]  # the same request was retried
    client.close()
    launcher.stop()


def test_given_the_restarted_service_also_exits_when_deciding_then_service_exited(
    install: Install, sock_dir: Path
) -> None:
    """S10 / F4: no third attempt."""
    # given
    launcher = ScriptedLauncher([Generation(replies=["close"]), Generation(replies=["close"])])
    client = make_client(install, sock_dir, launcher)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.SERVICE_EXITED
    assert exc.message.startswith("the QSIV service exited during the review twice")
    assert str(sock_dir / "service.log") in exc.message
    assert (launcher.starts, client.restarts) == (2, 1)
    launcher.stop()


def test_given_a_silent_service_when_the_request_timeout_passes_then_timeout_without_restart(
    install: Install, sock_dir: Path
) -> None:
    """S28 / F10 (F4 must not retry a timeout)."""
    # given
    launcher = ScriptedLauncher([Generation(replies=["hang"]), Generation(replies=[FOUR_LOW])])
    client = make_client(install, sock_dir, launcher, request_timeout=0.3)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.TIMEOUT
    assert exc.message.startswith("the QSIV service did not answer within 0.3 s")
    assert (launcher.starts, client.restarts) == (1, 0)
    launcher.stop()


def test_given_the_default_client_then_the_documented_timeouts_apply(
    install: Install, sock_dir: Path
) -> None:
    """Startup 120 s, request 60 s: the message for the default shows 60 s."""
    from qsiv_runtime import client as client_module

    assert client_module.DEFAULT_STARTUP_TIMEOUT == 120.0
    assert client_module.DEFAULT_REQUEST_TIMEOUT == 60.0
    assert client_module.timeout_message(60.0) == "the QSIV service did not answer within 60 s"
    assert os.path.isdir(sock_dir)


def test_given_an_invalid_runtime_in_the_environment_when_deciding_then_qsiv_unavailable(
    install: Install, sock_dir: Path
) -> None:
    """Q2: a bad QSIV_RUNTIME_RUNTIME is a QSIVUnavailable, not a raw ValueError."""
    # given
    client = QSIVClient(
        install.model,
        install.python,
        socket_directory=sock_dir,
        environ={"PATH": "/usr/bin", "QSIV_RUNTIME_RUNTIME": "bogus"},
        machine=FakeMachine(),
        launcher=ThreadServiceLauncher(),
    )
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("state", QUESTIONS)
    # then
    assert "QSIV_RUNTIME_RUNTIME" in caught.value.message
    assert "fp8" in caught.value.message and "reference" in caught.value.message
    assert caught.value.diagnostic.cause is Cause.STARTUP_FAILED
