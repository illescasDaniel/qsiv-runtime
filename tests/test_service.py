"""The resident service itself: handshake, error mapping, idle exit, model lock."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from qsiv_runtime import Cause, QSIVDiagnostic, QSIVUnavailable
from qsiv_runtime._protocol import PROTOCOL_VERSION, ServiceConfig
from qsiv_runtime.service import Engine, _lock_model, main, serve
from tests.fakes.fake_service import FakeEngine

CONFIG = ServiceConfig("/venv/bin/python", "/model", "fp8")


class RunningService:
    def __init__(self, sock_dir: Path, engine: Engine, idle_seconds: float = 30.0) -> None:
        self.sock_dir, self.stop = sock_dir, threading.Event()
        self.thread = threading.Thread(
            target=serve,
            args=(CONFIG, lambda: engine, idle_seconds),
            kwargs={"socket_directory": sock_dir, "stop": self.stop},
            daemon=True,
        )
        self.thread.start()
        deadline = time.monotonic() + 5
        while not (sock_dir / "service.sock").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        # The public socket name only appears once the service is listening (atomic rename).

    def connect(self) -> tuple[socket.socket, socket.SocketIO]:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(str(self.sock_dir / "service.sock"))
        return sock, sock.makefile("rwb")  # type: ignore[return-value]

    @staticmethod
    def exchange(stream: socket.SocketIO, message: object) -> dict[str, object]:
        stream.write((json.dumps(message) + "\n").encode())
        stream.flush()
        reply = json.loads(stream.readline())
        assert isinstance(reply, dict)
        return reply


@pytest.fixture
def service(sock_dir: Path) -> Iterator[RunningService]:
    running = RunningService(sock_dir, FakeEngine([0.5] * 2))
    yield running
    running.stop.set()
    running.thread.join(timeout=5)


def hello() -> dict[str, object]:
    return {"protocol": PROTOCOL_VERSION, "config": CONFIG.as_wire()}


def test_given_a_service_when_connecting_then_the_socket_is_private(
    service: RunningService, sock_dir: Path
) -> None:
    # given / when
    mode = (sock_dir / "service.sock").stat().st_mode & 0o777
    # then
    assert mode == 0o600


def test_given_a_matching_hello_when_requesting_then_answers_follow(
    service: RunningService,
) -> None:
    # given
    sock, stream = service.connect()
    # when
    ready = service.exchange(stream, hello())
    answers = service.exchange(stream, {"request": {"state": "s", "questions": []}})
    # then
    assert ready == {"ready": True}
    assert answers == {"answers": [{"p_true": 0.5}, {"p_true": 0.5}]}
    stream.close()
    sock.close()


def test_given_another_protocol_version_when_connecting_then_an_error_is_returned(
    service: RunningService,
) -> None:
    # given
    sock, stream = service.connect()
    # when
    reply = service.exchange(stream, {"protocol": 99, "config": CONFIG.as_wire()})
    # then
    assert "error" in reply and "ready" not in reply
    stream.close()
    sock.close()


def test_given_an_engine_crash_when_requesting_then_a_bounded_error_and_the_connection_lives(
    sock_dir: Path,
) -> None:
    # given
    class Crashing:
        calls = 0

        def decide(self, request: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("secret internals /home/x")
            return [{"p_true": 0.1}]

    running = RunningService(sock_dir, Crashing())
    sock, stream = running.connect()
    running.exchange(stream, hello())
    # when
    first = running.exchange(stream, {"request": {"state": "s", "questions": []}})
    second = running.exchange(stream, {"request": {"state": "s", "questions": []}})
    # then
    assert first == {"error": "Local QSIV inference failed"}  # no traceback text on the wire
    assert second == {"answers": [{"p_true": 0.1}]}
    running.stop.set()
    running.thread.join(timeout=5)


def test_given_a_token_count_that_fails_when_requesting_then_decide_never_runs_and_it_fails(
    sock_dir: Path,
) -> None:
    """Fail closed: an unverifiable size is never sent to inference."""

    # given
    class Uncountable:
        max_input_tokens = 100
        decided = 0

        def count_tokens(self, request: Mapping[str, object]) -> int:
            raise RuntimeError("tokenizer broke /home/x")

        def decide(self, request: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
            self.decided += 1
            return [{"p_true": 0.1}]

    engine = Uncountable()
    running = RunningService(sock_dir, engine)
    sock, stream = running.connect()
    running.exchange(stream, hello())
    # when
    reply = running.exchange(stream, {"request": {"state": "s", "questions": []}})
    # then
    assert reply == {"error": "Local QSIV inference failed"}
    assert engine.decided == 0
    running.stop.set()
    running.thread.join(timeout=5)


def test_given_no_requests_when_the_idle_time_passes_then_the_service_exits_even_with_a_client(
    sock_dir: Path,
) -> None:
    """Connected clients do not keep the model resident (they reattach on the next request)."""
    # given
    running = RunningService(sock_dir, FakeEngine(), idle_seconds=0.4)
    sock, stream = running.connect()
    running.exchange(stream, hello())
    # when
    running.thread.join(timeout=5)
    # then
    assert not running.thread.is_alive()
    stream.close()
    sock.close()


def test_given_a_held_model_lock_when_locking_then_model_in_use(tmp_path: Path) -> None:
    """Q13: the interim lock shared with agemem."""
    # given
    directory = tmp_path / "lockdir"
    first = _lock_model(directory)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        _lock_model(directory)
    # then
    assert caught.value.diagnostic.cause is Cause.MODEL_IN_USE
    import os

    os.close(first)


def test_given_missing_arguments_when_running_main_then_usage_and_exit_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # given / when
    code = main([])
    # then
    assert code == 2
    assert "usage:" in capsys.readouterr().err


def test_given_a_runtime_that_is_not_known_when_running_main_then_exit_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # given / when
    code = main(["/py", "/model", "int8"])
    # then
    assert code == 2


def test_given_the_gpu_slot_is_taken_when_running_main_then_the_dedicated_exit_code(
    sock_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q13: the client maps this exit code to model_in_use."""
    # given
    from qsiv_runtime import service as service_module
    from qsiv_runtime.errors import EXIT_MODEL_IN_USE
    from qsiv_runtime.paths import ENV_SOCKET_DIRECTORY

    def taken(config: ServiceConfig) -> Engine:
        raise QSIVUnavailable("taken", QSIVDiagnostic(Cause.MODEL_IN_USE))

    monkeypatch.setattr(service_module, "load_engine", taken)
    monkeypatch.setenv(ENV_SOCKET_DIRECTORY, str(sock_dir))
    # when
    code = main(["/py", "/model", "fp8"])
    # then
    assert code == EXIT_MODEL_IN_USE
