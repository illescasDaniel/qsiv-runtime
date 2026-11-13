"""The stdlib-only client: lazy attach, startup lock, restart once, request timeout, validation."""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from . import _preflight
from ._launch import build_service_env
from ._ports import (
    Clock,
    Launcher,
    Machine,
    ServiceProcess,
    SubprocessLauncher,
    SystemClock,
    SystemMachine,
)
from ._protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    ServiceConfig,
    WireQuestion,
    decode,
    encode,
    invalid_answer,
    validate_probabilities,
)
from .errors import EXIT_MODEL_IN_USE, Cause, QSIVDiagnostic, QSIVUnavailable
from .paths import ENV_RUNTIME, ResolvedPaths, Runtime, resolve_paths

MAX_QUESTIONS: Final = 8  # one physical call; the engine is built for at most 8
DEFAULT_STARTUP_TIMEOUT: Final = 120.0
DEFAULT_REQUEST_TIMEOUT: Final = 60.0
IDLE_EXIT_SECONDS: Final = 300
_MAX_LINE: Final = 1 << 20


def _seconds(value: float) -> str:
    return f"{value:g}"


def timeout_message(seconds: float) -> str:
    return f"the QSIV service did not answer within {_seconds(seconds)} s"


class _Connection:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.stream = sock.makefile("rwb")

    def exchange(self, message: Mapping[str, object]) -> dict[str, object]:
        self.stream.write(encode(message))
        self.stream.flush()
        line = self.stream.readline(_MAX_LINE)
        if not line:
            raise EOFError("QSIV service disconnected")
        return decode(line)

    def close(self) -> None:
        try:
            self.stream.close()
        except OSError:  # flushing to a service that already exited
            pass
        self.sock.close()


class QSIVClient:
    def __init__(
        self,
        model_directory: str | os.PathLike[str] | None = None,
        python: str | os.PathLike[str] | None = None,
        runtime: Runtime | None = None,
        *,
        socket_directory: str | os.PathLike[str] | None = None,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        environ: Mapping[str, str] | None = None,
        machine: Machine | None = None,
        launcher: Launcher | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Lazy: no socket, lock or process until the first decide()."""
        self._arguments = (model_directory, python, runtime, socket_directory)
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self._environ = os.environ if environ is None else environ
        self._machine = machine or SystemMachine()
        self._launcher = launcher or SubprocessLauncher()
        self._clock = clock or SystemClock()
        self._connection: _Connection | None = None
        self._log_path: Path | None = None
        self._mutex = threading.Lock()
        self._requested = ""
        self.restarts = 0  # restarts after a closed connection (F4), since creation

    def decide(self, state: str, questions: Sequence[str]) -> list[float]:
        """One probability per question, validated, same order. Raises QSIVUnavailable
        (never a traceback) for F1 to F10; ValueError for more than MAX_QUESTIONS or none."""
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(
                f"QSIV takes 1 to {MAX_QUESTIONS} questions per call, got {len(questions)}"
            )
        wire: list[WireQuestion] = [{"type": "noul", "text": text} for text in questions]
        request = {"request": {"state": state, "questions": wire}}
        with self._mutex:
            try:
                reply = self._request(request, len(questions))
                return self._answers(reply, len(questions))
            except QSIVUnavailable:
                self._drop()  # never reuse a connection after a failure
                raise

    def _request(self, request: Mapping[str, object], expected: int) -> dict[str, object]:
        for attempt in (0, 1):
            if self._connection is None:
                try:
                    self._attach()
                except OSError as exc:  # mkdir, lock file, ...: never a raw traceback
                    raise self._startup_failed(
                        f"the QSIV service could not be set up ({type(exc).__name__})"
                    ) from None
            assert self._connection is not None
            try:
                return self._connection.exchange(request)
            except TimeoutError:  # before OSError: a slow service is not a dead one (F10)
                raise QSIVUnavailable(
                    timeout_message(self.request_timeout),
                    QSIVDiagnostic(Cause.TIMEOUT, "Retry; read the log", self._log()),
                ) from None
            except ProtocolError:
                raise invalid_answer(expected, "a malformed reply") from None
            except (OSError, EOFError):  # the service closed the connection (idle exit): F4
                self._drop()
                if attempt == 1:
                    break
                self.restarts += 1
        raise QSIVUnavailable(
            f"the QSIV service exited during the review twice (see {self._log()})",
            QSIVDiagnostic(Cause.SERVICE_EXITED, "Retry; read the log", self._log()),
        )

    def _answers(self, reply: Mapping[str, object], expected: int) -> list[float]:
        if "error" in reply:
            raise self._service_error(reply)
        return validate_probabilities(reply.get("answers"), expected)

    def _service_error(self, reply: Mapping[str, object]) -> QSIVUnavailable:
        diagnostic = reply.get("diagnostic")
        detail = diagnostic if isinstance(diagnostic, dict) else {}
        cause = detail.get("cause")
        if cause == Cause.TOO_LARGE.value:
            return QSIVUnavailable(
                "this script is too long for the QSIV model, so it was not reviewed "
                "(it is never truncated)",
                QSIVDiagnostic(Cause.TOO_LARGE, "Use a smaller script or another provider"),
            )
        if cause == Cause.MODEL_IN_USE.value:
            resident = detail.get("resident")
            requested = self._requested
            shown = _describe(resident) if isinstance(resident, list) else "unknown"
            return QSIVUnavailable(
                f"another QSIV model is already loaded (resident: {shown}; requested: {requested})",
                QSIVDiagnostic(
                    Cause.MODEL_IN_USE,
                    f"Use the same settings, or wait for idle exit ({IDLE_EXIT_SECONDS} s)",
                    None,
                    {"resident": shown, "requested": requested},
                ),
            )
        log = self._log()
        known = next((c for c in Cause if c.value == cause), None)
        reported = f"{known.value}; " if known else ""
        return QSIVUnavailable(
            f"QSIV service failed ({reported}see {log})",
            QSIVDiagnostic(known or Cause.STARTUP_FAILED, "Retry; read the log", log),
        )

    def _log(self) -> str:
        return str(self._log_path) if self._log_path else "the service log"

    def _resolve(self) -> ResolvedPaths:
        model, python, runtime, sockets = self._arguments
        try:
            return resolve_paths(
                model, python, runtime, socket_directory=sockets, environ=self._environ
            )
        except ValueError as exc:
            raise QSIVUnavailable(
                f"invalid QSIV runtime setting ({exc}); set {ENV_RUNTIME} or the runtime "
                "argument to fp8 or reference",
                QSIVDiagnostic(Cause.STARTUP_FAILED, f"Use {ENV_RUNTIME}=fp8 or reference"),
            ) from None

    def _startup_failed(self, message: str) -> QSIVUnavailable:
        log = self._log()
        return QSIVUnavailable(
            f"{message} (see {log})",
            QSIVDiagnostic(Cause.STARTUP_FAILED, "Read the log; GPU memory or driver", log),
        )

    def _attach(self) -> None:
        """Preflight, then connect to the shared service, starting it under the lock if absent."""
        paths = self._resolve()
        _preflight.check_machine(self._machine, paths.runtime)
        import fcntl  # after the preflight: F1 must be raised on non-Linux, where it is missing

        _preflight.check_install(paths)
        directory = _preflight.ensure_private_directory(paths.socket_directory)
        config = ServiceConfig(
            str(paths.python), str(paths.model_directory.resolve()), paths.runtime
        )
        self._requested = _describe(config.as_wire())
        self._log_path = directory / "service.log"
        path = directory / "service.sock"
        deadline = self._clock.monotonic() + self.startup_timeout
        failure = f"the QSIV service did not become ready within {_seconds(self.startup_timeout)} s"
        fd = os.open(directory / "startup.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self._clock.monotonic() >= deadline:
                        raise self._startup_failed(failure) from None
                    self._clock.sleep(0.05)
            try:
                self._connection = self._connect(path, config, deadline)
            except (ConnectionRefusedError, FileNotFoundError):
                self._start(path, directory, config, deadline, failure)
            except (OSError, EOFError, ProtocolError):
                raise self._startup_failed(failure) from None
        finally:
            os.close(fd)

    def _start(
        self, path: Path, directory: Path, config: ServiceConfig, deadline: float, failure: str
    ) -> None:
        path.unlink(missing_ok=True)
        python = config.python
        argv = [python, "-m", "qsiv_runtime.service", *config.as_wire()]
        try:
            process = self._launcher.start(
                argv, build_service_env(self._environ, directory), directory / "service.log"
            )
        except OSError:
            raise self._startup_failed("the QSIV service could not be started") from None
        while True:
            code = process.poll()
            if code is not None:
                if code == EXIT_MODEL_IN_USE:  # another model holds the GPU slot (F5)
                    raise self._service_error(
                        {"error": "", "diagnostic": {"cause": Cause.MODEL_IN_USE.value}}
                    )
                raise self._startup_failed("the QSIV service failed to start")
            if self._clock.monotonic() >= deadline:
                _stop(process)
                raise self._startup_failed(failure)
            try:
                self._connection = self._connect(path, config, deadline)
                return
            except (OSError, EOFError, ProtocolError):
                self._clock.sleep(0.05)

    def _connect(self, path: Path, config: ServiceConfig, deadline: float) -> _Connection:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(max(0.1, deadline - self._clock.monotonic()))
            sock.connect(str(path))
            connection = _Connection(sock)
            reply = connection.exchange({"protocol": PROTOCOL_VERSION, "config": config.as_wire()})
            if reply.get("ready") is not True:
                connection.close()
                raise self._service_error(reply)
            sock.settimeout(self.request_timeout)
            return connection
        except BaseException:
            sock.close()
            raise

    def _drop(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def close(self) -> None:
        with self._mutex:
            self._drop()

    def __enter__(self) -> QSIVClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _describe(config: Sequence[object]) -> str:
    python, directory, runtime = (str(part) for part in config)
    return f"{runtime} at {directory} using {python}"


def _stop(process: ServiceProcess) -> None:
    process.terminate()
    process.wait()
