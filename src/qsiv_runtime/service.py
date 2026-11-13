"""The resident service: loads the model once, serializes inference, exits when idle.

Started by the client as `python -m qsiv_runtime.service <python> <model directory> <runtime>`
inside the GPU virtualenv, with this package on PYTHONPATH (standard library only until
`load_engine`)."""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol

from ._preflight import ensure_private_directory, weights_name
from ._protocol import PROTOCOL_VERSION, ServiceConfig, decode, encode
from .errors import EXIT_MODEL_IN_USE, Cause, QSIVDiagnostic, QSIVError, QSIVUnavailable
from .paths import (
    ENV_IDLE_SECONDS,
    default_socket_directory,
    parse_runtime,
)

DEFAULT_IDLE_SECONDS: Final = 300.0
USAGE: Final = "usage: python -m qsiv_runtime.service <python> <model-directory> <fp8|reference>"


class Engine(Protocol):
    """Optional hooks, both needed to enforce a limit: `max_input_tokens: int | None` and
    `count_tokens(request) -> int` (the whole request as the model renders it: state plus every
    question block). An engine without them is never size-checked here."""

    def decide(self, request: Mapping[str, object]) -> Sequence[Mapping[str, object]]: ...


def check_input_size(engine: object, request: Mapping[str, object]) -> None:
    """Reject input over the engine's known token limit before inference (never truncates).

    Does nothing when the engine exposes no limit."""
    limit = getattr(engine, "max_input_tokens", None)
    count = getattr(engine, "count_tokens", None)
    if not isinstance(limit, int) or isinstance(limit, bool) or not callable(count):
        return
    total = int(count(request))
    if total > limit:
        raise QSIVUnavailable(
            f"input is over the QSIV model's input limit ({total} > {limit} tokens)",
            QSIVDiagnostic(Cause.TOO_LARGE),
        )


def _uid() -> int:
    return getattr(os, "getuid", lambda: 0)()


def legacy_lock_directory() -> Path:
    # Until agemem migrates, its service holds this lock; taking it keeps one model per GPU (Q13).
    return Path("/tmp") / f"agemem-qsiv-{_uid()}"


def _lock_model(directory: Path) -> int:
    """Reserve the local user's QSIV GPU slot until this process exits; F5 when it is taken."""
    import fcntl

    ensure_private_directory(directory)
    fd = os.open(directory / "model.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise QSIVUnavailable(
            "another QSIV model is already running; stop it before starting another",
            QSIVDiagnostic(Cause.MODEL_IN_USE),
        ) from None
    except BaseException:
        os.close(fd)
        raise
    return fd


class _NativeEngine:
    def __init__(
        self,
        service: object,
        max_input_tokens: int | None = None,
        count_tokens: Callable[[Mapping[str, object]], int] | None = None,
    ) -> None:
        self._decide: Callable[[dict[str, object]], Sequence[Mapping[str, object]]] = service.decide  # type: ignore[attr-defined]
        self.max_input_tokens = max_input_tokens
        self.count_tokens = count_tokens

    def decide(self, request: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
        return [dict(answer) for answer in self._decide(dict(request))]


def native_engine(service: object, constants: object, render: object) -> _NativeEngine:
    """The engine over a loaded QSIV service, with the limit the qsiv package documents.

    `qsiv.constants.MAX_CONTEXT` (65,536 tokens in QSIV 1.0) bounds the whole rendered request
    (state plus every templated question block); the service raises a bare ValueError past it,
    so the count uses the package's own `qsiv.render.encode_request` and the service's tokenizer
    and rejects the request as too_large first. Without those attributes no limit is enforced."""
    limit = getattr(constants, "MAX_CONTEXT", None)
    encode = getattr(render, "encode_request", None)
    tokenizer = getattr(service, "tok", None)
    if not isinstance(limit, int) or not callable(encode) or tokenizer is None:
        return _NativeEngine(service)

    def count_tokens(request: Mapping[str, object]) -> int:
        questions = request.get("questions")
        encoded = encode(
            tokenizer, request.get("state", ""), questions if isinstance(questions, list) else []
        )
        return len(encoded["state_ids"]) + sum(len(q["ids"]) for q in encoded["questions"])

    return _NativeEngine(service, limit, count_tokens)


def load_engine(config: ServiceConfig) -> Engine:
    """Imports torch and qsiv lazily, checks the GPU, takes the model lock. Local files only."""
    root = Path(config.model_directory)
    for name in ("calibration.json", "config.json", "tokenizer", weights_name(config.runtime)):
        if not (root / name).exists():
            raise QSIVUnavailable(
                f"missing QSIV local artifact: {name}", QSIVDiagnostic(Cause.NOT_INSTALLED)
            )
    # The descriptor is retained for the process lifetime: CUDA may keep allocations after the
    # Python service object is collected.
    _lock_model(legacy_lock_directory())
    try:
        import torch

        if not torch.cuda.is_available():
            raise QSIVUnavailable("QSIV requires an NVIDIA GPU", QSIVDiagnostic(Cause.NO_GPU))
        minimum = (8, 9) if config.runtime == "fp8" else (8, 0)
        if torch.cuda.get_device_capability() < minimum:
            raise QSIVUnavailable(
                "the GPU does not meet this QSIV runtime's requirements",
                QSIVDiagnostic(Cause.NO_GPU),
            )
        if config.runtime == "reference":
            from qsiv.reference import ReferenceService

            service: object = ReferenceService(
                str(root / "model.safetensors"),
                str(root / "calibration.json"),
                tokenizer_dir=str(root / "tokenizer"),
            )
        else:
            from qsiv.runtime.service import QSIVService

            service = QSIVService(
                str(root / "qsiv_fp8.safetensors"),
                str(root / "calibration.json"),
                tokenizer_dir=str(root / "tokenizer"),
                plan_memory_gb=1,
            )
        from qsiv import constants, render
    except (ImportError, OSError, RuntimeError) as exc:
        raise QSIVUnavailable(
            f"QSIV GPU runtime unavailable ({type(exc).__name__})",
            QSIVDiagnostic(Cause.STARTUP_FAILED),
        ) from exc
    return native_engine(service, constants, render)


def _idle_seconds() -> float:
    try:
        return float(os.environ[ENV_IDLE_SECONDS])
    except (KeyError, ValueError):
        return DEFAULT_IDLE_SECONDS


def serve(
    config: ServiceConfig,
    loader: Callable[[], Engine] | None = None,
    idle_seconds: float | None = None,
    *,
    socket_directory: Path | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Load once, serialize inference, exit after idle_seconds (default 300) without requests.

    Connected clients do not keep the model resident; they reattach on their next request.
    `stop` ends the loop early (tests and embedding)."""
    idle = _idle_seconds() if idle_seconds is None else idle_seconds
    root = ensure_private_directory(socket_directory or default_socket_directory())
    with contextlib.redirect_stdout(sys.stderr):
        engine = (loader or (lambda: load_engine(config)))()
    path = root / "service.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Bind and listen under a temporary name, then rename: the public name never exists without a
    # listener behind it, so a client cannot mistake a starting service for a stale socket.
    pending = root / f"service.sock.{os.getpid()}"
    pending.unlink(missing_ok=True)
    server.bind(str(pending))
    os.chmod(pending, 0o600)
    server.listen()
    os.replace(pending, path)
    server.settimeout(0.1)
    inference = threading.Lock()
    state_lock = threading.Lock()
    clients: set[socket.socket] = set()
    idle_since = time.monotonic()
    busy = 0
    wire = config.as_wire()

    def respond(stream: object, message: Mapping[str, object]) -> None:
        stream.write(encode(message))  # type: ignore[attr-defined]
        stream.flush()  # type: ignore[attr-defined]

    def handle(conn: socket.socket) -> None:
        nonlocal idle_since, busy
        stream = None
        try:
            stream = conn.makefile("rwb")
            hello = decode(stream.readline())
            if hello.get("protocol") != PROTOCOL_VERSION:
                respond(
                    stream,
                    {
                        "error": "unsupported QSIV protocol version",
                        "diagnostic": {"cause": Cause.STARTUP_FAILED.value},
                    },
                )
                return
            if hello.get("config") != wire:
                respond(
                    stream,
                    {
                        "error": "a QSIV service with a different model or runtime is running",
                        "diagnostic": {"cause": Cause.MODEL_IN_USE.value, "resident": wire},
                    },
                )
                return
            respond(stream, {"ready": True})
            for line in stream:
                try:
                    request = decode(line)["request"]
                    assert isinstance(request, dict)
                    with state_lock:
                        busy += 1
                    try:
                        with inference, contextlib.redirect_stdout(sys.stderr):
                            # Under the inference lock: counting uses the model's tokenizer,
                            # whose caches are not thread-safe.
                            check_input_size(engine, request)
                            result = list(engine.decide(request))
                    finally:
                        with state_lock:
                            busy -= 1
                            idle_since = time.monotonic()
                    response: dict[str, object] = {"answers": result}
                except QSIVUnavailable as exc:
                    response = {
                        "error": exc.message,
                        "diagnostic": {"cause": exc.diagnostic.cause.value},
                    }
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    response = {"error": "Local QSIV inference failed"}
                respond(stream, response)
        except (OSError, ValueError, KeyError, QSIVError):
            pass
        finally:
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
            conn.close()
            with state_lock:
                clients.discard(conn)

    try:
        while stop is None or not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                with state_lock:
                    if not busy and time.monotonic() - idle_since >= idle:
                        break
                continue
            with state_lock:
                clients.add(conn)
                idle_since = time.monotonic()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    finally:
        server.close()
        with contextlib.suppress(OSError):
            path.unlink()
        with state_lock:
            for conn in clients:
                with contextlib.suppress(OSError):
                    conn.shutdown(socket.SHUT_RDWR)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[2] not in ("fp8", "reference"):
        print(USAGE, file=sys.stderr)
        return 2
    config = ServiceConfig(args[0], args[1], parse_runtime(args[2]))
    try:
        serve(config)
    except QSIVUnavailable as exc:
        print(f"qsiv_runtime.service: {exc.diagnostic.cause.value}: {exc.message}", file=sys.stderr)
        return EXIT_MODEL_IN_USE if exc.diagnostic.cause is Cause.MODEL_IN_USE else 1
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
