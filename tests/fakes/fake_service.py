"""Fake services: the real `serve()` on a fake engine, and a scripted raw socket server."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qsiv_runtime._protocol import ServiceConfig
from qsiv_runtime.errors import Cause, QSIVDiagnostic, QSIVUnavailable
from qsiv_runtime.paths import ENV_SOCKET_DIRECTORY, Runtime
from qsiv_runtime.service import Engine, serve

GOOD = [0.02, 0.11, 0.03, 0.01]


class FakeEngine:
    """Records requests; answers every question with the given values (verbatim, even invalid)."""

    def __init__(self, probabilities: Sequence[object] = GOOD, *, limit: int | None = None) -> None:
        self.probabilities = list(probabilities)
        self.limit = limit
        self.requests: list[Mapping[str, object]] = []

    def decide(self, request: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
        self.requests.append(request)
        state = request["state"]
        assert isinstance(state, str)
        if self.limit is not None and len(state) > self.limit:
            raise QSIVUnavailable("input too long", QSIVDiagnostic(Cause.TOO_LARGE))
        return [{"p_true": p} for p in self.probabilities]


class FakeProcess:
    def __init__(self, alive: Callable[[], bool], on_terminate: Callable[[], None]) -> None:
        self._alive, self._on_terminate = alive, on_terminate
        self.terminated = False

    def poll(self) -> int | None:
        return None if self._alive() else 0

    def terminate(self) -> None:
        self.terminated = True
        self._on_terminate()

    def wait(self) -> int:
        return 0


class ExitedProcess:
    """A service that crashed during startup."""

    terminated = False

    def __init__(self, code: int = 1) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None: ...

    def wait(self) -> int:
        return 1


class ThreadServiceLauncher:
    """Runs the real `serve()` in a thread with a fake engine instead of a GPU process."""

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        load_delay: float = 0.0,
        idle_seconds: float = 30.0,
    ) -> None:
        self.engine: Engine = engine if engine is not None else FakeEngine()
        self.load_delay, self.idle_seconds = load_delay, idle_seconds
        self.starts = 0
        self.loads = 0
        self.argvs: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self._mutex = threading.Lock()

    def _load(self) -> Engine:
        with self._mutex:
            self.loads += 1
        time.sleep(self.load_delay)
        return self.engine

    def start(self, argv: Sequence[str], env: Mapping[str, str], log_path: Path) -> FakeProcess:
        with self._mutex:
            self.starts += 1
            self.argvs.append(list(argv))
            self.envs.append(dict(env))
        python, directory, runtime = argv[-3:]
        assert runtime in ("fp8", "reference")
        runtime_: Runtime = "fp8" if runtime == "fp8" else "reference"
        config = ServiceConfig(python, directory, runtime_)
        thread = threading.Thread(
            target=serve,
            args=(config, self._load, self.idle_seconds),
            kwargs={"socket_directory": Path(env[ENV_SOCKET_DIRECTORY]), "stop": self.stop_event},
            daemon=True,
        )
        thread.start()
        self.threads.append(thread)
        return FakeProcess(thread.is_alive, self.stop_event.set)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)


@dataclass
class Generation:
    """What one started service does: its replies to successive requests."""

    replies: list[Sequence[object] | str | dict[str, object]] = field(
        default_factory=list
    )  # answers, "close", "hang"
    startup: str = "ready"  # "ready", "never" (no socket), "silent" (accepts, no ready), "crash"
    exit_code: int = 1  # for "crash"


class ScriptedLauncher:
    """Each start() is the next Generation of a raw socket server scripted by the test."""

    def __init__(self, generations: Sequence[Generation]) -> None:
        self.generations = list(generations)
        self.starts = 0
        self.hellos: list[object] = []
        self.requests: list[object] = []
        self.envs: list[dict[str, str]] = []
        self._servers: list[_ScriptedServer] = []

    def start(
        self, argv: Sequence[str], env: Mapping[str, str], log_path: Path
    ) -> FakeProcess | ExitedProcess:
        generation = self.generations[self.starts]
        self.starts += 1
        self.envs.append(dict(env))
        if generation.startup == "crash":
            return ExitedProcess(generation.exit_code)
        if generation.startup == "never":
            return FakeProcess(lambda: True, lambda: None)
        server = _ScriptedServer(Path(env[ENV_SOCKET_DIRECTORY]) / "service.sock", generation, self)
        self._servers.append(server)
        return FakeProcess(server.alive, server.stop)

    def stop(self) -> None:
        for server in self._servers:
            server.stop()


class _ScriptedServer:
    def __init__(self, path: Path, generation: Generation, owner: ScriptedLauncher) -> None:
        self.path, self.generation, self.owner = path, generation, owner
        self._stop = threading.Event()
        self._listener_closed = threading.Event()
        self._replies = list(generation.replies)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen()
        self.server.settimeout(0.05)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def alive(self) -> bool:
        return not self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        self.server.close()
        self._listener_closed.set()

    def _handle(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as stream:
            hello = json.loads(stream.readline())
            self.owner.hellos.append(hello)
            if self.generation.startup == "silent":
                self._stop.wait()
                return
            stream.write(b'{"ready":true}\n')
            stream.flush()
            for line in stream:
                self.owner.requests.append(json.loads(line))
                reply = self._replies.pop(0) if self._replies else "close"
                if isinstance(reply, str):
                    if reply == "close":
                        # The service exits: stop listening first, so the client's next connect is
                        # refused (the stale socket file stays behind), then drop this connection.
                        self._stop.set()
                        self._listener_closed.wait(timeout=2)
                    else:
                        self._stop.wait()  # "hang": accepted, never answered
                    return
                if isinstance(reply, dict):  # a raw message, e.g. an error reply
                    stream.write((json.dumps(reply) + "\n").encode())
                    stream.flush()
                    continue
                answers = {"answers": [{"p_true": p} for p in reply]}
                stream.write((json.dumps(answers) + "\n").encode())
                stream.flush()
