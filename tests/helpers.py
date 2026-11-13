from __future__ import annotations

from pathlib import Path

from qsiv_runtime import QSIVClient
from qsiv_runtime._ports import Clock, Launcher, Machine
from qsiv_runtime.paths import Runtime
from tests.conftest import Install
from tests.fakes.fake_machine import FakeMachine

QUESTIONS = ["privilege escalation?", "filesystem?", "networking?", "remote code execution?"]


def make_client(
    install: Install,
    sock_dir: Path,
    launcher: Launcher,
    *,
    machine: Machine | None = None,
    clock: Clock | None = None,
    runtime: Runtime = "fp8",
    startup_timeout: float = 120.0,
    request_timeout: float = 60.0,
) -> QSIVClient:
    return QSIVClient(
        install.model,
        install.python,
        runtime,
        socket_directory=sock_dir,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        environ={"PATH": "/usr/bin"},
        machine=machine or FakeMachine(),
        launcher=launcher,
        clock=clock,
    )
