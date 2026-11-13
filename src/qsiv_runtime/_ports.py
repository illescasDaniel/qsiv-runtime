"""Seams the tests fake: clock, machine facts, service launcher, command runner."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class GpuInfo:
    name: str
    compute_capability: tuple[int, int]


class Machine(Protocol):
    def system(self) -> str: ...  # "Linux"

    def machine(self) -> str: ...  # "x86_64"

    def gpus(self) -> Sequence[GpuInfo]: ...  # nvidia-smi; empty when none

    def which(self, program: str) -> str | None: ...


class ServiceProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self) -> int: ...


class Launcher(Protocol):
    def start(
        self, argv: Sequence[str], env: Mapping[str, str], log_path: Path
    ) -> ServiceProcess: ...


class CommandRunner(Protocol):
    def run(self, command: Sequence[str], env: Mapping[str, str] | None = None) -> None:
        """Run a command to completion; raise on a nonzero exit."""


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class SystemMachine:
    """Real platform facts; the GPU list comes from nvidia-smi and is empty when it is unusable."""

    def system(self) -> str:
        import platform

        return platform.system()

    def machine(self) -> str:
        import platform

        return platform.machine()

    def gpus(self) -> Sequence[GpuInfo]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            devices = []
            for row in completed.stdout.splitlines():
                name, capability = row.rsplit(",", 1)
                major, minor = (int(part) for part in capability.strip().split("."))
                devices.append(GpuInfo(name.strip(), (major, minor)))
            return devices
        except (OSError, subprocess.SubprocessError, ValueError):
            return []

    def which(self, program: str) -> str | None:
        import shutil

        return shutil.which(program)


class SubprocessLauncher:
    """Starts the detached service with its stderr appended to the private log."""

    def start(self, argv: Sequence[str], env: Mapping[str, str], log_path: Path) -> ServiceProcess:
        import os
        import threading

        log_fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_fd,
                env=dict(env),
                cwd=log_path.parent,  # the private socket directory, never the caller's cwd
                start_new_session=True,
            )
        finally:
            os.close(log_fd)
        # Reap the detached service if this client stays alive after it exits.
        threading.Thread(target=process.wait, daemon=True).start()
        return process


class SubprocessRunner:
    def run(self, command: Sequence[str], env: Mapping[str, str] | None = None) -> None:
        import os
        import sys
        import tempfile

        base = dict(os.environ if env is None else env)
        base["PYTHONSAFEPATH"] = "1"
        # Keep stdout clean for callers; progress goes to stderr. Run from an empty temp dir so
        # nothing planted in the caller's working directory can shadow a module.
        with tempfile.TemporaryDirectory(prefix="qsiv-runtime-run-") as cwd:
            subprocess.run(
                list(command),
                check=True,
                stdout=sys.stderr,
                stderr=sys.stderr,
                env=base,
                cwd=cwd,
            )
