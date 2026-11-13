from __future__ import annotations

from collections.abc import Sequence

from qsiv_runtime._ports import GpuInfo

SUPPORTED_GPU = GpuInfo("NVIDIA GeForce RTX 4090", (8, 9))
OLD_GPU = GpuInfo("NVIDIA GeForce GTX 1080", (6, 1))


class FakeMachine:
    def __init__(
        self,
        *,
        system: str = "Linux",
        machine: str = "x86_64",
        gpus: Sequence[GpuInfo] = (SUPPORTED_GPU,),
        programs: Sequence[str] = ("uv",),
    ) -> None:
        self._system, self._machine, self._gpus = system, machine, list(gpus)
        self._programs = set(programs)

    def system(self) -> str:
        return self._system

    def machine(self) -> str:
        return self._machine

    def gpus(self) -> Sequence[GpuInfo]:
        return self._gpus

    def which(self, program: str) -> str | None:
        return f"/usr/bin/{program}" if program in self._programs else None
