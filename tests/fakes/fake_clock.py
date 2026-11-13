from __future__ import annotations


class FakeClock:
    """Sleeping advances time instantly, so a 120 s timeout costs no wall time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds
