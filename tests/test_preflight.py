"""S6, S7, S13, S15: the checks that fail before any socket, lock or service work."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qsiv_runtime import Cause, QSIVUnavailable
from qsiv_runtime._preflight import ensure_private_directory
from qsiv_runtime.paths import ENV_SOCKET_DIRECTORY, INSTALL_MARKER, LEGACY_INSTALL_MARKER
from qsiv_runtime.setup import QSIV_REVISION, QSIV_SHA256
from tests.conftest import Install, make_install
from tests.fakes.fake_machine import OLD_GPU, FakeMachine
from tests.fakes.fake_service import (
    FakeEngine,
    Generation,
    ScriptedLauncher,
    ThreadServiceLauncher,
)
from tests.helpers import QUESTIONS, make_client


def test_given_an_old_gpu_when_deciding_then_no_gpu_before_any_work(
    install: Install, sock_dir: Path
) -> None:
    """S6 / F1"""
    # given
    launcher = ThreadServiceLauncher()
    client = make_client(install, sock_dir, launcher, machine=FakeMachine(gpus=[OLD_GPU]))
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.NO_GPU
    assert exc.message == (
        "QSIV needs Linux x86_64 and an NVIDIA GPU with compute capability 8.9 or higher (FP8); "
        "found: NVIDIA GeForce GTX 1080 (6.1)"
    )
    assert exc.diagnostic.hint is not None and "qsiv-runtime doctor" in exc.diagnostic.hint
    assert launcher.starts == 0
    assert list(sock_dir.iterdir()) == []  # no lock, no socket


@pytest.mark.parametrize(
    ("machine", "found"),
    [
        (FakeMachine(system="Darwin", machine="arm64"), "found: Darwin arm64"),
        (FakeMachine(machine="aarch64"), "found: Linux aarch64"),
        (FakeMachine(gpus=[]), "found: no NVIDIA GPU"),
    ],
)
def test_given_an_unsupported_platform_or_no_gpu_when_deciding_then_no_gpu(
    install: Install, sock_dir: Path, machine: FakeMachine, found: str
) -> None:
    """S6 (variants)"""
    # given
    client = make_client(install, sock_dir, ThreadServiceLauncher(), machine=machine)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.NO_GPU
    assert caught.value.message.endswith(found)


def test_given_a_reference_runtime_when_the_gpu_is_8_0_then_it_is_accepted(
    install: Install, sock_dir: Path
) -> None:
    """The reference runtime needs compute capability 8.0, not 8.9."""
    # given
    (install.model / "model.safetensors").write_text("{}")
    from qsiv_runtime._ports import GpuInfo

    launcher = ThreadServiceLauncher(FakeEngine([0.1] * 4))
    client = make_client(
        install,
        sock_dir,
        launcher,
        machine=FakeMachine(gpus=[GpuInfo("NVIDIA A100", (8, 0))]),
        runtime="reference",
    )
    # when / then
    assert client.decide("x", QUESTIONS) == [0.1] * 4
    client.close()
    launcher.stop()


def test_given_no_install_anywhere_when_deciding_then_not_installed_and_nothing_started(
    tmp_path: Path, sock_dir: Path
) -> None:
    """S7 / F2: defaults and legacy locations both absent."""
    # given
    from qsiv_runtime import QSIVClient

    launcher = ThreadServiceLauncher()
    client = QSIVClient(
        socket_directory=sock_dir,
        environ={"HOME": str(tmp_path)},
        machine=FakeMachine(),
        launcher=launcher,
    )
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.NOT_INSTALLED
    assert exc.message == "QSIV is not installed"
    assert exc.diagnostic.hint is not None and "Run: qsiv-runtime setup" in exc.diagnostic.hint
    checked = exc.diagnostic.detail["checked"]
    assert str(tmp_path / ".local/share/qsiv-runtime/model") in checked
    assert str(tmp_path / ".local/share/agemem/models/qsiv") in checked
    assert launcher.starts == 0


def test_given_missing_artifacts_when_deciding_then_not_installed(
    install: Install, sock_dir: Path
) -> None:
    """S7: a model directory without its weights is not an install."""
    # given
    (install.model / "qsiv_fp8.safetensors").unlink()
    client = make_client(install, sock_dir, ThreadServiceLauncher())
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.NOT_INSTALLED


def _bad_marker(root: Path, content: str | None) -> Install:
    install = make_install(root, marker=None)
    if content is not None:
        (install.model / INSTALL_MARKER).write_text(content)
    return install


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        json.dumps({"revision": "other", "weights_sha256": QSIV_SHA256}),
        json.dumps({"revision": QSIV_REVISION, "weights_sha256": "0" * 64}),
        json.dumps(["revision"]),
    ],
)
def test_given_a_missing_or_mismatched_marker_when_deciding_then_bad_checksum(
    tmp_path: Path, sock_dir: Path, content: str | None
) -> None:
    """S15 / F8"""
    # given
    install = _bad_marker(tmp_path, content)
    launcher = ThreadServiceLauncher()
    client = make_client(install, sock_dir, launcher)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    exc = caught.value
    assert exc.diagnostic.cause is Cause.BAD_CHECKSUM
    assert exc.message == "QSIV runtime is not installed (setup did not complete)"
    assert exc.diagnostic.hint == "Run: qsiv-runtime setup"
    assert launcher.starts == 0


def test_given_a_legacy_marker_with_matching_pins_when_deciding_then_it_is_accepted(
    tmp_path: Path, sock_dir: Path
) -> None:
    """S15/S27: the agemem marker still counts when revision and checksum match."""
    # given
    install = make_install(tmp_path, marker=LEGACY_INSTALL_MARKER)
    launcher = ThreadServiceLauncher()
    client = make_client(install, sock_dir, launcher)
    # when / then
    assert len(client.decide("x", QUESTIONS)) == 4
    client.close()
    launcher.stop()


def test_given_a_group_readable_directory_when_deciding_then_unsafe_and_nothing_sent(
    install: Install, tmp_path: Path
) -> None:
    """S13 / F7: mode 0755; a live service in it is never contacted."""
    # given
    unsafe = Path("/tmp") / f"qsv-unsafe-{os.getpid()}"
    unsafe.mkdir(mode=0o755, exist_ok=True)
    unsafe.chmod(0o755)
    launcher = ScriptedLauncher([Generation(replies=[[0.1] * 4]), Generation()])
    launcher.start([], {ENV_SOCKET_DIRECTORY: str(unsafe)}, unsafe / "service.log")
    client = make_client(install, unsafe, launcher)
    try:
        # when
        with pytest.raises(QSIVUnavailable) as caught:
            client.decide("x", QUESTIONS)
        # then
        exc = caught.value
        assert exc.diagnostic.cause is Cause.UNSAFE_DIRECTORY
        assert exc.message == (
            f"QSIV socket directory {unsafe} is not private (mode 0755, expected 0700)"
        )
        assert launcher.hellos == [] and launcher.starts == 1  # only the setup start above
        assert not (unsafe / "startup.lock").exists()
    finally:
        launcher.stop()
        for leftover in unsafe.iterdir():
            leftover.unlink()
        unsafe.rmdir()


def test_given_a_symlinked_directory_when_deciding_then_unsafe(
    install: Install, tmp_path: Path
) -> None:
    """S13 / F7: a symlink, even to a private directory."""
    # given
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    launcher = ThreadServiceLauncher()
    client = make_client(install, link, launcher)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        client.decide("x", QUESTIONS)
    # then
    assert caught.value.diagnostic.cause is Cause.UNSAFE_DIRECTORY
    assert "is not private" in caught.value.message
    assert launcher.starts == 0
    assert list(real.iterdir()) == []


def test_given_a_directory_owned_by_someone_else_then_unsafe(tmp_path: Path) -> None:
    """S13 / F7: another owner (simulated by passing a different expected uid)."""
    # given
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        ensure_private_directory(private, uid=os.getuid() + 1)
    # then
    assert caught.value.diagnostic.cause is Cause.UNSAFE_DIRECTORY
    assert "owned by another user" in caught.value.message


def test_given_a_missing_directory_when_ensuring_then_it_is_created_private(
    tmp_path: Path,
) -> None:
    """The socket directory is created with mode 0700."""
    # given / when
    path = ensure_private_directory(tmp_path / "new")
    # then
    assert path.stat().st_mode & 0o777 == 0o700
