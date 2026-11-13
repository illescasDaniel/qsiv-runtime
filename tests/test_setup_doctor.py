"""S14 (checksum mismatch), S20 (doctor and setup messages) with a fake machine and runner."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from qsiv_runtime import Cause, QSIVUnavailable, doctor, setup
from qsiv_runtime.cli import main
from qsiv_runtime.doctor import DoctorReport
from qsiv_runtime.paths import ENV_SOCKET_DIRECTORY, INSTALL_MARKER
from qsiv_runtime.setup import QSIV_SHA256
from tests.conftest import Install, make_install
from tests.fakes.fake_machine import OLD_GPU, FakeMachine

METADATA = '[project]\ndependencies = ["numpy==2.4.6"]\n'
FAKE_WEIGHTS = b"not the real weights"


class FakeRunner:
    """Records commands; the download command writes fake files instead of fetching anything."""

    def __init__(self, weights: bytes = FAKE_WEIGHTS) -> None:
        self.commands: list[list[str]] = []
        self.envs: list[Mapping[str, str] | None] = []
        self.weights = weights
        self.model: Path | None = None

    def run(self, command: Sequence[str], env: Mapping[str, str] | None = None) -> None:
        self.commands.append(list(command))
        self.envs.append(env)
        if "qsiv_runtime._download" in command:
            model = Path(command[-1])
            model.mkdir(parents=True, exist_ok=True)
            (model / "qsiv_fp8.safetensors").write_bytes(self.weights)
            (model / "pyproject.toml").write_text(METADATA)
            (model / "config.json").write_text("{}")
            (model / "calibration.json").write_text("{}")
            (model / "tokenizer").mkdir(exist_ok=True)

    def joined(self) -> list[str]:
        return [" ".join(c) for c in self.commands]


def test_given_wrong_weights_when_running_setup_then_checksum_mismatch_and_nothing_installed(
    tmp_path: Path,
) -> None:
    """S14 / F8"""
    # given
    runner = FakeRunner()
    model, venv = tmp_path / "model", tmp_path / "venv"
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        setup(model, venv, machine=FakeMachine(), runner=runner)
    # then
    exc = caught.value
    actual = hashlib.sha256(FAKE_WEIGHTS).hexdigest()
    assert exc.diagnostic.cause is Cause.BAD_CHECKSUM
    assert exc.message == (
        f"Checksum mismatch for qsiv_fp8.safetensors (expected {QSIV_SHA256}, got {actual}); "
        "the runtime is NOT installed"
    )
    assert not (model / "qsiv_fp8.safetensors").exists()  # discarded
    assert not (model / INSTALL_MARKER).exists()
    assert not any("pip install" in c and str(model) in c for c in runner.joined()[-2:])


def test_given_the_cli_and_wrong_weights_when_running_setup_then_nonzero_and_not_installed_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S14: the printed text and the exit code."""
    # given / when
    code = main(
        [
            "setup",
            "--model-directory",
            str(tmp_path / "m"),
            "--venv-directory",
            str(tmp_path / "v"),
        ],
        machine=FakeMachine(),
        runner=FakeRunner(),
    )
    # then
    assert code != 0
    err = capsys.readouterr().err
    assert "Checksum mismatch for qsiv_fp8.safetensors" in err
    assert "NOT installed" in err


def test_given_correct_weights_when_running_setup_then_installed_with_a_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup happy path (the pin is patched to the fake weights' hash)."""
    # given
    setup_module = importlib.import_module("qsiv_runtime.setup")  # the name `setup` is the function

    monkeypatch.setattr(setup_module, "QSIV_SHA256", hashlib.sha256(FAKE_WEIGHTS).hexdigest())
    runner = FakeRunner()
    model, venv = tmp_path / "model", tmp_path / "venv"
    # when
    result = setup(model, venv, machine=FakeMachine(), runner=runner)
    # then
    commands = runner.joined()
    assert commands[0].startswith("uv venv --python 3.12")
    assert any("qsiv-linux-py312.lock" in c for c in commands)
    assert commands.index(next(c for c in commands if "_download" in c)) < commands.index(
        next(c for c in commands if c.endswith("--no-deps " + str(model)))
    )
    assert commands[-1].startswith("uv pip check")
    assert "numpy==2.5.4" in (model / "pyproject.toml").read_text()  # the pin is adjusted
    marker = json.loads((model / INSTALL_MARKER).read_text())
    assert marker["weights_sha256"] == hashlib.sha256(FAKE_WEIGHTS).hexdigest()
    assert result.runtime_python == venv / "bin" / "python"
    # the download runs in the GPU venv through PYTHONPATH (no qsiv_runtime installed there)
    download_env = runner.envs[[i for i, c in enumerate(commands) if "_download" in c][0]]
    assert download_env is not None and "PYTHONPATH" in download_env


def test_given_an_unsupported_gpu_when_running_setup_then_it_refuses_before_any_command(
    tmp_path: Path,
) -> None:
    """S20 / F1: setup refuses before any download with the F1 text."""
    # given
    runner = FakeRunner()
    # when
    with pytest.raises(QSIVUnavailable) as caught:
        setup(tmp_path / "m", tmp_path / "v", machine=FakeMachine(gpus=[OLD_GPU]), runner=runner)
    # then
    assert caught.value.diagnostic.cause is Cause.NO_GPU
    assert caught.value.message.endswith("found: NVIDIA GeForce GTX 1080 (6.1)")
    assert runner.commands == []


def test_given_no_uv_when_running_setup_then_a_clear_error_before_any_command(
    tmp_path: Path,
) -> None:
    """Setup needs uv; nothing is run without it."""
    # given
    runner = FakeRunner()
    # when / then
    with pytest.raises(Exception, match="uv"):
        setup(tmp_path / "m", tmp_path / "v", machine=FakeMachine(programs=[]), runner=runner)
    assert runner.commands == []


def test_given_a_dry_run_when_running_setup_then_the_plan_is_returned_and_nothing_runs(
    tmp_path: Path,
) -> None:
    """Dry run: commands are listed, no runner is needed, nothing is created."""
    # given / when
    result = setup(tmp_path / "m", tmp_path / "v", dry_run=True, machine=FakeMachine())
    # then
    assert result.dry_run is True
    assert result.commands[0][:2] == ("uv", "venv")
    assert any("qsiv-linux-py312.lock" in " ".join(c) for c in result.commands)
    assert not (tmp_path / "m").exists() and not (tmp_path / "v").exists()


def test_given_unexpected_metadata_when_running_setup_then_it_stops_and_discards_nothing_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NumPy pin patch refuses metadata it does not recognise."""
    # given
    setup_module = importlib.import_module("qsiv_runtime.setup")  # the name `setup` is the function

    monkeypatch.setattr(setup_module, "QSIV_SHA256", hashlib.sha256(FAKE_WEIGHTS).hexdigest())

    class Odd(FakeRunner):
        def run(self, command: Sequence[str], env: Mapping[str, str] | None = None) -> None:
            super().run(command, env)
            if "qsiv_runtime._download" in command:
                (Path(command[-1]) / "pyproject.toml").write_text("[project]\n")

    # when / then
    with pytest.raises(Exception, match="metadata"):
        setup(tmp_path / "m", tmp_path / "v", machine=FakeMachine(), runner=Odd())
    assert not (tmp_path / "m" / INSTALL_MARKER).exists()


def _doctor(
    install: Install, sock_dir: Path, machine: FakeMachine, *, verify: bool = False
) -> DoctorReport:
    return doctor(
        install.model,
        install.python,
        "fp8",
        machine=machine,
        environ={ENV_SOCKET_DIRECTORY: str(sock_dir)},
        verify=verify,
    )


def test_given_a_ready_machine_when_running_doctor_then_every_check_is_ok(
    install: Install, sock_dir: Path
) -> None:
    """S20"""
    # given / when
    report = _doctor(install, sock_dir, FakeMachine())
    # then
    assert [c.name for c in report.checks] == [
        "platform",
        "gpu",
        "model directory",
        "runtime",
        "socket directory",
    ]
    assert report.ok and all(c.ok and c.cause is None for c in report.checks)


def test_given_an_unsupported_gpu_when_running_doctor_then_the_gpu_check_fails(
    install: Install, sock_dir: Path
) -> None:
    """S20 / F1"""
    # given / when
    report = _doctor(install, sock_dir, FakeMachine(gpus=[OLD_GPU]))
    # then
    gpu = next(c for c in report.checks if c.name == "gpu")
    assert not report.ok and not gpu.ok and gpu.cause is Cause.NO_GPU
    assert "6.1" in gpu.detail
    assert "qsiv-runtime doctor" not in gpu.detail
    assert "another provider" in gpu.detail


def test_given_no_install_when_running_doctor_then_the_model_check_fails_with_not_installed(
    tmp_path: Path, sock_dir: Path
) -> None:
    """S20 / F2"""
    # given
    install = Install(tmp_path / "absent", tmp_path / "absent-python")
    # when
    report = _doctor(install, sock_dir, FakeMachine())
    # then
    causes = {c.name: c.cause for c in report.checks if not c.ok}
    assert causes == {"model directory": Cause.NOT_INSTALLED, "runtime": Cause.NOT_INSTALLED}


def test_given_a_bad_marker_when_running_doctor_then_bad_checksum(
    tmp_path: Path, sock_dir: Path
) -> None:
    """S20 / F8"""
    # given
    install = make_install(tmp_path, marker=None)
    # when
    report = _doctor(install, sock_dir, FakeMachine())
    # then
    model = next(c for c in report.checks if c.name == "model directory")
    assert model.cause is Cause.BAD_CHECKSUM and not model.ok


def test_given_an_unsafe_socket_directory_when_running_doctor_then_that_check_fails(
    install: Install, sock_dir: Path
) -> None:
    """S20 / F7"""
    # given
    sock_dir.chmod(0o755)
    # when
    report = _doctor(install, sock_dir, FakeMachine())
    # then
    check = next(c for c in report.checks if c.name == "socket directory")
    assert check.cause is Cause.UNSAFE_DIRECTORY and not check.ok


def test_given_verify_and_wrong_weights_when_running_doctor_then_bad_checksum(
    install: Install, sock_dir: Path
) -> None:
    """doctor --verify hashes the weights against the pin."""
    # given / when
    report = _doctor(install, sock_dir, FakeMachine(), verify=True)
    # then
    model = next(c for c in report.checks if c.name == "model directory")
    assert model.cause is Cause.BAD_CHECKSUM and "Checksum mismatch" in model.detail


def test_given_the_cli_when_running_doctor_then_one_line_per_check_and_the_exit_code(
    install: Install, sock_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S20: doctor prints one line per check and exits 0 only when all pass."""
    # given
    args = [
        "doctor",
        "--model-directory",
        str(install.model),
        "--python",
        str(install.python),
    ]
    environ = {ENV_SOCKET_DIRECTORY: str(sock_dir)}
    # when
    ok = main(args, machine=FakeMachine(), environ=environ)
    ok_out = capsys.readouterr().out
    bad = main(args, machine=FakeMachine(gpus=[OLD_GPU]), environ=environ)
    bad_out = capsys.readouterr().out
    # then
    assert ok == 0 and bad == 1
    assert len(ok_out.splitlines()) == 5
    assert all(line.startswith("ok ") for line in ok_out.splitlines())
    assert "FAIL gpu" in bad_out and "no_gpu" in bad_out
    assert "qsiv-runtime doctor" not in bad_out
