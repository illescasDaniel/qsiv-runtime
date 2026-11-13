"""S26 (secrets scrubbed) and the PYTHONPATH launch (Q12)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import qsiv_runtime
from qsiv_runtime import QSIVClient
from qsiv_runtime._launch import stage_package
from qsiv_runtime.paths import ENV_SOCKET_DIRECTORY
from tests.conftest import Install
from tests.fakes.fake_machine import FakeMachine
from tests.fakes.fake_service import (
    ExitedProcess,
    FakeProcess,
    Generation,
    ScriptedLauncher,
)
from tests.helpers import QUESTIONS

SECRETS = {
    "OPENAI_API_KEY": "sk-1",
    "GH_TOKEN": "ghp",
    "MY_SECRET": "s",
    "lower_token_x": "t",
    "DB_PASSWORD": "p",
    "mysql_passwd": "p",
    "AWS_ACCESS_KEY_ID": "a",
    "AWS_CREDENTIALS_FILE": "c",
    "GPG_PRIVATE_KEY": "k",
    "SERVICE_APIKEY": "k",
    "SSH_AUTH_SOCK": "/run/agent",
    "auth_url": "u",
    "NPM_CONFIG_AUTH": "x",
}
KEPT = {
    "PATH": "/usr/bin",
    "HOME": "/home/x",
    "LANG": "C",
    "AUTHOR": "me",
    "GIT_AUTHOR_NAME": "me",
    "OAUTH_NOTE": "n",
    "AUTHORITY": "a",
    "KEYBOARD": "k",
}


def _start(install: Install, sock_dir: Path, environ: dict[str, str]) -> ScriptedLauncher:
    launcher = ScriptedLauncher([Generation(replies=[[0.1] * 4])])
    client = QSIVClient(
        install.model,
        install.python,
        "fp8",
        socket_directory=sock_dir,
        environ=environ,
        machine=FakeMachine(),
        launcher=launcher,
    )
    assert client.decide("x", QUESTIONS) == [0.1] * 4
    client.close()
    launcher.stop()
    return launcher


def test_given_secrets_in_the_environment_when_starting_then_the_service_env_has_none(
    install: Install, sock_dir: Path
) -> None:
    """S26"""
    # given
    environ = {**SECRETS, **KEPT}
    # when
    env = _start(install, sock_dir, environ).envs[0]
    # then
    assert not set(SECRETS) & set(env)
    assert KEPT.items() <= env.items()  # AUTH is a whole segment: AUTHOR and friends are passed on
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_given_a_start_then_the_service_gets_only_the_staged_pythonpath(
    install: Install, sock_dir: Path
) -> None:
    """Q12: PYTHONPATH holds a directory with only `qsiv_runtime`, never the host site-packages."""
    # given
    package = Path(qsiv_runtime.__file__).resolve().parent
    # when
    env = _start(install, sock_dir, {"PYTHONPATH": "/existing"}).envs[0]
    # then
    staged = Path(env["PYTHONPATH"])
    assert env[ENV_SOCKET_DIRECTORY] == str(sock_dir)
    assert ":" not in env["PYTHONPATH"]  # the caller's PYTHONPATH is not inherited
    assert [entry.name for entry in staged.iterdir()] == ["qsiv_runtime"]
    assert (staged / "qsiv_runtime").resolve() == package
    assert staged.parent == sock_dir  # inside the private directory


def test_given_a_staged_package_when_staging_again_then_it_is_reused(tmp_path: Path) -> None:
    """Concurrent or repeated launches leave one symlink."""
    # given / when
    first = stage_package(tmp_path / "p")
    second = stage_package(tmp_path / "p")
    # then
    assert first == second and [e.name for e in first.iterdir()] == ["qsiv_runtime"]


def test_given_the_launch_command_then_it_runs_the_service_module_in_the_venv_python(
    install: Install, sock_dir: Path
) -> None:
    """The argv is `<venv python> -m qsiv_runtime.service <python> <model dir> <runtime>`."""
    # given
    captured: list[list[str]] = []

    class Recording(ScriptedLauncher):
        def start(
            self, argv: Sequence[str], env: Mapping[str, str], log_path: Path
        ) -> FakeProcess | ExitedProcess:
            captured.append(list(argv))
            return super().start(argv, env, log_path)

    launcher = Recording([Generation(replies=[[0.1] * 4])])
    client = QSIVClient(
        install.model,
        install.python,
        "fp8",
        socket_directory=sock_dir,
        environ={},
        machine=FakeMachine(),
        launcher=launcher,
    )
    # when
    client.decide("x", QUESTIONS)
    client.close()
    launcher.stop()
    # then
    assert captured == [
        [
            str(install.python),
            "-m",
            "qsiv_runtime.service",
            str(install.python),
            str(install.model.resolve()),
            "fp8",
        ]
    ]


def test_given_a_bare_interpreter_without_site_packages_when_importing_the_service_then_it_works(
    tmp_path: Path,
) -> None:
    """Q12: the service module needs only the standard library (the GPU venv lacks qsiv_runtime)."""
    # given
    staged = stage_package(tmp_path / "pythonpath")
    # when: -S drops site-packages, so only PYTHONPATH can supply qsiv_runtime
    completed = subprocess.run(
        [sys.executable, "-S", "-m", "qsiv_runtime.service"],
        env={"PYTHONPATH": str(staged)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # then
    assert completed.returncode == 2
    assert "usage:" in completed.stderr
