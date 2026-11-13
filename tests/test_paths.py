"""S21 (precedence) and S27 (legacy agemem fallback)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qsiv_runtime.paths import (
    ENV_MODEL_DIRECTORY,
    ENV_PYTHON,
    ENV_RUNTIME,
    ENV_SOCKET_DIRECTORY,
    resolve_paths,
)


def _make(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


def test_given_nothing_set_when_resolving_then_package_defaults(tmp_path: Path) -> None:
    """S21: defaults under ~/.local/share/qsiv-runtime and /tmp/qsiv-runtime-<uid>."""
    # given / when
    paths = resolve_paths(environ={}, home=tmp_path, uid=1234)
    # then
    assert paths.model_directory == tmp_path / ".local/share/qsiv-runtime/model"
    assert paths.python == tmp_path / ".local/share/qsiv-runtime/venv/bin/python"
    assert paths.runtime == "fp8"
    assert paths.socket_directory == Path("/tmp/qsiv-runtime-1234")
    assert paths.legacy is False


def test_given_explicit_environment_and_default_when_resolving_then_explicit_wins(
    tmp_path: Path,
) -> None:
    """S21: explicit argument > environment > default."""
    # given
    _make(tmp_path / ".local/share/qsiv-runtime/model")
    environ = {
        ENV_MODEL_DIRECTORY: str(tmp_path / "env-model"),
        ENV_PYTHON: str(tmp_path / "env-python"),
        ENV_RUNTIME: "reference",
        ENV_SOCKET_DIRECTORY: str(tmp_path / "env-sock"),
    }
    # when
    from_env = resolve_paths(environ=environ, home=tmp_path)
    explicit = resolve_paths(
        tmp_path / "arg-model",
        tmp_path / "arg-python",
        "fp8",
        socket_directory=tmp_path / "arg-sock",
        environ=environ,
        home=tmp_path,
    )
    # then
    assert from_env.model_directory == tmp_path / "env-model"
    assert from_env.python == tmp_path / "env-python"
    assert from_env.runtime == "reference"
    assert from_env.socket_directory == tmp_path / "env-sock"
    assert explicit.model_directory == tmp_path / "arg-model"
    assert explicit.python == tmp_path / "arg-python"
    assert explicit.runtime == "fp8"
    assert explicit.socket_directory == tmp_path / "arg-sock"


def test_given_xdg_data_home_when_resolving_then_it_replaces_local_share(tmp_path: Path) -> None:
    """S21: XDG_DATA_HOME is honoured like agemem does."""
    # given / when
    paths = resolve_paths(environ={"XDG_DATA_HOME": str(tmp_path / "data")}, home=tmp_path)
    # then
    assert paths.model_directory == tmp_path / "data/qsiv-runtime/model"


def test_given_only_a_legacy_install_when_resolving_then_it_is_used(tmp_path: Path) -> None:
    """S27: no second download, the agemem model and runtime are used."""
    # given
    legacy_model = _make(tmp_path / ".local/share/agemem/models/qsiv")
    # when
    paths = resolve_paths(environ={}, home=tmp_path)
    # then
    assert paths.model_directory == legacy_model
    assert paths.python == tmp_path / ".local/share/agemem/runtimes/qsiv/bin/python"
    assert paths.legacy is True


def test_given_both_installs_when_resolving_then_the_new_default_wins(tmp_path: Path) -> None:
    """S27"""
    # given
    _make(tmp_path / ".local/share/agemem/models/qsiv")
    new = _make(tmp_path / ".local/share/qsiv-runtime/model")
    # when
    paths = resolve_paths(environ={}, home=tmp_path)
    # then
    assert paths.model_directory == new
    assert paths.legacy is False


def test_given_a_legacy_install_and_an_explicit_directory_when_resolving_then_no_legacy(
    tmp_path: Path,
) -> None:
    """S21: an explicit directory is never replaced by the legacy location."""
    # given
    _make(tmp_path / ".local/share/agemem/models/qsiv")
    # when
    paths = resolve_paths(tmp_path / "mine", environ={}, home=tmp_path)
    # then
    assert paths.model_directory == tmp_path / "mine"
    assert paths.legacy is False


@pytest.mark.parametrize("runtime", ["int8", "FP8"])
def test_given_an_unknown_runtime_when_resolving_then_value_error(
    tmp_path: Path, runtime: str
) -> None:
    """The runtime is fp8 or reference only."""
    # given / when / then
    with pytest.raises(ValueError, match="fp8 or reference"):
        resolve_paths(environ={ENV_RUNTIME: runtime}, home=tmp_path)


def test_given_the_real_uid_default_then_the_socket_directory_is_package_neutral() -> None:
    """The default does not mention agemem."""
    # given / when
    paths = resolve_paths(environ={})
    # then
    assert paths.socket_directory == Path(f"/tmp/qsiv-runtime-{os.getuid()}")
