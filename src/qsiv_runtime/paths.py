"""Where things live: defaults, environment overrides, the legacy agemem fallback (S21, S27)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast, get_args

Runtime = Literal["fp8", "reference"]
ENV_MODEL_DIRECTORY: Final = "QSIV_RUNTIME_MODEL_DIRECTORY"
ENV_PYTHON: Final = "QSIV_RUNTIME_PYTHON"
ENV_RUNTIME: Final = "QSIV_RUNTIME_RUNTIME"
ENV_SOCKET_DIRECTORY: Final = "QSIV_RUNTIME_SOCKET_DIRECTORY"  # tests and unusual hosts
ENV_IDLE_SECONDS: Final = "QSIV_RUNTIME_IDLE_SECONDS"
INSTALL_MARKER: Final = "qsiv-runtime-install.json"
LEGACY_INSTALL_MARKER: Final = "agemem-install.json"


@dataclass(frozen=True)
class ResolvedPaths:
    model_directory: Path
    python: Path  # interpreter of the GPU virtualenv
    runtime: Runtime
    socket_directory: Path  # default /tmp/qsiv-runtime-<uid>
    legacy: bool  # True when the agemem location was used (S27)
    checked: tuple[Path, ...] = ()  # model directories considered, for the "not installed" text


def _uid() -> int:
    return getattr(os, "getuid", lambda: 0)()


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(value)))


def data_directory(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """`$XDG_DATA_HOME` or `~/.local/share`."""
    environ = os.environ if environ is None else environ
    xdg = environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg)
    return (home or Path(environ.get("HOME") or Path.home())) / ".local" / "share"


def parse_runtime(value: str) -> Runtime:
    if value not in get_args(Runtime):
        raise ValueError("QSIV runtime must be fp8 or reference")
    return cast(Runtime, value)


def default_socket_directory(
    environ: Mapping[str, str] | None = None, uid: int | None = None
) -> Path:
    environ = os.environ if environ is None else environ
    explicit = environ.get(ENV_SOCKET_DIRECTORY)
    if explicit:
        return _path(explicit)
    return Path("/tmp") / f"qsiv-runtime-{_uid() if uid is None else uid}"


def resolve_paths(
    model_directory: str | os.PathLike[str] | None = None,
    python: str | os.PathLike[str] | None = None,
    runtime: Runtime | None = None,
    *,
    socket_directory: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    uid: int | None = None,
) -> ResolvedPaths:
    """explicit argument > environment > ~/.local/share/qsiv-runtime/ > legacy agemem (S21, S27)."""
    environ = os.environ if environ is None else environ
    data = data_directory(environ, home)
    chosen_runtime = parse_runtime(runtime or environ.get(ENV_RUNTIME) or "fp8")
    explicit_model = model_directory or environ.get(ENV_MODEL_DIRECTORY)
    explicit_python = python or environ.get(ENV_PYTHON)
    legacy = False
    checked: tuple[Path, ...]
    if explicit_model:
        model = _path(explicit_model)
        checked = (model,)
        default_python = data / "qsiv-runtime" / "venv" / "bin" / "python"
    else:
        model = data / "qsiv-runtime" / "model"
        default_python = data / "qsiv-runtime" / "venv" / "bin" / "python"
        legacy_model = data / "agemem" / "models" / "qsiv"
        checked = (model, legacy_model)
        if not model.exists() and legacy_model.exists():
            model, legacy = legacy_model, True
            default_python = data / "agemem" / "runtimes" / "qsiv" / "bin" / "python"
    if socket_directory:
        sockets = _path(socket_directory)
    else:
        sockets = default_socket_directory(environ, uid)
    return ResolvedPaths(
        model_directory=model,
        python=_path(explicit_python) if explicit_python else default_python,
        runtime=chosen_runtime,
        socket_directory=sockets,
        legacy=legacy,
        checked=checked,
    )
