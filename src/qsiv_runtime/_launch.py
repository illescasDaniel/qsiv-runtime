"""How the service process is launched: a scrubbed environment and a minimal PYTHONPATH."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from .paths import ENV_SOCKET_DIRECTORY

_SECRET_WORDS: Final = (
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "ACCESS_KEY",
)
_SECRET_SEGMENT: Final = "AUTH"
PACKAGE_DIRECTORY: Final = Path(__file__).resolve().parent


def stage_package(directory: Path) -> Path:
    """A directory holding only a `qsiv_runtime` symlink, for PYTHONPATH (Q12).

    The package's own parent directory cannot be used: for a wheel install it is the host
    environment's whole site-packages, which would shadow the pinned packages of the GPU
    environment. The symlink is replaced atomically, so concurrent launches are safe."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    link = directory / "qsiv_runtime"
    if not (link.is_symlink() and link.resolve() == PACKAGE_DIRECTORY):
        pending = directory / f".qsiv_runtime.{os.getpid()}"
        pending.unlink(missing_ok=True)
        pending.symlink_to(PACKAGE_DIRECTORY, target_is_directory=True)
        os.replace(pending, link)
    return directory


def is_secret_name(name: str) -> bool:
    """The scrub rule (S26), case-insensitive: the name contains API_KEY, APIKEY, TOKEN, SECRET,
    PASSWORD, PASSWD, CREDENTIAL, PRIVATE_KEY or ACCESS_KEY anywhere, or has a whole
    `_`/`-`/`.`-separated segment equal to AUTH (SSH_AUTH_SOCK; AUTHOR and OAUTH_X stay)."""
    upper = name.upper()
    return any(w in upper for w in _SECRET_WORDS) or _SECRET_SEGMENT in re.split(r"[_.\-]", upper)


def build_service_env(environ: Mapping[str, str], socket_directory: Path) -> dict[str, str]:
    """No secrets (S26), Hugging Face offline, only the staged package on PYTHONPATH (Q12: the GPU
    virtualenv does not have qsiv_runtime installed), and PYTHONSAFEPATH so the working directory
    is not put on sys.path (python >= 3.11; older interpreters ignore the variable)."""
    env = {k: v for k, v in environ.items() if not is_secret_name(k)}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONSAFEPATH="1")
    env[ENV_SOCKET_DIRECTORY] = str(socket_directory)
    staged = str(stage_package(socket_directory / "pythonpath"))
    env["PYTHONPATH"] = staged  # never the caller's PYTHONPATH: only the staged symlink directory
    return env
