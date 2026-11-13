"""What ships: version, entry points, the trimmed lock, typed marker, stdlib-only client."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import qsiv_runtime

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(qsiv_runtime.__file__).resolve().parent
PYPROJECT = (ROOT / "pyproject.toml").read_text()


def test_given_the_project_then_the_version_matches_the_package() -> None:
    # given / when
    match = re.search(r'^version = "([^"]+)"', PYPROJECT, re.MULTILINE)
    # then
    assert match is not None
    assert match.group(1) == qsiv_runtime.__version__ == "0.1.0"


def test_given_the_project_then_it_declares_the_cli_and_python_floor() -> None:
    assert 'qsiv-runtime = "qsiv_runtime.cli:main"' in PYPROJECT
    assert 'requires-python = ">=3.10"' in PYPROJECT
    assert re.search(r"^dependencies = \[\]$", PYPROJECT, re.MULTILINE)  # stdlib-only client


def test_given_the_public_names_then_they_are_importable() -> None:
    for name in qsiv_runtime.__all__:
        assert hasattr(qsiv_runtime, name)
    assert {c.value for c in qsiv_runtime.Cause} == {
        "no_gpu",
        "not_installed",
        "startup_failed",
        "service_exited",
        "model_in_use",
        "invalid_answer",
        "unsafe_directory",
        "bad_checksum",
        "too_large",
        "timeout",
    }


def test_given_the_package_then_it_ships_the_typed_marker_and_the_lock() -> None:
    assert (PACKAGE / "py.typed").is_file()
    assert (PACKAGE / "data" / "qsiv-linux-py312.lock").is_file()


def test_given_the_lock_then_agemem_only_dependencies_are_gone_and_its_gpu_check_is_recorded() -> (
    None
):
    """S29: the lock says when it was last installed and run on a real GPU (manual check)."""
    # given
    text = (PACKAGE / "data" / "qsiv-linux-py312.lock").read_text()
    pinned = {
        line.split("==")[0] for line in text.splitlines() if line and not line.startswith("#")
    }
    # then
    assert not pinned & {
        "mcp",
        "mcp-types",
        "fastembed",
        "onnxruntime",
        "sqlite-vec",
        "typesafe-sdk",
    }
    assert {"torch", "transformers", "safetensors", "huggingface-hub", "numpy"} <= pinned
    assert "# Verified 2026-11-13 on an RTX 4070 Laptop GPU" in text


def test_given_the_runtime_modules_then_they_import_only_the_standard_library() -> None:
    """Q12: the service is started through PYTHONPATH in a venv that has no qsiv_runtime."""
    # given
    allowed = set(sys.stdlib_module_names) | {"qsiv_runtime"}
    offenders: list[str] = []
    # when: all imports; torch and qsiv are imported lazily by load_engine
    lazy = {"torch", "qsiv", "huggingface_hub"}
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            offenders += [f"{path.name}: {n}" for n in names if n not in allowed | lazy]
    # then
    assert offenders == []


def test_given_the_top_level_imports_then_torch_is_not_imported() -> None:
    assert "torch" not in sys.modules


def test_given_the_licenses_then_mit_and_the_qsiv_notice_are_present() -> None:
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    third_party = (ROOT / "THIRD_PARTY.md").read_text()
    assert "Apache-2.0" in third_party and "NOTICE" in third_party
