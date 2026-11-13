"""`qsiv-runtime setup | doctor`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence

from . import _preflight
from ._ports import CommandRunner, Machine
from .doctor import doctor
from .errors import QSIVError, QSIVUnavailable
from .setup import setup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsiv-runtime", description="Local QSIV model: install it and check the machine."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser(
        "setup", help="create the GPU environment and download the weights"
    )
    install.add_argument("--model-directory", help="where the model goes")
    install.add_argument("--venv-directory", help="where the GPU virtualenv goes")
    install.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    check = commands.add_parser("doctor", help="check this machine and the install (read-only)")
    check.add_argument("--model-directory")
    check.add_argument("--python", help="interpreter of the GPU virtualenv")
    check.add_argument("--runtime", choices=["fp8", "reference"])
    check.add_argument("--verify", action="store_true", help="also hash the weights (slow)")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    machine: Machine | None = None,
    runner: CommandRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Exit 0 only on success; doctor exits 0 only when every check passes."""
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        report = doctor(
            args.model_directory,
            args.python,
            args.runtime,
            verify=args.verify,
            machine=machine,
            environ=environ,
        )
        for check in report.checks:
            status = "ok  " if check.ok else "FAIL"
            cause = f" [{check.cause.value}]" if check.cause else ""
            print(f"{status} {check.name}: {check.detail}{cause}")
        return 0 if report.ok else 1
    try:
        result = setup(
            args.model_directory,
            args.venv_directory,
            dry_run=args.dry_run,
            machine=machine,
            runner=runner,
        )
    except QSIVUnavailable as exc:
        shown = _preflight.tool_hint(exc.diagnostic)
        hint = f" {shown}" if shown else ""
        print(f"qsiv-runtime: {exc.message}.{hint}", file=sys.stderr)
        return 1
    except (QSIVError, subprocess.CalledProcessError) as exc:
        print(f"qsiv-runtime: {exc}", file=sys.stderr)
        return 1
    if result.dry_run:
        print(f"Plan (nothing was changed): model {result.model_directory}")
        for command in result.commands:
            print("  " + " ".join(command))
    else:
        print(f"Installed QSIV {result.revision[:12]} in {result.model_directory}")
        print(f"Runtime python: {result.runtime_python}")
    return 0
