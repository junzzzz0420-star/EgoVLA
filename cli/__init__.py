"""Command-line entry for the Ego2LeRobot data pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


__version__ = "0.1.0"
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _run_script(relative: str, args: list[str]) -> int:
    return subprocess.call([sys.executable, str(PROJECT_DIR / relative), *args])


def _normalize_pipeline_args(args: list[str]) -> list[str]:
    aliases = {
        "--num-gpus": "--num_gpus",
        "--retry-failed": "--retry_failed",
        "--long-video-threshold-s": "--long_video_threshold_s",
    }
    return [aliases.get(item, item) for item in args]


def main(argv: list[str] | None = None) -> int:
    import argparse

    raw = list(sys.argv[1:] if argv is None else argv)
    commands = ("pipeline", "doctor")
    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description="Ego2LeRobot: egocentric video to LeRobot data pipeline",
        epilog="Use 'python -m cli <command> --help' for command-specific options.",
    )
    parser.add_argument("command", nargs="?", choices=commands)
    if not raw or raw[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if raw[0] not in commands:
        parser.error(f"unknown command: {raw[0]}")
    command, remainder = raw[0], raw[1:]

    if command == "pipeline":
        return _run_script("ego_pipeline/run_batch.py", _normalize_pipeline_args(remainder))
    from .doctor import doctor_main

    return doctor_main(remainder)
