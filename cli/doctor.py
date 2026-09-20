"""Environment and asset diagnostics with privacy-safe output."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _module(name: str, required: bool = True) -> Check:
    found = importlib.util.find_spec(name) is not None
    version = "not installed"
    if found:
        distribution = {
            "cv2": "opencv-python-headless",
            "yaml": "PyYAML",
        }.get(name, name)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "available"
    return Check(name, found, version, required)


def run_checks() -> list[Check]:
    checks = [
        Check("python", sys.version_info[:2] == (3, 10), sys.version.split()[0]),
        Check("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found"),
        _module("torch"),
        _module("numpy"),
        _module("cv2"),
        _module("yaml"),
        _module("pandas"),
        _module("ray"),
        _module("joblib"),
        _module("ultralytics"),
    ]
    for name in ("GeoCalib", "MoGe", "mega-sam", "HaWoR"):
        path = PROJECT_DIR / "third_party" / name
        checks.append(Check(
            f"backend:{name}", path.is_dir(),
            "source available" if path.is_dir() else "source missing",
        ))
    assets = {
        "GeoCalib weights": PROJECT_DIR / "model" / "geocalib" / "pinhole.tar",
        "MoGe weights": PROJECT_DIR / "model" / "moge2" / "model.pt",
        "Mega-SAM weights": PROJECT_DIR / "model" / "megasam" / "megasam_final.pth",
    }
    for name, path in assets.items():
        detail = str(path.relative_to(PROJECT_DIR))
        checks.append(Check(name, path.is_file(), detail, False))
    return checks


def doctor_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the Ego2LeRobot data-pipeline runtime without exposing host details.")
    parser.add_argument("--profile", choices=("data",), default="data")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero status for required failures.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    checks = run_checks()

    if args.as_json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        width = max(len(check.name) for check in checks)
        for check in checks:
            marker = "PASS" if check.ok else ("FAIL" if check.required else "NOTE")
            print(f"[{marker:4}] {check.name:<{width}}  {check.detail}")

    failed = [check for check in checks if check.required and not check.ok]
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(doctor_main())
