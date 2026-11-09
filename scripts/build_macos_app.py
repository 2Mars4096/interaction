#!/usr/bin/env python3
"""Build a local Finder-launchable app using this Mac's Python environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ID = "local.interaction.testapp"


def build(output: Path, python: Path, runtime: Path) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("Build this application on macOS.")
    subprocess.run([str(python), "-c", "import numpy, cv2, pydantic, tkinter"], check=True)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        info = output / "Contents/Info.plist"
        if not info.is_file() or plistlib.loads(info.read_bytes()).get("CFBundleIdentifier") != BUNDLE_ID:
            raise RuntimeError(f"Refusing to replace an unrelated application: {output}")
    with tempfile.TemporaryDirectory(prefix="interaction-app-", dir=output.parent) as temp:
        app = Path(temp) / "Interaction.app"
        contents = app / "Contents"
        resources = contents / "Resources"
        executable = contents / "MacOS/Interaction"
        executable.parent.mkdir(parents=True)
        resources.mkdir()
        shutil.copytree(ROOT / "src/interaction", resources / "engine/interaction", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (resources / "config.json").write_text(json.dumps({"python": str(python.resolve()), "runtime": str(runtime.resolve())}, indent=2))
        info = {
            "CFBundleName": "Interaction", "CFBundleDisplayName": "Interaction",
            "CFBundleIdentifier": BUNDLE_ID, "CFBundleExecutable": "Interaction",
            "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "0.1.0", "CFBundleVersion": "1",
            "LSMinimumSystemVersion": "13.0", "NSPrincipalClass": "NSApplication",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "Interaction listens for voice commands and dictated text when you press Start.",
            "NSSpeechRecognitionUsageDescription": "Interaction transcribes your spoken commands and dictation.",
            "NSCameraUsageDescription": "Interaction uses your webcam for gaze calibration and pointing when you start a gaze test.",
            "NSAppleEventsUsageDescription": "Interaction controls your selected applications when Control my Mac is enabled.",
        }
        (contents / "Info.plist").write_bytes(plistlib.dumps(info))
        subprocess.run(["clang", "-fobjc-arc", "-framework", "Cocoa", "-framework", "Carbon", "-Wall", "-Wextra", "-Wno-unused-parameter",
                        str(ROOT / "scripts/macos/Interaction.m"), "-o", str(executable)], check=True)
        subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(app), str(output))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/Interaction.app")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / ".interaction")
    args = parser.parse_args()
    print(build(args.output, args.python, args.runtime_dir))
