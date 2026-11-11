"""Install a wheel in a clean environment and verify it outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import venv
import zipfile


RESOURCE_NAMES = ("macos_speech_bridge.m", "macos_speech_info.plist")


def clean_environment(venv_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        environment.pop(key, None)
    environment.update({
        "VIRTUAL_ENV": str(venv_dir),
        "PYTHONNOUSERSITE": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })
    return environment


def require_installed_path(path: Path, *, checkout: Path, site_packages: Path) -> None:
    resolved = path.resolve()
    if resolved.is_relative_to(checkout.resolve()) or not resolved.is_relative_to(site_packages.resolve()):
        raise ValueError(f"Expected an installed package file, not checkout/host content: {resolved}")


def probe_installation(checkout: Path, expected_resources: dict[str, str], *, compile_native: bool) -> dict:
    """Run only inside the isolated child interpreter created below."""
    if not sys.flags.isolated or sys.prefix == sys.base_prefix:
        raise ValueError("The installed-package probe must use an isolated interpreter in a virtual environment.")
    if Path.cwd().resolve().is_relative_to(checkout.resolve()):
        raise ValueError("Run the installed-package probe outside the source checkout.")
    if any(Path(path).resolve().is_relative_to(checkout.resolve()) for path in sys.path):
        raise ValueError("The source checkout leaked into the installed interpreter's import path.")

    import cv2
    import interaction
    import interaction.app
    import interaction.audio

    site_packages = Path(sysconfig.get_path("purelib"))
    for module in (interaction, interaction.app, interaction.audio, cv2):
        require_installed_path(Path(module.__file__), checkout=checkout, site_packages=site_packages)
    distribution = importlib.metadata.distribution("interaction")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise ValueError("The probe loaded an editable distribution instead of the wheel.")
    entrypoints = [entry for entry in distribution.entry_points if entry.group == "console_scripts" and entry.name == "interaction"]
    if len(entrypoints) != 1 or entrypoints[0].value != "interaction.app:cli" or not callable(entrypoints[0].load()):
        raise ValueError("The installed interaction console entrypoint is missing or incorrect.")

    packaged_resources = {}
    for name in RESOURCE_NAMES:
        resource = importlib.resources.files("interaction.audio").joinpath(name)
        if not resource.is_file():
            raise ValueError(f"Installed native resource is missing: {name}")
        with importlib.resources.as_file(resource) as path:
            require_installed_path(path, checkout=checkout, site_packages=site_packages)
            data = path.read_bytes()
        if not data or hashlib.sha256(data).hexdigest() != expected_resources.get(name):
            raise ValueError(f"Installed native resource differs from the wheel: {name}")
        packaged_resources[name] = data
    info = plistlib.loads(packaged_resources["macos_speech_info.plist"])
    for key in ("CFBundleIdentifier", "NSMicrophoneUsageDescription", "NSSpeechRecognitionUsageDescription"):
        if not isinstance(info, dict) or not isinstance(info.get(key), str) or not info[key].strip():
            raise ValueError(f"The installed speech Info.plist needs a nonempty {key}.")

    cascades = ("haarcascade_frontalface_default.xml", "haarcascade_eye_tree_eyeglasses.xml", "haarcascade_eye.xml")
    for filename in cascades:
        path = Path(cv2.data.haarcascades) / filename
        if not path.is_file() or cv2.CascadeClassifier(str(path)).empty():
            raise ValueError(f"OpenCV Haar cascade is unavailable: {filename}")

    entrypoint = Path(sys.executable).parent / "interaction"
    if not entrypoint.is_file():
        raise ValueError("The installed console executable is missing.")
    result = subprocess.run(
        [str(entrypoint), "--help"], check=True, capture_output=True, text=True,
        env=clean_environment(Path(sys.prefix)), timeout=30,
    )
    if "gaze-calibrate" not in result.stdout or "voice-live" not in result.stdout:
        raise ValueError("The installed console executable did not expose the expected commands.")

    native_binary = None
    if compile_native:
        if platform.system() != "Darwin":
            raise ValueError("Native speech compilation is supported only on macOS.")
        from interaction.audio import MacOSLiveSpeechProvider
        from interaction.persistence import RuntimePaths

        provider = MacOSLiveSpeechProvider(RuntimePaths(Path.cwd() / "native-build"))
        native_binary = provider.ensure_helper()
        if not native_binary.is_file() or not native_binary.stat().st_size or not os.access(native_binary, os.X_OK):
            raise ValueError("The packaged native helper did not produce an executable.")
        original_mtime = native_binary.stat().st_mtime_ns
        if provider.ensure_helper() != native_binary or native_binary.stat().st_mtime_ns != original_mtime:
            raise ValueError("The second helper build did not reuse its valid cached binary.")
        # Invalid arguments must fail before permission APIs or audio startup.
        # This exercises the installed binary's versioned stream envelope while
        # keeping CI independent of microphone/speech authorization.
        for option, value in (("--duration", "0"), ("--duration", "31"),
                              ("--silence-duration", "0.1"), ("--silence-duration", "nan")):
            result = subprocess.run(
                [str(native_binary), "--stream", "--locale", "fr-FR", option, value],
                check=False, capture_output=True, text=True, timeout=5,
            )
            lines = result.stdout.splitlines()
            if result.returncode != 2 or len(lines) != 1:
                raise ValueError("The installed native stream probe did not reject invalid arguments before capture.")
            event = json.loads(lines[0])
            expected = {"schema_version": 1, "sequence": 1, "provider": "macos_speech", "locale": "fr-FR",
                        "event": "error", "error": "invalid_capture_arguments"}
            if not isinstance(event, dict) or any(event.get(key) != value for key, value in expected.items()):
                raise ValueError("The installed native helper returned an invalid stream error envelope.")
            if not isinstance(event.get("message"), str) or not event["message"].strip():
                raise ValueError("The installed native helper returned an empty stream error message.")

    return {
        "python": sys.version.split()[0], "package_path": str(Path(interaction.__file__).resolve()),
        "entrypoint": str(entrypoint), "opencv": cv2.__version__, "haar_cascades": list(cascades),
        "resources": expected_resources, "native_compile": "passed" if native_binary is not None else "not_requested",
        "native_stream_arguments": "passed" if native_binary is not None else "not_requested",
    }


def verify_wheel(
    wheel: Path, requirements: Path, *, compile_native: bool = False,
    find_links: Path | None = None, no_index: bool = False,
) -> dict:
    wheel = wheel.resolve(strict=True)
    requirements = requirements.resolve(strict=True)
    checkout = Path(__file__).resolve().parents[1]
    if wheel.suffix != ".whl":
        raise ValueError("The installed-package check requires a built wheel.")
    with zipfile.ZipFile(wheel) as archive:
        expected_resources = {
            name: hashlib.sha256(archive.read(f"interaction/audio/{name}")).hexdigest() for name in RESOURCE_NAMES
        }
    with tempfile.TemporaryDirectory(prefix="interaction-installed-") as temporary:
        workspace = Path(temporary).resolve()
        if workspace.is_relative_to(checkout):
            raise ValueError("The temporary directory must be outside the source checkout; choose another TMPDIR.")
        venv_dir = workspace / "venv"
        work_dir = workspace / "probe"
        work_dir.mkdir()
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(venv_dir)
        python = venv_dir / "bin" / "python"
        environment = clean_environment(venv_dir)
        installer = [str(python), "-I", "-m", "pip", "install", "--quiet"]
        dependency_options = ["--require-hashes", "--only-binary=:all:", "-r", str(requirements)]
        if find_links is not None:
            dependency_options.extend(["--find-links", str(find_links.resolve(strict=True))])
        if no_index:
            dependency_options.append("--no-index")
        subprocess.run(installer + dependency_options, cwd=work_dir, env=environment, check=True, timeout=300)
        subprocess.run(installer + ["--no-deps", str(wheel)], cwd=work_dir, env=environment, check=True, timeout=60)
        subprocess.run([str(python), "-I", "-m", "pip", "check"], cwd=work_dir, env=environment, check=True, timeout=30)
        probe = work_dir / "verify_installed_package.py"
        shutil.copyfile(Path(__file__), probe)
        command = [
            str(python), "-I", str(probe), "--probe", "--checkout", str(checkout),
            "--expected-resources", json.dumps(expected_resources),
        ]
        if compile_native:
            command.append("--compile-native")
        result = subprocess.run(command, cwd=work_dir, env=environment, check=True, text=True, capture_output=True, timeout=180)
        return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--requirements", type=Path, default=Path("requirements-dev.lock"))
    parser.add_argument("--compile-native", action="store_true")
    parser.add_argument("--find-links", type=Path, help="Optional local wheel directory for dependency installation.")
    parser.add_argument("--no-index", action="store_true", help="Install locked dependencies using local wheels/cache only.")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-resources", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.probe:
            if args.checkout is None or args.expected_resources is None:
                raise ValueError("The internal probe requires checkout and resource metadata.")
            result = probe_installation(args.checkout, json.loads(args.expected_resources), compile_native=args.compile_native)
        else:
            wheels = [args.wheel] if args.wheel else sorted(args.dist_dir.glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError("Expected exactly one wheel; use --wheel to select it explicitly.")
            result = verify_wheel(
                wheels[0], args.requirements, compile_native=args.compile_native,
                find_links=args.find_links, no_index=args.no_index,
            )
    except subprocess.CalledProcessError as error:
        details = error.stderr or str(error)
        parser.exit(1, f"Installed-package verification failed: {details}\n")
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired, zipfile.BadZipFile) as error:
        parser.exit(1, f"Installed-package verification failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
