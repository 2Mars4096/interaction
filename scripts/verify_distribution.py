"""Check packaged native resources in a wheel and its source distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import plistlib
import tarfile
import zipfile


RESOURCE_NAMES = ("macos_speech_bridge.m", "macos_speech_info.plist")
BASELINE_NAMES = ("requirements-dev.lock", "constraints-test.txt")
PLIST_KEYS = ("CFBundleIdentifier", "NSMicrophoneUsageDescription", "NSSpeechRecognitionUsageDescription")


def validate_resources(resources: dict[str, bytes]) -> None:
    for name in RESOURCE_NAMES:
        if not resources.get(name):
            raise ValueError(f"Missing or empty packaged resource: {name}")
    try:
        metadata = plistlib.loads(resources["macos_speech_info.plist"])
    except (ValueError, plistlib.InvalidFileException) as error:
        raise ValueError("The packaged speech Info.plist is invalid.") from error
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(key), str) or not metadata[key].strip() for key in PLIST_KEYS
    ):
        raise ValueError("The speech Info.plist needs a bundle identifier and microphone/speech usage descriptions.")


def _read_sdist_file(archive: tarfile.TarFile, relative_path: str) -> bytes:
    expected = PurePosixPath(relative_path).parts
    matches = [
        member for member in archive.getmembers()
        if len(PurePosixPath(member.name).parts) == len(expected) + 1
        and PurePosixPath(member.name).parts[1:] == expected
    ]
    if len(matches) != 1 or not matches[0].isfile():
        raise ValueError(f"Source distribution must contain exactly one regular {relative_path}.")
    stream = archive.extractfile(matches[0])
    if stream is None:
        raise ValueError(f"Cannot read sdist file: {relative_path}")
    with stream:
        return stream.read()


def verify_distributions(
    dist_dir: Path, *, source_dir: Path | None = None, baseline_dir: Path | None = None,
) -> dict:
    """Inspect archives without extraction and compare resources to the source."""
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one .tar.gz sdist in the distribution directory.")
    source_dir = source_dir or Path(__file__).resolve().parents[1] / "src" / "interaction" / "audio"
    baseline_dir = baseline_dir or Path(__file__).resolve().parents[1]
    source_resources = {name: (source_dir / name).read_bytes() for name in RESOURCE_NAMES}
    validate_resources(source_resources)

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_resources = {}
        for name in RESOURCE_NAMES:
            member_name = f"interaction/audio/{name}"
            matches = [member for member in archive.infolist() if member.filename == member_name]
            if len(matches) != 1 or matches[0].is_dir():
                raise ValueError(f"Wheel must contain exactly one {member_name}.")
            wheel_resources[name] = archive.read(matches[0])

    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_resources = {
            name: _read_sdist_file(archive, f"src/interaction/audio/{name}") for name in RESOURCE_NAMES
        }
        for name in BASELINE_NAMES:
            contents = _read_sdist_file(archive, name)
            if not contents or contents != (baseline_dir / name).read_bytes():
                raise ValueError(f"The sdist must include the current dependency baseline: {name}")

    for label, resources in (("wheel", wheel_resources), ("sdist", sdist_resources)):
        validate_resources(resources)
        if resources != source_resources:
            raise ValueError(f"The {label} native resources differ from the current source files.")
    return {
        "wheel": str(wheels[0].resolve()),
        "sdist": str(sdists[0].resolve()),
        "resources": {name: hashlib.sha256(data).hexdigest() for name, data in source_resources.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    try:
        result = verify_distributions(args.dist_dir)
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Distribution verification failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
