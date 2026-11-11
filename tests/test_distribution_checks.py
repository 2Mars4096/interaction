import io
import plistlib
import tarfile
import zipfile

import pytest

from scripts.verify_distribution import BASELINE_NAMES, RESOURCE_NAMES, verify_distributions
from scripts.verify_installed_package import clean_environment, require_installed_path


def build_archives(tmp_path, *, missing=None, changed=None, invalid_plist=False, missing_baseline=None):
    source = tmp_path / "source"
    source.mkdir()
    resources = {
        "macos_speech_bridge.m": b"#import <Foundation/Foundation.h>\nint main() { return 0; }\n",
        "macos_speech_info.plist": plistlib.dumps({
            "CFBundleIdentifier": "org.interaction.speech-helper",
            "NSMicrophoneUsageDescription": "Capture a spoken command.",
            "NSSpeechRecognitionUsageDescription": "Transcribe a spoken command.",
        }),
    }
    if invalid_plist:
        resources["macos_speech_info.plist"] = plistlib.dumps({"CFBundleIdentifier": "org.interaction"})
    for name, contents in resources.items():
        (source / name).write_bytes(contents)
    for name in BASELINE_NAMES:
        (tmp_path / name).write_text("example-package==1.0\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "interaction-0.1.0-py3-none-any.whl", "w") as archive:
        for name, contents in resources.items():
            if name != missing:
                archive.writestr(f"interaction/audio/{name}", b"changed" if name == changed else contents)
    with tarfile.open(dist / "interaction-0.1.0.tar.gz", "w:gz") as archive:
        for name, contents in resources.items():
            member = tarfile.TarInfo(f"interaction-0.1.0/src/interaction/audio/{name}")
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
        for name in BASELINE_NAMES:
            if name != missing_baseline:
                archive.add(tmp_path / name, arcname=f"interaction-0.1.0/{name}")
    return dist, source


def test_distribution_check_requires_matching_native_resources_in_both_archives(tmp_path):
    dist, source = build_archives(tmp_path)
    result = verify_distributions(dist, source_dir=source, baseline_dir=tmp_path)
    assert set(result["resources"]) == set(RESOURCE_NAMES)
    assert all(len(digest) == 64 for digest in result["resources"].values())


@pytest.mark.parametrize("missing", RESOURCE_NAMES)
def test_distribution_check_rejects_missing_wheel_resource(tmp_path, missing):
    dist, source = build_archives(tmp_path, missing=missing)
    with pytest.raises(ValueError, match="Wheel must contain"):
        verify_distributions(dist, source_dir=source, baseline_dir=tmp_path)


def test_distribution_check_rejects_resource_changed_during_packaging(tmp_path):
    dist, source = build_archives(tmp_path, changed="macos_speech_bridge.m")
    with pytest.raises(ValueError, match="differ from the current source"):
        verify_distributions(dist, source_dir=source, baseline_dir=tmp_path)


def test_distribution_check_rejects_missing_native_usage_descriptions(tmp_path):
    dist, source = build_archives(tmp_path, invalid_plist=True)
    with pytest.raises(ValueError, match="usage descriptions"):
        verify_distributions(dist, source_dir=source, baseline_dir=tmp_path)


def test_distribution_check_rejects_missing_sdist_resource(tmp_path):
    dist, source = build_archives(tmp_path)
    with tarfile.open(dist / "interaction-0.1.0.tar.gz", "w:gz"):
        pass
    with pytest.raises(ValueError, match="Source distribution must contain"):
        verify_distributions(dist, source_dir=source, baseline_dir=tmp_path)


@pytest.mark.parametrize("missing_baseline", BASELINE_NAMES)
def test_distribution_check_requires_dependency_baseline_in_sdist(tmp_path, missing_baseline):
    dist, source = build_archives(tmp_path, missing_baseline=missing_baseline)
    with pytest.raises(ValueError, match="Source distribution must contain"):
        verify_distributions(dist, source_dir=source, baseline_dir=tmp_path)


def test_installed_probe_rejects_checkout_and_host_import_origins(tmp_path):
    checkout = tmp_path / "checkout"
    site_packages = tmp_path / "venv" / "lib" / "site-packages"
    require_installed_path(site_packages / "interaction" / "__init__.py", checkout=checkout, site_packages=site_packages)
    for imported in (checkout / "src" / "interaction" / "__init__.py", tmp_path / "host" / "interaction" / "__init__.py"):
        with pytest.raises(ValueError, match="Expected an installed package file"):
            require_installed_path(imported, checkout=checkout, site_packages=site_packages)


def test_probe_environment_removes_import_and_install_redirection(monkeypatch, tmp_path):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        monkeypatch.setenv(name, "/unexpected")
    environment = clean_environment(tmp_path / "venv")
    assert not any(name in environment for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"))
    assert environment["VIRTUAL_ENV"] == str(tmp_path / "venv")
    assert environment["PYTHONNOUSERSITE"] == "1"
