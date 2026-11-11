"""Workspace-local persistence helpers for settings and calibration profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from interaction.vision import CalibrationProfile
from interaction.vision.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_TRACKER_ID,
    CalibrationQuality,
    CalibrationValidationPoint,
    NormalizedPoint,
)


@dataclass(frozen=True)
class RuntimePaths:
    base_dir: Path

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def bin_dir(self) -> Path:
        return self.base_dir / "bin"

    @property
    def profiles_dir(self) -> Path:
        return self.base_dir / "profiles"

    @property
    def settings_path(self) -> Path:
        return self.base_dir / "settings.json"

    def ensure(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def next_session_log_path(self, session_name: str) -> Path:
        self.ensure()
        stem = _slugify(session_name) or "session"
        candidate = self.logs_dir / f"{stem}.jsonl"
        if not candidate.exists():
            return candidate
        index = 2
        while True:
            candidate = self.logs_dir / f"{stem}-{index}.jsonl"
            if not candidate.exists():
                return candidate
            index += 1


@dataclass(frozen=True)
class UserSettings:
    dry_run: bool = True
    push_to_talk: bool = True
    overlay_compact: bool = True
    gaze_context_window_ms: int = 1500
    dwell_ms: int = 700
    camera_index: int = 0


class JsonStateStore:
    """Read and write simple JSON state for repeated local use."""

    def __init__(self, paths: RuntimePaths) -> None:
        self.paths = paths
        self.paths.ensure()

    def load_settings(self) -> UserSettings:
        if not self.paths.settings_path.exists():
            settings = UserSettings()
            self.save_settings(settings)
            return settings
        payload = json.loads(self.paths.settings_path.read_text())
        return UserSettings(**payload)

    def save_settings(self, settings: UserSettings) -> Path:
        self.paths.ensure()
        self.paths.settings_path.write_text(json.dumps(asdict(settings), indent=2) + "\n")
        return self.paths.settings_path

    def load_calibration_profile(
        self,
        name: str = "default",
        *,
        require_validated: bool = False,
        camera_index: int | None = None,
        tracker_id: str = DEFAULT_TRACKER_ID,
    ) -> CalibrationProfile | None:
        path = self.paths.profiles_dir / f"{_slugify(name) or 'default'}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
            profile = _decode_profile(payload)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError):
            return None
        if camera_index is not None and (type(camera_index) is not int or profile.camera_index != camera_index):
            return None
        if profile.tracker_id is not None and profile.tracker_id != tracker_id:
            return None
        if require_validated and not profile.is_usable(camera_index=camera_index, tracker_id=tracker_id):
            return None
        return profile

    def save_calibration_profile(self, profile: CalibrationProfile, *, name: str = "default") -> Path:
        self.paths.ensure()
        path = self.paths.profiles_dir / f"{_slugify(name) or 'default'}.json"
        if profile.quality is not None and not profile.is_usable(tracker_id=profile.tracker_id or DEFAULT_TRACKER_ID):
            raise ValueError("Cannot save inconsistent or unacceptable calibration validation evidence.")
        serialized = json.dumps(asdict(profile), indent=2, allow_nan=False) + "\n"
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.stem}-", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path


def _decode_profile(payload: object) -> CalibrationProfile:
    if not isinstance(payload, dict):
        raise ValueError("Calibration profile must be an object.")
    coefficients = {"x_scale", "y_scale", "x_offset", "y_offset"}
    if "schema_version" not in payload:
        if set(payload) != coefficients:
            raise ValueError("Legacy calibration must contain exactly four coefficients.")
        return CalibrationProfile(**payload, schema_version=1)
    if set(payload) != {field.name for field in fields(CalibrationProfile)}:
        raise ValueError("Calibration profile fields do not match its schema.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] not in {1, CALIBRATION_SCHEMA_VERSION}:
        raise ValueError("Unsupported calibration schema version.")
    if not isinstance(payload["training_expected"], list):
        raise ValueError("Fitting targets must be an array.")
    decoded = dict(payload)
    decoded["training_expected"] = tuple(_decode_point(point) for point in payload["training_expected"])
    decoded["quality"] = _decode_quality(payload["quality"]) if payload["quality"] is not None else None
    profile = CalibrationProfile(**decoded)
    if profile.schema_version == 1 and any((profile.training_expected, profile.quality, profile.camera_index is not None, profile.tracker_id)):
        raise ValueError("Legacy profiles cannot claim versioned validation evidence.")
    if profile.quality is not None and not profile.is_usable(tracker_id=profile.tracker_id or DEFAULT_TRACKER_ID):
        raise ValueError("Calibration validation evidence is inconsistent or outside current acceptance limits.")
    return profile


def _decode_quality(payload: object) -> CalibrationQuality:
    if not isinstance(payload, dict) or set(payload) != {field.name for field in fields(CalibrationQuality)}:
        raise ValueError("Malformed calibration validation quality.")
    for name in ("point_count", "sample_count"):
        if type(payload[name]) is not int or payload[name] < 0:
            raise ValueError("Calibration quality counts must be nonnegative integers.")
    for name in ("mean_error", "max_error", "max_mapped_stddev", "mean_error_limit", "error_limit", "mapped_stddev_limit"):
        if not _finite_number(payload[name]) or payload[name] < 0:
            raise ValueError("Calibration quality measurements must be finite and nonnegative.")
    if not isinstance(payload["points"], list):
        raise ValueError("Calibration quality points must be an array.")
    points = []
    for entry in payload["points"]:
        if not isinstance(entry, dict) or set(entry) != {field.name for field in fields(CalibrationValidationPoint)}:
            raise ValueError("Malformed per-point calibration evidence.")
        if not _finite_number(entry["error"]) or entry["error"] < 0:
            raise ValueError("Calibration point error must be finite and nonnegative.")
        decoded = dict(entry)
        for name in ("raw", "expected", "projected", "raw_stddev", "mapped_stddev"):
            decoded[name] = _decode_point(entry[name])
        points.append(CalibrationValidationPoint(**decoded))
    return CalibrationQuality(**{**payload, "points": tuple(points)})


def _decode_point(payload: object) -> NormalizedPoint:
    if not isinstance(payload, dict) or set(payload) != {"x", "y"} or not all(_finite_number(value) for value in payload.values()):
        raise ValueError("Calibration points must contain finite numeric coordinates.")
    return NormalizedPoint(**payload)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value {value} is not valid calibration evidence.")


def _slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")
