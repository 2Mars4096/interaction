"""Workspace-local persistence helpers for settings and calibration profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from interaction.vision import CalibrationProfile


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

    def load_calibration_profile(self, name: str = "default") -> CalibrationProfile | None:
        path = self.paths.profiles_dir / f"{_slugify(name) or 'default'}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return CalibrationProfile(**payload)

    def save_calibration_profile(self, profile: CalibrationProfile, *, name: str = "default") -> Path:
        self.paths.ensure()
        path = self.paths.profiles_dir / f"{_slugify(name) or 'default'}.json"
        path.write_text(json.dumps(asdict(profile), indent=2) + "\n")
        return path


def _slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")
