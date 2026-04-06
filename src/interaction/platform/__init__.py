"""Platform adapter interfaces and macOS scaffolding."""

from .base import CommandSpec, PlatformAdapter, PlatformCapabilities, TargetCandidateQuery
from .macos_context import capture_frontmost_environment, parse_frontmost_environment
from .macos import MacOSPlatformAdapter, normalized_to_screen_point
from .macos_profiles import DEFAULT_MACOS_PROFILE_REGISTRY, MacOSProfileRegistry
from .macos_runtime import click, current_position_normalized, double_click, move

__all__ = [
    "CommandSpec",
    "PlatformAdapter",
    "PlatformCapabilities",
    "TargetCandidateQuery",
    "MacOSPlatformAdapter",
    "MacOSProfileRegistry",
    "DEFAULT_MACOS_PROFILE_REGISTRY",
    "capture_frontmost_environment",
    "parse_frontmost_environment",
    "normalized_to_screen_point",
    "click",
    "current_position_normalized",
    "double_click",
    "move",
]
