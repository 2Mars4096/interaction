"""Platform adapter interfaces and macOS scaffolding."""

from .base import CommandSpec, PlatformAdapter, PlatformCapabilities
from .macos import MacOSPlatformAdapter
from .macos_runtime import click, double_click, move

__all__ = ["CommandSpec", "PlatformAdapter", "PlatformCapabilities", "MacOSPlatformAdapter", "click", "double_click", "move"]
