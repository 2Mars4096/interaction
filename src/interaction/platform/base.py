"""Platform adapter abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from interaction.contracts import ExecutionRequest, ExecutionResult


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]


@dataclass(frozen=True)
class PlatformCapabilities:
    platform_name: str
    supported_actions: tuple[str, ...]
    coordinate_actions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class PlatformAdapter(ABC):
    """Execute broker-approved requests on a specific platform."""

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute an approved request and return an execution result."""

    @abstractmethod
    def describe_capabilities(self) -> PlatformCapabilities:
        """Describe the currently supported action surface for the adapter."""
