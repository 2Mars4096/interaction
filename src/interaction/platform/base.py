"""Platform adapter abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from interaction.contracts import DesktopTargetCandidate, ExecutionRequest, ExecutionResult, ScreenPoint


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]


@dataclass(frozen=True)
class PlatformCapabilities:
    platform_name: str
    supported_actions: tuple[str, ...]
    coordinate_actions: tuple[str, ...] = ()
    candidate_collection: bool = False
    structured_navigation_scopes: tuple[str, ...] = ()
    app_profiles: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetCandidateQuery:
    screen_point: ScreenPoint
    radius_px: float = 48.0
    limit: int = 8
    active_app: str | None = None
    active_window_title: str | None = None
    require_actionable: bool = True

    def __post_init__(self) -> None:
        if self.radius_px <= 0.0:
            raise ValueError("radius_px must be positive")
        if self.limit < 1:
            raise ValueError("limit must be at least 1")


class PlatformAdapter(ABC):
    """Execute broker-approved requests on a specific platform."""

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute an approved request and return an execution result."""

    @abstractmethod
    def describe_capabilities(self) -> PlatformCapabilities:
        """Describe the currently supported action surface for the adapter."""

    def collect_target_candidates(self, query: TargetCandidateQuery) -> tuple[DesktopTargetCandidate, ...]:
        """Return candidate desktop targets near a screen point when supported."""
        return ()
