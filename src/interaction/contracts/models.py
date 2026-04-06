"""Typed runtime contracts for multimodal interaction flows."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Base model for all runtime contract objects."""

    model_config = ConfigDict(extra="forbid")


class PlatformName(StrEnum):
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"
    UNKNOWN = "unknown"


class SessionMode(StrEnum):
    COMMAND = "command"
    DICTATION = "dictation"
    CALIBRATION = "calibration"


class RiskLevel(StrEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class BrokerDecisionType(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    CLARIFY = "clarify"
    REJECT = "reject"


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ActionName(StrEnum):
    HIGHLIGHT_TARGET = "highlight_target"
    FOCUS_TARGET = "focus_target"
    CLICK_TARGET = "click_target"
    DOUBLE_CLICK_TARGET = "double_click_target"
    RIGHT_CLICK_TARGET = "right_click_target"
    DRAG_TARGET = "drag_target"
    SCROLL = "scroll"
    OPEN_APP = "open_app"
    SWITCH_APP = "switch_app"
    PRESS_KEY = "press_key"
    NAVIGATE_STRUCTURE = "navigate_structure"
    TYPE_TEXT = "type_text"
    TRANSLATE_TEXT = "translate_text"
    CLARIFY = "clarify"
    REJECT = "reject"


class NavigationScope(StrEnum):
    APP = "app"
    WINDOW = "window"
    TAB = "tab"
    PANE = "pane"
    SPACE = "space"
    DISPLAY = "display"


class NavigationCommand(StrEnum):
    NEXT = "next"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    NEW = "new"
    CLOSE = "close"
    REOPEN = "reopen"
    INDEX = "index"


class BoundingBox(ContractModel):
    x: float = Field(ge=0.0)
    y: float = Field(ge=0.0)
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)


class ScreenPoint(ContractModel):
    x: float
    y: float


class ScreenBounds(ContractModel):
    x: float
    y: float
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)


class TargetCandidateSource(StrEnum):
    ACCESSIBILITY = "accessibility"
    VISION = "vision"
    OCR = "ocr"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class DesktopTargetCandidate(ContractModel):
    candidate_id: str = Field(min_length=1)
    source: TargetCandidateSource = TargetCandidateSource.UNKNOWN
    display_label: str | None = None
    label: str | None = None
    role: str | None = None
    subrole: str | None = None
    value_text: str | None = None
    normalized_bounds: BoundingBox | None = None
    screen_bounds: ScreenBounds | None = None
    screen_point: ScreenPoint | None = None
    enabled: bool | None = None
    focusable: bool | None = None
    editable: bool | None = None
    invocable: bool | None = None
    available_actions: list[str] = Field(default_factory=list)
    app_name: str | None = None
    bundle_id: str | None = None
    pid: int | None = Field(default=None, ge=0)
    window_title: str | None = None
    window_id: str | None = None
    snapshot_id: str | None = None
    collected_at_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_location_reference(self) -> "DesktopTargetCandidate":
        if self.normalized_bounds is None and self.screen_bounds is None and self.screen_point is None:
            raise ValueError("desktop target candidates require normalized_bounds, screen_bounds, or screen_point")
        return self


class TranscriptSegment(ContractModel):
    text: str = Field(min_length=1)
    is_final: bool = True
    language: str = Field(default="en", min_length=2)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GazeObservation(ContractModel):
    target_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    x_norm: float | None = Field(default=None, ge=0.0, le=1.0)
    y_norm: float | None = Field(default=None, ge=0.0, le=1.0)
    fixation_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_reference(self) -> "GazeObservation":
        has_target = self.target_id is not None
        has_point = self.x_norm is not None and self.y_norm is not None
        if not has_target and not has_point:
            raise ValueError("gaze observation requires a target_id or normalized coordinates")
        return self


class GroundedTarget(ContractModel):
    target_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    role: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    screen_region: str | None = None
    bounds: BoundingBox | None = None
    source: TargetCandidateSource | None = None
    candidate: DesktopTargetCandidate | None = None

    @model_validator(mode="after")
    def validate_candidate_reference(self) -> "GroundedTarget":
        if self.candidate is not None and self.candidate.candidate_id != self.target_id:
            raise ValueError("grounded target candidate_id must match target_id")
        return self


class StructuredNavigationRequest(ContractModel):
    scope: NavigationScope
    command: NavigationCommand
    count: int = Field(default=1, ge=1)
    index: int | None = Field(default=None, ge=1)
    app_profile: str | None = Field(default=None, min_length=1)

    @classmethod
    def from_action_arguments(cls, arguments: dict[str, Any]) -> "StructuredNavigationRequest":
        return cls.model_validate(
            {
                "scope": arguments.get("scope"),
                "command": arguments.get("command"),
                "count": arguments.get("count", 1),
                "index": arguments.get("index"),
                "app_profile": arguments.get("app_profile"),
            }
        )

    @model_validator(mode="after")
    def validate_index_usage(self) -> "StructuredNavigationRequest":
        if self.command == NavigationCommand.INDEX and self.index is None:
            raise ValueError("index navigation requires an index value")
        if self.command != NavigationCommand.INDEX and self.index is not None:
            raise ValueError("only index navigation may include an index value")
        return self


class EnvironmentSnapshot(ContractModel):
    platform: PlatformName = PlatformName.MACOS
    active_app: str | None = None
    active_bundle_id: str | None = None
    active_window_title: str | None = None
    active_profile_id: str | None = None


class SessionState(ContractModel):
    mode: SessionMode = SessionMode.COMMAND
    last_confirmed_action_id: str | None = None


class MultimodalContextPacket(ContractModel):
    transcript: TranscriptSegment | None = None
    gaze: GazeObservation | None = None
    environment: EnvironmentSnapshot
    session: SessionState = Field(default_factory=SessionState)
    grounded_targets: list[GroundedTarget] = Field(default_factory=list)


class NormalizedIntent(ContractModel):
    intent: str = Field(min_length=1)
    target_ref: str | None = None
    modifiers: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    utterance: str | None = None


class ActionProposal(ContractModel):
    action: ActionName
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    risk: RiskLevel
    requires_confirmation: bool
    rationale: str = Field(min_length=1)
    spoken_response: str | None = None

    @model_validator(mode="after")
    def validate_high_risk_confirmation(self) -> "ActionProposal":
        if self.risk == RiskLevel.L3 and not self.requires_confirmation:
            raise ValueError("L3 proposals must require confirmation")
        if self.action == ActionName.NAVIGATE_STRUCTURE:
            StructuredNavigationRequest.from_action_arguments(self.arguments)
        return self


class BrokerDecision(ContractModel):
    decision: BrokerDecisionType
    reason: str = Field(min_length=1)
    proposal: ActionProposal | None = None
    confirmation_prompt: str | None = None
    clarification_prompt: str | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "BrokerDecision":
        if self.decision in {BrokerDecisionType.ALLOW, BrokerDecisionType.CONFIRM} and self.proposal is None:
            raise ValueError("allow and confirm decisions require an action proposal")
        if self.decision == BrokerDecisionType.ALLOW and self.proposal is not None and self.proposal.risk == RiskLevel.L3:
            raise ValueError("L3 proposals cannot be auto-allowed")
        if self.decision == BrokerDecisionType.CONFIRM and not self.confirmation_prompt:
            raise ValueError("confirm decisions require a confirmation_prompt")
        if self.decision == BrokerDecisionType.CLARIFY and not self.clarification_prompt:
            raise ValueError("clarify decisions require a clarification_prompt")
        return self


class ExecutionRequest(ContractModel):
    request_id: str = Field(min_length=1)
    proposal: ActionProposal
    broker_reason: str = Field(min_length=1)
    environment: EnvironmentSnapshot


class ExecutionResult(ContractModel):
    status: ExecutionStatus
    message: str = Field(min_length=1)
    proposal: ActionProposal | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_success_reference(self) -> "ExecutionResult":
        if self.status == ExecutionStatus.SUCCESS and self.proposal is None:
            raise ValueError("successful execution results require a proposal")
        return self
