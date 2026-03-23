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
    TYPE_TEXT = "type_text"
    TRANSLATE_TEXT = "translate_text"
    CLARIFY = "clarify"
    REJECT = "reject"


class BoundingBox(ContractModel):
    x: float = Field(ge=0.0)
    y: float = Field(ge=0.0)
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)


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


class EnvironmentSnapshot(ContractModel):
    platform: PlatformName = PlatformName.MACOS
    active_app: str | None = None
    active_window_title: str | None = None


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
