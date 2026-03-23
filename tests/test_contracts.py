import pytest
from pydantic import ValidationError

from interaction.contracts import (
    ActionName,
    ActionProposal,
    BrokerDecision,
    BrokerDecisionType,
    EnvironmentSnapshot,
    ExecutionResult,
    ExecutionStatus,
    GazeObservation,
    GroundedTarget,
    MultimodalContextPacket,
    RiskLevel,
    TranscriptSegment,
)


def test_multimodal_context_packet_round_trip() -> None:
    packet = MultimodalContextPacket(
        transcript=TranscriptSegment(text="open this", is_final=True, confidence=0.92),
        gaze=GazeObservation(target_id="target_1", confidence=0.81),
        environment=EnvironmentSnapshot(active_app="Safari", active_window_title="OpenAI Docs"),
        grounded_targets=[
            GroundedTarget(
                target_id="target_1",
                label="OpenAI Docs tab",
                role="tab",
                confidence=0.76,
            )
        ],
    )

    restored = MultimodalContextPacket.model_validate(packet.model_dump())

    assert restored == packet


def test_gaze_observation_requires_target_or_coordinates() -> None:
    with pytest.raises(ValidationError):
        GazeObservation(confidence=0.4)


def test_confirm_decision_requires_prompt() -> None:
    proposal = ActionProposal(
        action=ActionName.CLICK_TARGET,
        arguments={"target_ref": "target_1"},
        confidence=0.88,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="The user said click this and a target is grounded.",
    )

    with pytest.raises(ValidationError):
        BrokerDecision(
            decision=BrokerDecisionType.CONFIRM,
            reason="Clicks confirm by default in MVP.",
            proposal=proposal,
        )


def test_allow_decision_rejects_l3_proposal() -> None:
    proposal = ActionProposal(
        action=ActionName.TYPE_TEXT,
        arguments={"text": "send it now"},
        confidence=0.91,
        risk=RiskLevel.L3,
        requires_confirmation=True,
        rationale="This would enter text into a sensitive context.",
    )

    with pytest.raises(ValidationError):
        BrokerDecision(
            decision=BrokerDecisionType.ALLOW,
            reason="Should not auto-allow L3 actions.",
            proposal=proposal,
        )


def test_successful_execution_requires_proposal() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            message="Executed successfully.",
        )
