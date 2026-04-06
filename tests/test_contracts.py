import pytest
from pydantic import ValidationError

from interaction.contracts import (
    ActionName,
    ActionProposal,
    BrokerDecision,
    BrokerDecisionType,
    DesktopTargetCandidate,
    EnvironmentSnapshot,
    ExecutionResult,
    ExecutionStatus,
    GazeObservation,
    GroundedTarget,
    MultimodalContextPacket,
    NavigationCommand,
    NavigationScope,
    RiskLevel,
    ScreenBounds,
    ScreenPoint,
    StructuredNavigationRequest,
    TargetCandidateSource,
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


def test_structured_navigation_request_validates_index_usage() -> None:
    request = StructuredNavigationRequest(scope=NavigationScope.TAB, command=NavigationCommand.INDEX, index=3)

    assert request.index == 3

    with pytest.raises(ValidationError):
        StructuredNavigationRequest(scope=NavigationScope.TAB, command=NavigationCommand.INDEX)
    with pytest.raises(ValidationError):
        StructuredNavigationRequest(scope=NavigationScope.TAB, command=NavigationCommand.NEXT, index=3)


def test_navigation_action_proposal_requires_typed_navigation_arguments() -> None:
    proposal = ActionProposal(
        action=ActionName.NAVIGATE_STRUCTURE,
        arguments={"scope": "tab", "command": "next"},
        confidence=0.9,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Structured tab navigation is reversible.",
    )

    assert proposal.action == ActionName.NAVIGATE_STRUCTURE

    with pytest.raises(ValidationError):
        ActionProposal(
            action=ActionName.NAVIGATE_STRUCTURE,
            arguments={"scope": "tab", "command": "index"},
            confidence=0.9,
            risk=RiskLevel.L1,
            requires_confirmation=False,
            rationale="Missing index should fail validation.",
        )


def test_desktop_target_candidate_round_trip_supports_real_screen_geometry() -> None:
    candidate = DesktopTargetCandidate(
        candidate_id="candidate_1",
        source=TargetCandidateSource.ACCESSIBILITY,
        display_label="Bold button",
        label="Bold",
        role="button",
        screen_bounds=ScreenBounds(x=-1280.0, y=84.0, width=24.0, height=24.0),
        screen_point=ScreenPoint(x=-1268.0, y=96.0),
        app_name="Pages",
        window_title="Document 1",
        available_actions=["press"],
        invocable=True,
        enabled=True,
        collected_at_ms=1234,
    )
    target = GroundedTarget(
        target_id="candidate_1",
        label="Bold button",
        role="button",
        confidence=0.91,
        source=TargetCandidateSource.ACCESSIBILITY,
        candidate=candidate,
    )

    restored = GroundedTarget.model_validate(target.model_dump())

    assert restored == target
    assert restored.candidate is not None
    assert restored.candidate.screen_bounds is not None
    assert restored.candidate.screen_bounds.x == pytest.approx(-1280.0)


def test_desktop_target_candidate_requires_location_reference() -> None:
    with pytest.raises(ValidationError):
        DesktopTargetCandidate(
            candidate_id="candidate_1",
            source=TargetCandidateSource.ACCESSIBILITY,
        )


def test_grounded_target_rejects_mismatched_candidate_reference() -> None:
    candidate = DesktopTargetCandidate(
        candidate_id="candidate_2",
        source=TargetCandidateSource.ACCESSIBILITY,
        screen_point=ScreenPoint(x=640.0, y=480.0),
    )

    with pytest.raises(ValidationError):
        GroundedTarget(
            target_id="candidate_1",
            label="Mismatch",
            role="button",
            confidence=0.6,
            candidate=candidate,
        )
