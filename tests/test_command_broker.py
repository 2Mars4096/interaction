from interaction.contracts import (
    ActionName,
    ActionProposal,
    BrokerDecisionType,
    EnvironmentSnapshot,
    RiskLevel,
)
from interaction.control import CommandBroker


def test_scroll_proposal_is_allowed() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.SCROLL,
        arguments={"direction": "down"},
        confidence=0.93,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Scroll is a reversible navigation action.",
    )

    decision = broker.decide(proposal)

    assert decision.decision == BrokerDecisionType.ALLOW


def test_click_proposal_requires_confirmation() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.CLICK_TARGET,
        arguments={"target_ref": "target_1"},
        confidence=0.84,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Clicking changes local application state.",
    )

    decision = broker.decide(proposal)

    assert decision.decision == BrokerDecisionType.CONFIRM
    assert decision.confirmation_prompt is not None


def test_l3_proposal_is_rejected() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.TYPE_TEXT,
        arguments={"text": "buy now"},
        confidence=0.9,
        risk=RiskLevel.L3,
        requires_confirmation=True,
        rationale="This could submit sensitive content.",
    )

    decision = broker.decide(proposal)

    assert decision.decision == BrokerDecisionType.REJECT


def test_safe_navigation_key_is_allowed() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.PRESS_KEY,
        arguments={"key": "escape"},
        confidence=0.88,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Escape is a safe navigation key in MVP.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))

    assert decision.decision == BrokerDecisionType.ALLOW
    assert request.proposal.action == ActionName.PRESS_KEY


def test_structured_navigation_is_allowed_when_low_risk() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.NAVIGATE_STRUCTURE,
        arguments={"scope": "tab", "command": "next"},
        confidence=0.9,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Tab next is a reversible shortcut command.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(
        decision,
        EnvironmentSnapshot(active_app="Google Chrome"),
    )

    assert decision.decision == BrokerDecisionType.ALLOW
    assert "tab next" in decision.reason
    assert request.proposal.action == ActionName.NAVIGATE_STRUCTURE


def test_unsupported_structured_navigation_is_rejected_with_profile_reason() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.NAVIGATE_STRUCTURE,
        arguments={
            "scope": "pane",
            "command": "right",
            "profile_supported": False,
            "profile_reject_reason": "Pane navigation is not supported for Mail.",
        },
        confidence=0.88,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Profile validation already rejected the command.",
    )

    decision = broker.decide(proposal)

    assert decision.decision == BrokerDecisionType.REJECT
    assert decision.reason == "Pane navigation is not supported for Mail."


def test_confirm_decision_can_be_promoted_after_user_confirmation() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.TYPE_TEXT,
        arguments={"text": "hello"},
        confidence=0.91,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Typing changes state.",
    )

    decision = broker.decide(proposal)
    promoted = broker.confirm(decision)

    assert decision.decision == BrokerDecisionType.CONFIRM
    assert promoted.decision == BrokerDecisionType.ALLOW
