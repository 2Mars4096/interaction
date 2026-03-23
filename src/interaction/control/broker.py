"""Command broker implementation for proposal review and execution preparation."""

from __future__ import annotations

from uuid import uuid4

from interaction.contracts import (
    ActionName,
    ActionProposal,
    BrokerDecision,
    BrokerDecisionType,
    EnvironmentSnapshot,
    ExecutionRequest,
)

from .policy import BrokerPolicy


class CommandBroker:
    """Evaluate action proposals against policy and prepare execution requests."""

    def __init__(self, policy: BrokerPolicy | None = None) -> None:
        self.policy = policy or BrokerPolicy()

    def decide(self, proposal: ActionProposal) -> BrokerDecision:
        decision = self.policy.default_decision_for(
            proposal.action,
            risk=proposal.risk,
            arguments=proposal.arguments,
        )
        if proposal.requires_confirmation and decision == BrokerDecisionType.ALLOW:
            decision = BrokerDecisionType.CONFIRM

        if decision == BrokerDecisionType.ALLOW:
            return BrokerDecision(
                decision=decision,
                reason=self._allow_reason(proposal),
                proposal=proposal,
            )
        if decision == BrokerDecisionType.CONFIRM:
            return BrokerDecision(
                decision=decision,
                reason=self._confirm_reason(proposal),
                proposal=proposal,
                confirmation_prompt=self._confirmation_prompt(proposal),
            )
        if decision == BrokerDecisionType.CLARIFY:
            return BrokerDecision(
                decision=decision,
                reason="The proposal requires clarification before any action can be executed.",
                clarification_prompt=proposal.spoken_response or "What exactly would you like me to do?",
            )
        return BrokerDecision(
            decision=BrokerDecisionType.REJECT,
            reason=self._reject_reason(proposal),
            proposal=proposal,
        )

    def confirm(self, decision: BrokerDecision) -> BrokerDecision:
        if decision.decision != BrokerDecisionType.CONFIRM or decision.proposal is None:
            raise ValueError("Only confirm decisions with proposals can be promoted to allow.")
        return BrokerDecision(
            decision=BrokerDecisionType.ALLOW,
            reason=f"User confirmed the action. {decision.reason}",
            proposal=decision.proposal,
        )

    def build_execution_request(
        self,
        decision: BrokerDecision,
        environment: EnvironmentSnapshot,
        *,
        request_id: str | None = None,
    ) -> ExecutionRequest:
        if decision.decision != BrokerDecisionType.ALLOW or decision.proposal is None:
            raise ValueError("Execution requests require an allow decision with a proposal.")
        return ExecutionRequest(
            request_id=request_id or f"exec_{uuid4().hex[:12]}",
            proposal=decision.proposal,
            broker_reason=decision.reason,
            environment=environment,
        )

    def _allow_reason(self, proposal: ActionProposal) -> str:
        return f"Action {proposal.action.value} is allowed by the current MVP broker policy."

    def _confirm_reason(self, proposal: ActionProposal) -> str:
        if proposal.action in {ActionName.CLICK_TARGET, ActionName.DOUBLE_CLICK_TARGET, ActionName.RIGHT_CLICK_TARGET, ActionName.DRAG_TARGET}:
            return "Pointer actions require confirmation by default in MVP."
        if proposal.action == ActionName.TYPE_TEXT:
            return "Typing text changes application state and requires confirmation in MVP."
        if proposal.action == ActionName.TRANSLATE_TEXT:
            return "Text transformation actions require confirmation in MVP."
        if proposal.action == ActionName.PRESS_KEY:
            return "Only a small allow-list of navigation keys auto-executes in MVP."
        return "This action requires confirmation under the current broker policy."

    def _confirmation_prompt(self, proposal: ActionProposal) -> str:
        if proposal.action in {ActionName.CLICK_TARGET, ActionName.DOUBLE_CLICK_TARGET, ActionName.RIGHT_CLICK_TARGET, ActionName.DRAG_TARGET}:
            if proposal.action == ActionName.DRAG_TARGET:
                return "Confirm dragging from the current source to the current target?"
            if proposal.action == ActionName.RIGHT_CLICK_TARGET:
                return "Confirm right-clicking the grounded target?"
            return "Confirm clicking the grounded target?"
        if proposal.action == ActionName.TYPE_TEXT:
            text = str(proposal.arguments.get("text", "")).strip()
            if text:
                return f'Confirm typing: "{text}"?'
            return "Confirm typing into the active field?"
        if proposal.action == ActionName.TRANSLATE_TEXT:
            language = str(proposal.arguments.get("target_language", "")).strip()
            if language:
                return f"Confirm translating the selected text into {language}?"
            return "Confirm translating the selected text?"
        if proposal.action == ActionName.PRESS_KEY:
            key = str(proposal.arguments.get("key", "")).strip()
            return f"Confirm pressing {key or 'the requested key'}?"
        return proposal.spoken_response or "Confirm this action?"

    def _reject_reason(self, proposal: ActionProposal) -> str:
        if proposal.risk.value == "L3":
            return "L3 actions are blocked by default in the MVP broker policy."
        if proposal.action == ActionName.REJECT:
            return "The proposal explicitly rejected the request."
        return "This action is rejected by the current broker policy."
