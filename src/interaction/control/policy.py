"""Default policy rules for broker decisions."""

from __future__ import annotations

from interaction.contracts import ActionName, BrokerDecisionType, RiskLevel

SAFE_NAVIGATION_KEYS = {"escape", "tab", "enter", "back"}


class BrokerPolicy:
    """Default MVP policy for converting proposals into broker decisions."""

    def default_decision_for(self, action: ActionName, *, risk: RiskLevel, arguments: dict[str, object]) -> BrokerDecisionType:
        if action == ActionName.CLARIFY:
            return BrokerDecisionType.CLARIFY
        if action == ActionName.REJECT:
            return BrokerDecisionType.REJECT
        if risk == RiskLevel.L3:
            return BrokerDecisionType.REJECT
        if action in {
            ActionName.CLICK_TARGET,
            ActionName.DOUBLE_CLICK_TARGET,
            ActionName.RIGHT_CLICK_TARGET,
            ActionName.TYPE_TEXT,
            ActionName.TRANSLATE_TEXT,
        }:
            return BrokerDecisionType.CONFIRM
        if action == ActionName.PRESS_KEY:
            key = str(arguments.get("key", "")).strip().lower()
            if key in SAFE_NAVIGATION_KEYS and risk in {RiskLevel.L0, RiskLevel.L1}:
                return BrokerDecisionType.ALLOW
            return BrokerDecisionType.CONFIRM
        if action in {
            ActionName.HIGHLIGHT_TARGET,
            ActionName.FOCUS_TARGET,
            ActionName.SCROLL,
            ActionName.OPEN_APP,
            ActionName.SWITCH_APP,
        }:
            return BrokerDecisionType.ALLOW if risk in {RiskLevel.L0, RiskLevel.L1} else BrokerDecisionType.CONFIRM
        return BrokerDecisionType.CONFIRM
