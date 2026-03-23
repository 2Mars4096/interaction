"""Multimodal fusion intent helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from interaction.contracts import ActionName, ActionProposal, GazeObservation, GroundedTarget, NormalizedIntent, RiskLevel


_DEICTIC_PATTERNS = (
    (re.compile(r"^(right(?: |-)?click|secondary click)\s+(?:this|that|here|there)$", re.IGNORECASE), ActionName.RIGHT_CLICK_TARGET, RiskLevel.L2),
    (re.compile(r"^(click)\s+(?:this|that|here|there)$", re.IGNORECASE), ActionName.CLICK_TARGET, RiskLevel.L2),
    (re.compile(r"^(open)\s+(?:this|that|here|there)$", re.IGNORECASE), ActionName.DOUBLE_CLICK_TARGET, RiskLevel.L2),
    (re.compile(r"^(focus|move(?: cursor)?)\s+(?:this|that|here|there)$", re.IGNORECASE), ActionName.FOCUS_TARGET, RiskLevel.L1),
    (re.compile(r"^(show)\s+(?:this|that|here|there)$", re.IGNORECASE), ActionName.HIGHLIGHT_TARGET, RiskLevel.L0),
)


@dataclass(frozen=True)
class FusionResolution:
    intent: NormalizedIntent | None = None
    proposal: ActionProposal | None = None
    clarification: str | None = None


class FusionIntentResolver:
    """Resolve deictic voice commands against grounded gaze targets."""

    def __init__(self, *, min_confidence_to_ground: float = 0.78) -> None:
        self.min_confidence_to_ground = min_confidence_to_ground

    def resolve(
        self,
        utterance: str,
        target: GroundedTarget | None,
        *,
        observation: GazeObservation | None = None,
        fused_confidence: float,
        stale_reason: str | None = None,
        candidate_targets: list[GroundedTarget] | None = None,
    ) -> FusionResolution:
        text = " ".join(utterance.strip().split())
        for pattern, action, risk in _DEICTIC_PATTERNS:
            if not pattern.match(text):
                continue
            if target is None:
                clarification = stale_reason or "I need a recent gaze target before I can resolve that command."
                return FusionResolution(clarification=clarification)
            if fused_confidence < self.min_confidence_to_ground:
                candidates = candidate_targets or [target]
                labels = ", ".join(candidate.label for candidate in candidates[:3])
                clarification = (
                    f'I am not confident enough to act on "{text}" yet. '
                    f'Likely targets: {labels}.'
                )
                return FusionResolution(clarification=clarification)
            requires_confirmation = risk in {RiskLevel.L2, RiskLevel.L3}
            arguments: dict[str, object] = {
                "target_ref": target.target_id,
                "target_label": target.label,
            }
            if target.bounds is not None:
                arguments["target_bounds"] = target.bounds.model_dump(mode="json")
            if observation is not None and observation.x_norm is not None and observation.y_norm is not None:
                arguments["normalized_point"] = {
                    "x": observation.x_norm,
                    "y": observation.y_norm,
                }
            proposal = ActionProposal(
                action=action,
                arguments=arguments,
                confidence=fused_confidence,
                risk=risk,
                requires_confirmation=requires_confirmation,
                rationale=f'Fusion resolved "{text}" against the recent gaze target "{target.label}".',
            )
            intent = NormalizedIntent(
                intent=action.value,
                target_ref=target.target_id,
                modifiers={},
                confidence=fused_confidence,
                needs_clarification=False,
                utterance=text,
            )
            return FusionResolution(intent=intent, proposal=proposal)
        return FusionResolution()
