"""Scripted gaze tracking loop for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
import math

from interaction.contracts import ActionName, ActionProposal, BrokerDecisionType, EnvironmentSnapshot, GazeObservation, GroundedTarget, RiskLevel
from interaction.control import CommandBroker
from interaction.feedback import GazeFeedbackEvent, GazeLoopPhase
from interaction.platform import MacOSPlatformAdapter, PlatformAdapter
from interaction.vision import CalibrationProfile, CalibrationSample, DwellTrigger, GazeSample, GazeSmoother, GazeTargetInferencer, NormalizedPoint, NormalizedScreenTarget


@dataclass(frozen=True)
class DragOrigin:
    observation: GazeObservation
    target: GroundedTarget
    environment: EnvironmentSnapshot


class GazeTrackingLoop:
    """Drive calibration, smoothing, target inference, and dwell triggers."""

    def __init__(
        self,
        *,
        broker: CommandBroker | None = None,
        adapter: PlatformAdapter | None = None,
        inferencer: GazeTargetInferencer | None = None,
        smoother: GazeSmoother | None = None,
        dwell_trigger: DwellTrigger | None = None,
        auto_confirm_actions: set[ActionName] | None = None,
        drag_timeout_ms: int = 3000,
    ) -> None:
        self.broker = broker or CommandBroker()
        self.adapter = adapter or MacOSPlatformAdapter(dry_run=True)
        self.inferencer = inferencer or GazeTargetInferencer()
        self.smoother = smoother or GazeSmoother()
        self.dwell_trigger = dwell_trigger or DwellTrigger()
        self.auto_confirm_actions = auto_confirm_actions or set()
        self.drag_timeout_ms = max(0, drag_timeout_ms)
        self.calibration_profile = CalibrationProfile()
        self.pending_drag_origin: DragOrigin | None = None
        self.pending_drag_elapsed_ms = 0
        self._awaiting_reacquisition = False
        self._has_previous_gaze = False
        self._previous_target_id: str | None = None

    def calibrate(self, samples: list[CalibrationSample]) -> list[GazeFeedbackEvent]:
        self.calibration_profile = CalibrationProfile.fit(samples)
        return [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.CALIBRATING,
                message="Calibration profile fitted for scripted gaze tracking.",
            )
        ]

    def run_trace(
        self,
        samples: list[GazeSample],
        targets: list[NormalizedScreenTarget],
        environment: EnvironmentSnapshot,
    ) -> list[GazeFeedbackEvent]:
        events: list[GazeFeedbackEvent] = []
        for sample in samples:
            events.extend(self.process_sample(sample, targets, environment))

        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
        return events

    def tracking_lost(
        self,
        *,
        elapsed_ms: int = 0,
        reason: str = "Gaze tracking was lost. Look steadily at a target to resume.",
    ) -> list[GazeFeedbackEvent]:
        self.smoother.reset()
        self.dwell_trigger.reset(elapsed_ms=elapsed_ms, preserve_trigger=True)
        self._awaiting_reacquisition = True
        self._has_previous_gaze = False
        self._previous_target_id = None
        cancellation = self._cancel_drag(reason)
        return [cancellation or GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=reason)]

    def _cancel_drag(self, reason: str) -> GazeFeedbackEvent | None:
        origin = self.pending_drag_origin
        self.pending_drag_origin = None
        self.pending_drag_elapsed_ms = 0
        if origin is None:
            return None
        return GazeFeedbackEvent(
            phase=GazeLoopPhase.RECOVERING,
            message=f'Drag origin at "{origin.target.label}" cancelled. {reason}',
        )

    def process_sample(
        self,
        sample: GazeSample,
        targets: list[NormalizedScreenTarget],
        environment: EnvironmentSnapshot,
        *,
        measured_timing: bool = False,
    ) -> list[GazeFeedbackEvent]:
        events: list[GazeFeedbackEvent] = []
        elapsed_ms = sample.delta_ms if type(sample.delta_ms) is int and sample.delta_ms >= 0 else 0
        if self.pending_drag_origin is not None and self.pending_drag_origin.environment != environment:
            events.extend(self.tracking_lost(
                elapsed_ms=elapsed_ms,
                reason="The application or window changed. Start the drag again in the intended location.",
            ))
            elapsed_ms = 0
        if (
            not isinstance(sample.point, NormalizedPoint)
            or not all(_finite_unit_value(value) for value in (sample.point.x, sample.point.y, sample.confidence))
            or sample.confidence < max(self.inferencer.min_confidence, self.dwell_trigger.min_confidence)
            or type(sample.delta_ms) is not int
            or sample.delta_ms < 0
        ):
            return events + self.tracking_lost(elapsed_ms=elapsed_ms, reason="Gaze tracking is weak or invalid. Look steadily at a target to resume.")
        try:
            calibrated_point = self.calibration_profile.project(sample.point)
        except ValueError:
            return events + self.tracking_lost(elapsed_ms=elapsed_ms, reason="The calibrated gaze point is invalid. Reacquire tracking before continuing.")
        if not all(-1e-9 <= value <= 1 + 1e-9 for value in (calibrated_point.x, calibrated_point.y)):
            return events + self.tracking_lost(elapsed_ms=elapsed_ms, reason="The calibrated gaze point is outside the screen. Look back at a target.")
        calibrated_point = NormalizedPoint(
            max(0.0, min(1.0, calibrated_point.x)), max(0.0, min(1.0, calibrated_point.y)),
        )
        if self.pending_drag_origin is not None:
            self.pending_drag_elapsed_ms += elapsed_ms
            if self.pending_drag_elapsed_ms >= self.drag_timeout_ms > 0:
                events.extend(self.tracking_lost(elapsed_ms=elapsed_ms, reason="The drag timed out. Start the drag again."))
                elapsed_ms = 0
        # A reacquired sample locates the eyes; elapsed missing time is never
        # evidence that the eyes were dwelling at this newly observed point.
        evidence_delta_ms = 0 if self._awaiting_reacquisition else elapsed_ms
        self._awaiting_reacquisition = False
        evidence = GazeObservation(
            confidence=sample.confidence, x_norm=calibrated_point.x, y_norm=calibrated_point.y,
            fixation_ms=evidence_delta_ms,
        )
        target = self.inferencer.infer(evidence, targets)
        if measured_timing:
            target_id = target.target_id if target is not None else None
            if not self._has_previous_gaze or self._previous_target_id != target_id:
                evidence_delta_ms = 0
                evidence = evidence.model_copy(update={"fixation_ms": 0})
            self._previous_target_id = target_id
            self._has_previous_gaze = True
        else:
            self._has_previous_gaze = False
            self._previous_target_id = None
        if target is None:
            self.smoother.reset()
            self.dwell_trigger.update(evidence, None, elapsed_ms=elapsed_ms)
            reason = "No large target is currently grounded. Partial dwell was cleared."
            cancellation = self._cancel_drag(reason)
            events.append(cancellation or GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING, message=reason, observation=evidence,
            ))
            return events
        observation = self.smoother.smooth(
            GazeSample(
                point=calibrated_point,
                confidence=sample.confidence,
                delta_ms=evidence_delta_ms,
            )
        )

        proposal = self.dwell_trigger.update(evidence, target, elapsed_ms=elapsed_ms)
        if proposal is not None:
            progress_message = "Dwell complete."
        elif self.dwell_trigger.triggered_target_id == target.target_id:
            progress_message = "Look away to rearm."
        elif self.dwell_trigger.cooldown_remaining_ms > 0:
            progress_message = "Waiting briefly before another action."
        else:
            progress = min(100, int(100 * self.dwell_trigger.accumulated_ms / max(1, self.dwell_trigger.dwell_ms)))
            progress_message = f"Dwell {progress}%."
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message=f'Grounded target "{target.label}". {progress_message}',
                observation=observation,
                target=target,
            )
        )
        if proposal is None:
            return events
        if proposal.action == ActionName.DRAG_TARGET:
            return events + self._handle_drag_stage(proposal, evidence, target, environment)

        decision = self.broker.decide(proposal)
        auto_confirmed = False
        if decision.decision == BrokerDecisionType.CONFIRM and proposal.action in self.auto_confirm_actions:
            decision = self.broker.confirm(decision)
            auto_confirmed = True
        if decision.decision != BrokerDecisionType.ALLOW:
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message=decision.reason,
                    observation=observation,
                    target=target,
                    proposal=proposal,
                )
            )
            return events
        request = self.broker.build_execution_request(decision, environment)
        result = self.adapter.execute(request)
        action_label = proposal.action.value.replace("_target", "").replace("_", " ")
        message = f"Stable dwell triggered a {action_label} action."
        if auto_confirmed:
            message = f"Stable dwell triggered an explicit gaze-mode {action_label} action."
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRIGGERED,
                message=message,
                observation=observation,
                target=target,
                proposal=proposal,
                result=result,
            )
        )
        return events

    def _handle_drag_stage(
        self,
        proposal: ActionProposal,
        observation: GazeObservation,
        target: GroundedTarget,
        environment: EnvironmentSnapshot,
    ) -> list[GazeFeedbackEvent]:
        if self.pending_drag_origin is None:
            self.pending_drag_origin = DragOrigin(
                observation=observation, target=target, environment=environment.model_copy(deep=True),
            )
            self.pending_drag_elapsed_ms = 0
            return [
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.TRIGGERED,
                    message=f'Drag origin armed at "{target.label}". Look at the drop target and dwell again.',
                    observation=observation,
                    target=target,
                    proposal=proposal,
                )
            ]

        origin = self.pending_drag_origin
        if target.target_id == origin.target.target_id:
            return [
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="Drag destination still matches the source. Look at a different drop target and dwell again.",
                    observation=observation,
                    target=target,
                    proposal=proposal,
                )
            ]

        self.pending_drag_origin = None
        self.pending_drag_elapsed_ms = 0
        drag_proposal = self._build_drag_proposal(origin, observation, target)
        decision = self.broker.decide(drag_proposal)
        auto_confirmed = False
        if decision.decision == BrokerDecisionType.CONFIRM and drag_proposal.action in self.auto_confirm_actions:
            decision = self.broker.confirm(decision)
            auto_confirmed = True
        if decision.decision != BrokerDecisionType.ALLOW:
            return [
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message=decision.reason,
                    observation=observation,
                    target=target,
                    proposal=drag_proposal,
                )
            ]
        request = self.broker.build_execution_request(decision, environment)
        result = self.adapter.execute(request)
        self.dwell_trigger.start_cooldown()
        message = f'Drag completed from "{origin.target.label}" to "{target.label}".'
        if auto_confirmed:
            message = f'Explicit gaze-mode drag completed from "{origin.target.label}" to "{target.label}".'
        return [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRIGGERED,
                message=message,
                observation=observation,
                target=target,
                proposal=drag_proposal,
                result=result,
            )
        ]

    def _build_drag_proposal(
        self,
        origin: DragOrigin,
        observation: GazeObservation,
        target: GroundedTarget,
    ) -> ActionProposal:
        arguments: dict[str, object] = {
            "start_target_ref": origin.target.target_id,
            "start_target_label": origin.target.label,
            "target_ref": target.target_id,
            "target_label": target.label,
        }
        if origin.target.bounds is not None:
            arguments["start_target_bounds"] = origin.target.bounds.model_dump(mode="json")
        if target.bounds is not None:
            arguments["end_target_bounds"] = target.bounds.model_dump(mode="json")
        if origin.observation.x_norm is not None and origin.observation.y_norm is not None:
            arguments["start_normalized_point"] = {
                "x": origin.observation.x_norm,
                "y": origin.observation.y_norm,
            }
        if observation.x_norm is not None and observation.y_norm is not None:
            arguments["end_normalized_point"] = {
                "x": observation.x_norm,
                "y": observation.y_norm,
            }
        confidence = min(origin.target.confidence, target.confidence)
        return ActionProposal(
            action=ActionName.DRAG_TARGET,
            arguments=arguments,
            confidence=confidence,
            risk=RiskLevel.L2,
            requires_confirmation=True,
            rationale=f'Explicit gaze drag resolved from "{origin.target.label}" to "{target.label}".',
        )


def _finite_unit_value(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
