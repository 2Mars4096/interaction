"""Scripted gaze tracking loop for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

from interaction.contracts import ActionName, ActionProposal, BrokerDecisionType, EnvironmentSnapshot, GazeObservation, GroundedTarget, RiskLevel
from interaction.control import CommandBroker
from interaction.feedback import GazeFeedbackEvent, GazeLoopPhase
from interaction.platform import MacOSPlatformAdapter, PlatformAdapter
from interaction.vision import CalibrationProfile, CalibrationSample, DwellTrigger, GazeSample, GazeSmoother, GazeTargetInferencer, NormalizedScreenTarget


@dataclass(frozen=True)
class DragOrigin:
    observation: GazeObservation
    target: GroundedTarget


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

    def process_sample(
        self,
        sample: GazeSample,
        targets: list[NormalizedScreenTarget],
        environment: EnvironmentSnapshot,
    ) -> list[GazeFeedbackEvent]:
        events: list[GazeFeedbackEvent] = []
        if self.pending_drag_origin is not None:
            self.pending_drag_elapsed_ms += max(0, sample.delta_ms)
            if self.pending_drag_elapsed_ms >= self.drag_timeout_ms > 0:
                timed_out_origin = self.pending_drag_origin
                self.pending_drag_origin = None
                self.pending_drag_elapsed_ms = 0
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.RECOVERING,
                        message=f'Drag origin at "{timed_out_origin.target.label}" timed out. Start the drag again.',
                        observation=timed_out_origin.observation,
                        target=timed_out_origin.target,
                    )
                )
        calibrated_point = self.calibration_profile.apply(sample.point)
        observation = self.smoother.smooth(
            GazeSample(
                point=calibrated_point,
                confidence=sample.confidence,
                delta_ms=sample.delta_ms,
            )
        )
        target = self.inferencer.infer(observation, targets)
        if target is None:
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="No large target is currently grounded.",
                    observation=observation,
                )
            )
            return events

        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message=f'Grounded target "{target.label}".',
                observation=observation,
                target=target,
            )
        )
        proposal = self.dwell_trigger.update(observation, target)
        if proposal is None:
            return events
        if proposal.action == ActionName.DRAG_TARGET:
            return events + self._handle_drag_stage(proposal, observation, target, environment)

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
            self.pending_drag_origin = DragOrigin(observation=observation, target=target)
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
            self.pending_drag_elapsed_ms = 0
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
