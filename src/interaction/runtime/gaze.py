"""Scripted gaze tracking loop for Phase 3."""

from __future__ import annotations

from interaction.contracts import ActionName, BrokerDecisionType, EnvironmentSnapshot
from interaction.control import CommandBroker
from interaction.feedback import GazeFeedbackEvent, GazeLoopPhase
from interaction.platform import MacOSPlatformAdapter, PlatformAdapter
from interaction.vision import CalibrationProfile, CalibrationSample, DwellTrigger, GazeSample, GazeSmoother, GazeTargetInferencer, NormalizedScreenTarget


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
    ) -> None:
        self.broker = broker or CommandBroker()
        self.adapter = adapter or MacOSPlatformAdapter(dry_run=True)
        self.inferencer = inferencer or GazeTargetInferencer()
        self.smoother = smoother or GazeSmoother()
        self.dwell_trigger = dwell_trigger or DwellTrigger()
        self.auto_confirm_actions = auto_confirm_actions or set()
        self.calibration_profile = CalibrationProfile()

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
