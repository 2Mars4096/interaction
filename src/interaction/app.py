"""Small CLI entry point for repeatable local testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict
from typing import Any
from types import SimpleNamespace

from interaction.audio import MacOSLiveSpeechProvider, SpeechCaptureError
from interaction.contracts import ActionName, EnvironmentSnapshot, GazeObservation
from interaction.feedback import (
    FusionFeedbackEvent,
    FusionLoopPhase,
    GazeFeedbackEvent,
    GazeLoopPhase,
    VoiceFeedbackEvent,
    VoiceLoopPhase,
)
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.platform import MacOSPlatformAdapter
from interaction.platform.environment import capture_environment
from interaction.runtime.live_gaze import LiveGazeSampler
from interaction.runtime import (
    GazeTrackingLoop,
    MultimodalFusionLoop,
    VoiceCommandLoop,
    WebcamCalibrationError,
    collect_webcam_calibration,
    default_webcam_targets,
    run_live_cursor_follow,
    run_live_webcam_trace,
)
from interaction.session import SessionLogger, SessionReplay, serialize_feedback_event
from interaction.session.live import LiveEventBuffer
from interaction.intent.voice import normalize_command
from interaction.ui import ConsoleOverlayRenderer, LiveGazeDotOverlay, OverlayController
from interaction.ui.calibration import CalibrationTargetOverlay, CalibrationUIError
from interaction.vision import CalibrationSample, DwellTrigger, GazeSample, NormalizedPoint, NormalizedScreenTarget, OpenCVWebcamGazeProvider, WebcamProviderError


def cli() -> int:
    return main()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    runtime_paths = RuntimePaths(Path(args.runtime_dir))
    store = JsonStateStore(runtime_paths)
    if args.command == "voice-smoke":
        payload = _run_voice_smoke(args, store)
    elif args.command == "voice-live":
        payload = _run_voice_live(args, store)
    elif args.command == "gaze-calibrate":
        payload = _run_gaze_calibrate(args, store)
    elif args.command == "gaze-smoke":
        payload = _run_gaze_smoke(args, store)
    elif args.command == "gaze-live":
        payload = _run_gaze_live(args, store)
    elif args.command == "fusion-smoke":
        payload = _run_fusion_smoke(args, store)
    elif args.command == "fusion-live":
        payload = _run_fusion_live(args, store)
    else:
        payload = _run_replay(args)
    print(json.dumps(payload, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interaction", description="Repeatable local smoke and replay CLI.")
    subparsers = parser.add_subparsers(dest="command")

    voice = subparsers.add_parser("voice-smoke")
    voice.add_argument("--utterance", default="open Safari")
    voice.add_argument("--follow-up", default=None)
    voice.add_argument("--runtime-dir", default=".interaction")
    voice.add_argument("--session-name", default="voice-smoke")
    voice.add_argument("--execute", action="store_true")

    voice_live = subparsers.add_parser("voice-live")
    voice_live.add_argument("--duration", type=float, default=4.0)
    voice_live.add_argument("--locale", default="en-US")
    voice_live.add_argument("--runtime-dir", default=".interaction")
    voice_live.add_argument("--session-name", default="voice-live")
    voice_live.add_argument("--execute", action="store_true")
    voice_live.add_argument("--confirm-duration", type=float, default=2.5)
    voice_live.add_argument("--confirmation-attempts", type=int, default=2)
    voice_live.add_argument("--turns", type=int, default=1, help="Number of commands in the same session.")
    voice_live.add_argument("--continuous", action="store_true", help="Listen for successive commands until 'stop listening' or Ctrl-C.")

    gaze_calibrate = subparsers.add_parser("gaze-calibrate")
    gaze_calibrate.add_argument("--camera-index", type=int, default=None)
    gaze_calibrate.add_argument("--frames-per-step", type=int, default=6)
    gaze_calibrate.add_argument("--delta-ms", type=int, default=100)
    gaze_calibrate.add_argument("--settle-ms", type=int, default=800)
    gaze_calibrate.add_argument("--runtime-dir", default=".interaction")
    gaze_calibrate.add_argument("--session-name", default="gaze-calibrate")

    gaze = subparsers.add_parser("gaze-smoke")
    gaze.add_argument("--action", choices=["highlight", "move", "click", "right-click", "double-click", "drag", "cursor"], default="highlight")
    gaze.add_argument("--action-cooldown-ms", type=int, default=900)
    gaze.add_argument("--drag-timeout-ms", type=int, default=3000)
    gaze.add_argument("--cursor-deadzone", type=float, default=0.015)
    gaze.add_argument("--cursor-smoothing", type=float, default=0.28)
    gaze.add_argument("--cursor-edge-padding", type=float, default=0.03)
    gaze.add_argument("--cursor-max-step", type=float, default=0.16)
    gaze.add_argument("--runtime-dir", default=".interaction")
    gaze.add_argument("--session-name", default="gaze-smoke")
    gaze.add_argument("--execute", action="store_true")

    gaze_live = subparsers.add_parser("gaze-live")
    gaze_live.add_argument("--camera-index", type=int, default=None)
    gaze_live.add_argument("--frames", type=int, default=18)
    gaze_live.add_argument("--delta-ms", type=int, default=100)
    gaze_live.add_argument("--action", choices=["highlight", "move", "click", "right-click", "double-click", "drag", "cursor"], default="highlight")
    gaze_live.add_argument("--action-cooldown-ms", type=int, default=900)
    gaze_live.add_argument("--drag-timeout-ms", type=int, default=3000)
    gaze_live.add_argument("--cursor-deadzone", type=float, default=0.015)
    gaze_live.add_argument("--cursor-smoothing", type=float, default=0.28)
    gaze_live.add_argument("--cursor-edge-padding", type=float, default=0.03)
    gaze_live.add_argument("--cursor-max-step", type=float, default=0.16)
    gaze_live.add_argument("--show-dot", action="store_true")
    gaze_live.add_argument("--dot-size", type=int, default=28)
    gaze_live.add_argument("--runtime-dir", default=".interaction")
    gaze_live.add_argument("--session-name", default="gaze-live")
    gaze_live.add_argument("--execute", action="store_true")

    fusion = subparsers.add_parser("fusion-smoke")
    fusion.add_argument("--utterance", default="open this")
    fusion.add_argument("--follow-up", default="yes")
    fusion.add_argument("--runtime-dir", default=".interaction")
    fusion.add_argument("--session-name", default="fusion-smoke")
    fusion.add_argument("--execute", action="store_true")

    fusion_live = subparsers.add_parser("fusion-live")
    fusion_live.add_argument("--camera-index", type=int, default=None)
    fusion_live.add_argument("--gaze-frames", type=int, default=12)
    fusion_live.add_argument("--gaze-delta-ms", type=int, default=100, help="Minimum delay between camera samples; freshness uses actual elapsed time.")
    fusion_live.add_argument("--duration", type=float, default=4.0)
    fusion_live.add_argument("--confirm-duration", type=float, default=2.5)
    fusion_live.add_argument("--locale", default="en-US")
    fusion_live.add_argument("--runtime-dir", default=".interaction")
    fusion_live.add_argument("--session-name", default="fusion-live")
    fusion_live.add_argument("--execute", action="store_true")

    replay = subparsers.add_parser("replay")
    replay.add_argument("--session-log", required=True)
    replay.add_argument("--runtime-dir", default=".interaction")
    return parser


def _run_voice_smoke(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=effective_dry_run))
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events = loop.run_text_turn(args.utterance, EnvironmentSnapshot(active_app="Finder"))
    if args.follow_up:
        events.extend(loop.run_text_turn(args.follow_up, EnvironmentSnapshot(active_app="Finder")))
    return _build_payload(
        source="voice",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={"settings": _settings_payload(settings, effective_dry_run)},
    )


def _run_voice_live(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    provider = MacOSLiveSpeechProvider(store.paths)
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=effective_dry_run))
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events = LiveEventBuffer("voice", logger, overlay)
    captures: list[dict[str, Any]] = []
    confirmation_captures: list[dict[str, Any]] = []
    stopped = False
    failed = False
    commands = 0

    def capture_and_handle(*, confirmation: bool = False) -> bool:
        nonlocal stopped, failed
        duration = args.confirm_duration if confirmation else args.duration
        events.append(VoiceFeedbackEvent(
            phase=VoiceLoopPhase.LISTENING,
            message="Listening for confirmation or cancel." if confirmation else "Listening. Say a command, or 'stop listening' to finish.",
        ))
        try:
            before = capture_environment()
            if not effective_dry_run and before.active_app is None:
                raise SpeechCaptureError("context_unavailable", "Cannot identify the active application. Enable Accessibility access and retry.")
            capture = provider.capture_turn(duration_s=duration, locale=args.locale)
            after = capture_environment()
            if before != after:
                loop.pending_decision = None
                raise SpeechCaptureError("context_changed", "The active application or window changed while listening. Please repeat the command.")
        except SpeechCaptureError as error:
            payload = {"status": "error", "error": error.code, "message": error.message, "payload": error.payload}
            (confirmation_captures if confirmation else captures).append(payload)
            if error.code == "empty_transcript" and (args.continuous or confirmation):
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.LISTENING, message="No speech heard."))
                return True
            events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, message=error.message))
            loop.pending_decision = None
            if error.code == "context_changed" and args.continuous:
                return True
            failed = True
            return False
        payload = _speech_capture_payload(capture)
        (confirmation_captures if confirmation else captures).append(payload)
        events.extend(loop.run_text_turn(
            capture.transcript,
            after,
            final_confidence=capture.confidence,
            begin_turn=False,
        ))
        stopped = normalize_command(capture.transcript).lower() in {"stop", "stop listening", "pause", "quit", "exit"}
        return True

    try:
        while not stopped and (args.continuous or commands < max(1, args.turns)):
            if not capture_and_handle():
                break
            if captures[-1]["status"] == "success":
                commands += 1
            attempts = 0
            while not stopped and loop.pending_decision is not None and attempts < max(1, args.confirmation_attempts):
                attempts += 1
                if not capture_and_handle(confirmation=True):
                    break
            if loop.pending_decision is not None:
                loop.pending_decision = None
                events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, message="The pending action was cancelled because no clear confirmation arrived."))
            if failed:
                break
    except KeyboardInterrupt:
        stopped = True
        loop.pending_decision = None
        events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.RECOVERING, message="Listening stopped."))
    events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice session ended."))
    return _build_payload(
        source="voice", events=events, logger=logger, overlay=overlay, renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run),
            "live_capture": captures[-1] if captures else None,
            "captures": captures,
            "confirmation_captures": confirmation_captures,
            "session": {"commands": commands, "stopped": stopped, "failed": failed},
            "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
        },
    )


def _speech_capture_payload(capture) -> dict[str, Any]:
    return {"status": "success", **capture.__dict__}


def _run_gaze_calibrate(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events = LiveEventBuffer("gaze", logger, overlay)
    provider = None
    target_screen = None
    profile = None
    step_results = []
    calibration = {}
    step_number = 0

    def before_step(step):
        nonlocal step_number
        step_number += 1
        target_screen.show_target(step.expected.x, step.expected.y,
                                  f"{step_number}/9: {step.label}", phase=step.phase)
        events.append(GazeFeedbackEvent(
            phase=GazeLoopPhase.CALIBRATING,
            message=f"{step.phase.title()} {step_number}/9: look at the {step.label} dot. Press Escape to cancel.",
        ))
        target_screen.wait(max(0, args.settle_ms))

    try:
        target_screen = CalibrationTargetOverlay()
        target_screen.open()
        provider = OpenCVWebcamGazeProvider(camera_index=camera_index)
        provider.open()
        profile, step_results = collect_webcam_calibration(
            provider, frames_per_step=args.frames_per_step, delta_ms=args.delta_ms,
            before_step=before_step, on_frame=target_screen.pump,
        )
        target_screen.pump()
        calibration = {"status": "success", "display_size": target_screen.screen_size}
    except (WebcamProviderError, WebcamCalibrationError, CalibrationUIError) as error:
        profile = None
        calibration = {"status": "error", "error": error.code, "message": error.message, "payload": error.payload}
        step_results = list(error.payload.get("steps", []))
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message))
    except KeyboardInterrupt:
        profile = None
        calibration = {"status": "error", "error": "calibration_cancelled", "message": "Calibration cancelled."}
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message="Calibration cancelled."))
    except Exception as error:
        profile = None
        calibration = {"status": "error", "error": "calibration_failed", "message": str(error)}
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=f"Calibration failed: {error}"))
    finally:
        for resource in (provider, target_screen):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    profile = None
                    calibration = {"status": "error", "error": "calibration_cleanup_failed", "message": str(error)}
                    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=f"Calibration cleanup failed: {error}"))
    if profile is not None:
        try:
            store.save_calibration_profile(profile, name="webcam-live")
        except (OSError, ValueError) as error:
            profile = None
            calibration = {"status": "error", "error": "calibration_save_failed", "message": str(error)}
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=f"Could not save calibration: {error}"))
        else:
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.CALIBRATING, message="Validation passed. Saved live webcam calibration profile."))
    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Calibration session ended."))
    return _build_payload(
        source="gaze", events=events, logger=logger, overlay=overlay, renderer=renderer,
        extra={
            "settings": settings.__dict__, "camera_index": camera_index,
            "calibration": calibration, "calibration_steps": step_results,
            "calibration_profile": _profile_to_payload(profile),
        },
    )


def _run_gaze_smoke(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    gaze_mode = args.action
    gaze_action = _gaze_action_name(gaze_mode)
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=effective_dry_run),
        dwell_trigger=DwellTrigger(dwell_ms=settings.dwell_ms, action=gaze_action, cooldown_ms=max(0, args.action_cooldown_ms)),
        auto_confirm_actions=_gaze_auto_confirm_actions(gaze_action),
        drag_timeout_ms=max(0, args.drag_timeout_ms),
    )
    existing = store.load_calibration_profile("default")
    calibration_events: list[GazeFeedbackEvent] = []
    if existing is None:
        calibration_samples = [
            CalibrationSample(raw=NormalizedPoint(0.54, 0.20), expected=NormalizedPoint(0.55, 0.20)),
            CalibrationSample(raw=NormalizedPoint(0.18, 0.50), expected=NormalizedPoint(0.18, 0.50)),
            CalibrationSample(raw=NormalizedPoint(0.70, 0.20), expected=NormalizedPoint(0.70, 0.20)),
        ]
        calibration_events = loop.calibrate(calibration_samples)
        store.save_calibration_profile(loop.calibration_profile, name="default")
    else:
        loop.calibration_profile = existing
    targets = [
        NormalizedScreenTarget(target_id="compose", label="Compose button", role="button", x=0.52, y=0.16, width=0.22, height=0.12),
        NormalizedScreenTarget(target_id="sidebar", label="Mail sidebar", role="panel", x=0.02, y=0.1, width=0.20, height=0.75),
    ]
    if gaze_action == ActionName.DRAG_TARGET:
        trace = [
            GazeSample(point=NormalizedPoint(0.55, 0.20), confidence=0.82, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.56, 0.20), confidence=0.84, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.56, 0.20), confidence=0.85, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.57, 0.20), confidence=0.86, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.09, 0.45), confidence=0.86, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.08, 0.46), confidence=0.87, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.08, 0.47), confidence=0.88, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.07, 0.48), confidence=0.89, delta_ms=250),
            GazeSample(point=NormalizedPoint(0.07, 0.49), confidence=0.89, delta_ms=250),
        ]
    else:
        trace = [
            GazeSample(point=NormalizedPoint(0.55, 0.20), confidence=0.82, delta_ms=200),
            GazeSample(point=NormalizedPoint(0.56, 0.20), confidence=0.84, delta_ms=200),
            GazeSample(point=NormalizedPoint(0.56, 0.20), confidence=0.85, delta_ms=200),
            GazeSample(point=NormalizedPoint(0.57, 0.20), confidence=0.86, delta_ms=200),
        ]
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events: list[object] = []
    events.extend(calibration_events)
    events.extend(loop.run_trace(trace, targets, EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")))
    return _build_payload(
        source="gaze",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run),
            "gaze_action": gaze_action.value,
            "gaze_mode": gaze_mode,
            "cursor_settings": _cursor_settings_payload(args),
            "gaze_control_settings": _gaze_control_settings_payload(args),
            "calibration_profile": {
                "x_scale": loop.calibration_profile.x_scale,
                "y_scale": loop.calibration_profile.y_scale,
                "x_offset": loop.calibration_profile.x_offset,
                "y_offset": loop.calibration_profile.y_offset,
            },
        },
    )


def _run_gaze_live(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    gaze_mode = args.action
    gaze_action = _gaze_action_name(gaze_mode)
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    profile = store.load_calibration_profile("webcam-live", require_validated=True, camera_index=camera_index)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    visualization = _gaze_visualization_payload(args)
    if profile is None:
        events = [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING,
                message="No current validated calibration profile is available for this camera. Run `interaction gaze-calibrate` first.",
            ),
            GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."),
        ]
        return _build_payload(
            source="gaze",
            events=events,
            logger=logger,
            overlay=overlay,
            renderer=renderer,
            extra={
                "settings": _settings_payload(settings, effective_dry_run),
                "gaze_action": gaze_action.value,
                "gaze_mode": gaze_mode,
                "cursor_settings": _cursor_settings_payload(args),
                "gaze_control_settings": _gaze_control_settings_payload(args),
                "camera_index": camera_index,
                "live_gaze": {
                    "status": "error",
                    "error": "missing_calibration_profile",
                    "message": "No current validated calibration profile is available for this camera. Run `interaction gaze-calibrate` first.",
                },
                "gaze_visualization": visualization,
                "calibration_profile": None,
            },
        )

    provider = OpenCVWebcamGazeProvider(camera_index=camera_index)
    point_visualizer = None
    if args.show_dot:
        try:
            point_visualizer = LiveGazeDotOverlay(size=max(12, args.dot_size))
            point_visualizer.open()
            visualization["status"] = "enabled"
        except Exception as error:
            visualization["status"] = "error"
            visualization["error"] = str(error)
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=effective_dry_run),
        dwell_trigger=DwellTrigger(dwell_ms=settings.dwell_ms, action=gaze_action, cooldown_ms=max(0, args.action_cooldown_ms)),
        auto_confirm_actions=_gaze_auto_confirm_actions(gaze_action),
        drag_timeout_ms=max(0, args.drag_timeout_ms),
    )
    loop.calibration_profile = profile
    try:
        provider.open()
        if gaze_mode == "cursor":
            events, summary = run_live_cursor_follow(
                provider,
                profile,
                adapter=loop.adapter,
                frames=args.frames,
                delta_ms=args.delta_ms,
                environment=EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera"),
                targets=default_webcam_targets(),
                min_move_distance=max(0.0, args.cursor_deadzone),
                smoothing=max(0.0, min(1.0, args.cursor_smoothing)),
                edge_padding=max(0.0, min(0.45, args.cursor_edge_padding)),
                max_step=max(0.01, min(1.0, args.cursor_max_step)),
                point_visualizer=point_visualizer,
            )
        else:
            events, summary = run_live_webcam_trace(
                provider,
                loop,
                frames=args.frames,
                delta_ms=args.delta_ms,
                targets=default_webcam_targets(),
                environment=EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera"),
                point_visualizer=point_visualizer,
            )
        live_gaze = {"status": "success", **summary}
    except WebcamProviderError as error:
        events = [
            GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message),
            GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."),
        ]
        live_gaze = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
    finally:
        provider.close()
        if point_visualizer is not None:
            point_visualizer.close()
    return _build_payload(
        source="gaze",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run),
            "gaze_action": gaze_action.value,
            "gaze_mode": gaze_mode,
            "camera_index": camera_index,
            "live_gaze": live_gaze,
            "cursor_settings": _cursor_settings_payload(args),
            "gaze_control_settings": _gaze_control_settings_payload(args),
            "gaze_visualization": visualization,
            "calibration_profile": _profile_to_payload(profile),
        },
    )


def _run_fusion_smoke(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=effective_dry_run))
    loop.state.gaze_context_window_ms = settings.gaze_context_window_ms
    target = NormalizedScreenTarget(
        target_id="compose",
        label="Compose button",
        role="button",
        x=0.52,
        y=0.16,
        width=0.22,
        height=0.12,
    ).to_grounded_target(confidence=0.91)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events: list[object] = []
    events.append(loop.update_gaze_context(GazeObservation(confidence=0.88, x_norm=0.56, y_norm=0.20, fixation_ms=250), target))
    events.extend(loop.run_voice_turn(args.utterance, EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")))
    if args.follow_up:
        events.extend(loop.run_voice_turn(args.follow_up, EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")))
    return _build_payload(
        source="fusion",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run),
            "metrics": loop.metrics.to_dict(),
            "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
        },
    )


def _run_fusion_live(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    profile = store.load_calibration_profile("webcam-live", require_validated=True, camera_index=camera_index)
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=effective_dry_run))
    loop.state.gaze_context_window_ms = settings.gaze_context_window_ms
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events = LiveEventBuffer("fusion", logger, overlay)
    live_gaze_context: dict[str, Any] = {}
    live_capture = None
    confirmation_capture = None
    sampler = None
    capture_stage = "command"
    gaze_revision = None

    def update_current_gaze():
        nonlocal gaze_revision
        snapshot = sampler.latest()
        if snapshot is None:
            events.append(loop.clear_gaze_context())
            live_gaze_context.clear()
            live_gaze_context.update({
                "status": "unavailable", "error": "gaze_tracking_lost",
                "message": "No fresh stable gaze target is available.",
            })
            return
        if gaze_revision is not None and snapshot.revision != gaze_revision:
            events.append(loop.clear_gaze_context(observed_at_ms=snapshot.observed_at_ms))
        gaze_revision = snapshot.revision
        events.append(loop.update_gaze_context(snapshot.observation, snapshot.target, observed_at_ms=snapshot.observed_at_ms))
        live_gaze_context.update({
            "status": "success", "target": snapshot.target.model_dump(mode="json"),
            "observation": snapshot.observation.model_dump(mode="json"), "observed_at_ms": snapshot.observed_at_ms,
        })

    def capture_command(provider, *, confirmation=False):
        before = capture_environment()
        if not effective_dry_run and before.active_app is None:
            raise SpeechCaptureError("context_unavailable", "Cannot identify the active application. Enable Accessibility access and retry.")
        events.append(FusionFeedbackEvent(
            phase=FusionLoopPhase.CONFIRMING if confirmation else FusionLoopPhase.TRACKING,
            message="Listening for confirmation or cancel." if confirmation else "Listening. Gaze tracking remains active.",
        ))
        capture = provider.capture_turn(duration_s=args.confirm_duration if confirmation else args.duration, locale=args.locale)
        after = capture_environment()
        if before != after:
            loop.pending_decision = None
            raise SpeechCaptureError("context_changed", "The active application or window changed while listening. Please repeat the command.")
        update_current_gaze()
        events.extend(loop.run_voice_turn(capture.transcript, after, final_confidence=capture.confidence))
        return _speech_capture_payload(capture)

    try:
        if profile is None:
            raise WebcamProviderError("missing_calibration_profile", "No current validated calibration profile is available for this camera. Run `interaction gaze-calibrate` first.")
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.TRACKING, message="Starting gaze tracking. Look steadily at a target."))
        sampler = LiveGazeSampler(OpenCVWebcamGazeProvider(camera_index=camera_index), profile,
                                  max_age_ms=settings.gaze_context_window_ms,
                                  interval_s=max(5, args.gaze_delta_ms) / 1000)
        sampler.start()
        initial = sampler.wait_for_target(max_frames=max(3, args.gaze_frames))
        if initial is None:
            raise WebcamProviderError("missing_gaze_lock", "No stable gaze target was established. Look steadily at a large target and retry.")
        update_current_gaze()
        provider = MacOSLiveSpeechProvider(store.paths)
        live_capture = capture_command(provider)
        if loop.pending_decision is not None:
            capture_stage = "confirmation"
            confirmation_capture = capture_command(provider, confirmation=True)
            if loop.pending_decision is not None:
                loop.pending_decision = None
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="The action was cancelled because no clear confirmation arrived."))
    except WebcamProviderError as error:
        live_gaze_context = {"status": "error", "error": error.code, "message": error.message, "payload": error.payload}
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=error.message))
    except SpeechCaptureError as error:
        failure = {"status": "error", "error": error.code, "message": error.message, "payload": error.payload}
        if capture_stage == "confirmation":
            confirmation_capture = failure
        else:
            live_capture = failure
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=error.message))
    except KeyboardInterrupt:
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message="Fusion session stopped."))
    finally:
        loop.pending_decision = None
        if sampler is not None:
            try:
                sampler.stop()
            except WebcamProviderError as error:
                live_gaze_context.update({"status": "error", "error": error.code, "message": error.message})
                events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=error.message))
            live_gaze_context.update(sampler.summary())
    events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion session ended."))
    return _build_payload(
        source="fusion", events=events, logger=logger, overlay=overlay, renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run), "camera_index": camera_index,
            "live_gaze_context": live_gaze_context, "live_capture": live_capture,
            "confirmation_capture": confirmation_capture, "metrics": loop.metrics.to_dict(),
            "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
            "calibration_profile": _profile_to_payload(profile) if profile else None,
        },
    )


def _run_replay(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.session_log)
    records = SessionReplay.load(path)
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    snapshots: list[dict[str, Any]] = []
    for record in records:
        event = _replay_event(record)
        state = overlay.apply_event(record.source, event)
        snapshots.append(state.to_dict())
    final_overlay = overlay.state
    return {
        "session_log": str(path),
        "records": [record.to_dict() for record in records],
        "final_overlay": final_overlay.to_dict(),
        "final_overlay_rendered": renderer.render(final_overlay),
        "overlay_snapshots": snapshots,
    }


def _build_payload(
    *,
    source: str,
    events: list[object],
    logger: SessionLogger,
    overlay: OverlayController,
    renderer: ConsoleOverlayRenderer,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(events, LiveEventBuffer):
        snapshots = events.snapshots
        serialized_events = events.serialized_events
    else:
        snapshots = []
        serialized_events = []
        for event in events:
            logger.record_event(source, event)
            state = overlay.apply_event(source, event)
            snapshots.append(state.to_dict())
            serialized_events.append(serialize_feedback_event(event))
    final_overlay = overlay.state
    payload = {
        "session_log": str(logger.path),
        "events": serialized_events,
        "overlay_snapshots": snapshots,
        "final_overlay": final_overlay.to_dict(),
        "final_overlay_rendered": renderer.render(final_overlay),
    }
    payload.update(extra)
    return payload


def _replay_event(record: Any) -> object:
    return SimpleNamespace(
        phase=record.phase,
        transcript=record.transcript,
        message=record.message,
        target=record.target,
        decision=record.decision,
        result=record.result,
    )


def _effective_dry_run(settings, args: argparse.Namespace) -> bool:
    return False if getattr(args, "execute", False) else settings.dry_run


def _settings_payload(settings, effective_dry_run: bool) -> dict[str, Any]:
    payload = dict(settings.__dict__)
    payload["effective_dry_run"] = effective_dry_run
    return payload


def _cursor_settings_payload(args: argparse.Namespace) -> dict[str, float]:
    return {
        "deadzone": max(0.0, float(getattr(args, "cursor_deadzone", 0.015))),
        "smoothing": max(0.0, min(1.0, float(getattr(args, "cursor_smoothing", 0.28)))),
        "edge_padding": max(0.0, min(0.45, float(getattr(args, "cursor_edge_padding", 0.03)))),
        "max_step": max(0.01, min(1.0, float(getattr(args, "cursor_max_step", 0.16)))),
    }


def _gaze_control_settings_payload(args: argparse.Namespace) -> dict[str, int]:
    return {
        "action_cooldown_ms": max(0, int(getattr(args, "action_cooldown_ms", 900))),
        "drag_timeout_ms": max(0, int(getattr(args, "drag_timeout_ms", 3000))),
    }


def _gaze_visualization_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "show_dot": bool(getattr(args, "show_dot", False)),
        "dot_size": max(12, int(getattr(args, "dot_size", 28))),
        "status": "disabled",
    }


def _gaze_action_name(value: str) -> ActionName:
    mapping = {
        "highlight": ActionName.HIGHLIGHT_TARGET,
        "move": ActionName.FOCUS_TARGET,
        "click": ActionName.CLICK_TARGET,
        "right-click": ActionName.RIGHT_CLICK_TARGET,
        "double-click": ActionName.DOUBLE_CLICK_TARGET,
        "drag": ActionName.DRAG_TARGET,
        "cursor": ActionName.FOCUS_TARGET,
    }
    return mapping[value]


def _gaze_auto_confirm_actions(action: ActionName) -> set[ActionName]:
    if action in {ActionName.CLICK_TARGET, ActionName.RIGHT_CLICK_TARGET, ActionName.DOUBLE_CLICK_TARGET, ActionName.DRAG_TARGET}:
        return {action}
    return set()


def _profile_to_payload(profile) -> dict[str, Any] | None:
    return asdict(profile) if profile is not None else None
