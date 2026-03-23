"""Small CLI entry point for repeatable local testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
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
from interaction.runtime import (
    GazeTrackingLoop,
    MultimodalFusionLoop,
    VoiceCommandLoop,
    WebcamCalibrationError,
    capture_live_gaze_context,
    collect_webcam_calibration,
    default_webcam_calibration_targets,
    default_webcam_targets,
    run_live_cursor_follow,
    run_live_webcam_trace,
)
from interaction.session import SessionLogger, SessionReplay, serialize_feedback_event
from interaction.ui import ConsoleOverlayRenderer, OverlayController
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

    gaze_calibrate = subparsers.add_parser("gaze-calibrate")
    gaze_calibrate.add_argument("--camera-index", type=int, default=None)
    gaze_calibrate.add_argument("--frames-per-step", type=int, default=6)
    gaze_calibrate.add_argument("--delta-ms", type=int, default=100)
    gaze_calibrate.add_argument("--settle-ms", type=int, default=800)
    gaze_calibrate.add_argument("--runtime-dir", default=".interaction")
    gaze_calibrate.add_argument("--session-name", default="gaze-calibrate")

    gaze = subparsers.add_parser("gaze-smoke")
    gaze.add_argument("--action", choices=["highlight", "move", "click", "right-click", "double-click", "drag", "cursor"], default="highlight")
    gaze.add_argument("--runtime-dir", default=".interaction")
    gaze.add_argument("--session-name", default="gaze-smoke")
    gaze.add_argument("--execute", action="store_true")

    gaze_live = subparsers.add_parser("gaze-live")
    gaze_live.add_argument("--camera-index", type=int, default=None)
    gaze_live.add_argument("--frames", type=int, default=18)
    gaze_live.add_argument("--delta-ms", type=int, default=100)
    gaze_live.add_argument("--action", choices=["highlight", "move", "click", "right-click", "double-click", "drag", "cursor"], default="highlight")
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
    fusion_live.add_argument("--gaze-delta-ms", type=int, default=100)
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
    events: list[object] = [
        VoiceFeedbackEvent(
            phase=VoiceLoopPhase.LISTENING,
            message=f"Starting live macOS speech capture for {args.duration:.1f}s.",
        )
    ]
    live_capture: dict[str, Any] | None = None
    try:
        capture = provider.capture_turn(duration_s=args.duration, locale=args.locale)
    except SpeechCaptureError as error:
        events.append(
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.RECOVERING,
                message=error.message,
            )
        )
        events.append(VoiceFeedbackEvent(phase=VoiceLoopPhase.IDLE, message="Voice loop is idle."))
        live_capture = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
    else:
        live_capture = {
            "status": "success",
            "transcript": capture.transcript,
            "confidence": capture.confidence,
            "locale": capture.locale,
            "duration_s": capture.duration_s,
            "provider": capture.provider,
            "used_on_device": capture.used_on_device,
            "permission_state": capture.permission_state,
        }
        events.append(
            VoiceFeedbackEvent(
                phase=VoiceLoopPhase.INTERPRETING,
                transcript=capture.transcript,
                message="Live macOS speech capture completed.",
            )
        )
        events.extend(
            loop.run_text_turn(
                capture.transcript,
                EnvironmentSnapshot(active_app="Finder"),
                final_confidence=capture.confidence,
                begin_turn=False,
            )
        )
    return _build_payload(
        source="voice",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run),
            "live_capture": live_capture,
            "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
        },
    )


def _run_gaze_calibrate(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    provider = OpenCVWebcamGazeProvider(camera_index=camera_index)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events: list[object] = []
    try:
        provider.open()
        profile, step_results = collect_webcam_calibration(
            provider,
            frames_per_step=args.frames_per_step,
            delta_ms=args.delta_ms,
            before_step=lambda step: _append_calibration_prompt(events, step.label, args.frames_per_step, args.settle_ms),
        )
    except WebcamProviderError as error:
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message))
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
        calibration = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
        step_results = []
        profile_payload = None
    except WebcamCalibrationError as error:
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message))
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
        calibration = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
        step_results = list(error.payload.get("steps", []))
        profile_payload = None
    else:
        store.save_calibration_profile(profile, name="webcam-live")
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.CALIBRATING, message="Saved live webcam calibration profile."))
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
        calibration = {"status": "success"}
        profile_payload = {
            "x_scale": profile.x_scale,
            "y_scale": profile.y_scale,
            "x_offset": profile.x_offset,
            "y_offset": profile.y_offset,
        }
    finally:
        provider.close()
    return _build_payload(
        source="gaze",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": settings.__dict__,
            "camera_index": camera_index,
            "calibration": calibration,
            "calibration_steps": step_results,
            "calibration_profile": profile_payload,
        },
    )


def _run_gaze_smoke(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    gaze_mode = args.action
    gaze_action = _gaze_action_name(gaze_mode)
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=effective_dry_run),
        dwell_trigger=DwellTrigger(dwell_ms=settings.dwell_ms, action=gaze_action),
        auto_confirm_actions=_gaze_auto_confirm_actions(gaze_action),
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
    profile = store.load_calibration_profile("webcam-live")
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    if profile is None:
        events = [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING,
                message="No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
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
                "camera_index": camera_index,
                "live_gaze": {
                    "status": "error",
                    "error": "missing_calibration_profile",
                    "message": "No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
                },
                "calibration_profile": None,
            },
        )

    provider = OpenCVWebcamGazeProvider(camera_index=camera_index)
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=effective_dry_run),
        dwell_trigger=DwellTrigger(dwell_ms=settings.dwell_ms, action=gaze_action),
        auto_confirm_actions=_gaze_auto_confirm_actions(gaze_action),
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
            )
        else:
            events, summary = run_live_webcam_trace(
                provider,
                loop,
                frames=args.frames,
                delta_ms=args.delta_ms,
                targets=default_webcam_targets(),
                environment=EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera"),
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
            "calibration_profile": {
                "x_scale": profile.x_scale,
                "y_scale": profile.y_scale,
                "x_offset": profile.x_offset,
                "y_offset": profile.y_offset,
            },
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
    profile = store.load_calibration_profile("webcam-live")
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=effective_dry_run))
    loop.state.gaze_context_window_ms = settings.gaze_context_window_ms
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()

    if profile is None:
        events: list[object] = [
            FusionFeedbackEvent(
                phase=FusionLoopPhase.RECOVERING,
                message="No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
            ),
            FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."),
        ]
        return _build_payload(
            source="fusion",
            events=events,
            logger=logger,
            overlay=overlay,
            renderer=renderer,
            extra={
                "settings": _settings_payload(settings, effective_dry_run),
                "camera_index": camera_index,
                "live_gaze_context": {
                    "status": "error",
                    "error": "missing_calibration_profile",
                    "message": "No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
                },
                "live_capture": None,
                "confirmation_capture": None,
                "metrics": loop.metrics.to_dict(),
                "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
                "calibration_profile": None,
            },
        )

    environment = EnvironmentSnapshot(active_app="Finder", active_window_title="Live Fusion")
    events = [
        FusionFeedbackEvent(
            phase=FusionLoopPhase.TRACKING,
            message=f"Capturing live gaze context from webcam for {args.gaze_frames} frames.",
        )
    ]
    provider = OpenCVWebcamGazeProvider(camera_index=camera_index)
    live_gaze_context: dict[str, Any]
    try:
        provider.open()
        gaze_context, gaze_events, gaze_summary = capture_live_gaze_context(
            provider,
            profile,
            frames=args.gaze_frames,
            delta_ms=args.gaze_delta_ms,
            targets=default_webcam_targets(),
        )
        events.extend(gaze_events)
    except WebcamProviderError as error:
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=error.message))
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
        live_gaze_context = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
        return _build_payload(
            source="fusion",
            events=events,
            logger=logger,
            overlay=overlay,
            renderer=renderer,
            extra={
                "settings": _settings_payload(settings, effective_dry_run),
                "camera_index": camera_index,
                "live_gaze_context": live_gaze_context,
                "live_capture": None,
                "confirmation_capture": None,
                "metrics": loop.metrics.to_dict(),
                "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
                "calibration_profile": _profile_to_payload(profile),
            },
        )
    finally:
        provider.close()

    if gaze_context is None:
        events.append(
            FusionFeedbackEvent(
                phase=FusionLoopPhase.RECOVERING,
                message="No grounded live gaze target was established. Look steadily at a large target and retry.",
            )
        )
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
        live_gaze_context = {
            "status": "error",
            "error": "missing_gaze_lock",
            "message": "No grounded live gaze target was established. Look steadily at a large target and retry.",
            **gaze_summary,
        }
        return _build_payload(
            source="fusion",
            events=events,
            logger=logger,
            overlay=overlay,
            renderer=renderer,
            extra={
                "settings": _settings_payload(settings, effective_dry_run),
                "camera_index": camera_index,
                "live_gaze_context": live_gaze_context,
                "live_capture": None,
                "confirmation_capture": None,
                "metrics": loop.metrics.to_dict(),
                "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
                "calibration_profile": _profile_to_payload(profile),
            },
        )

    live_gaze_context = {
        "status": "success",
        "target": gaze_context.target.model_dump(mode="json"),
        "observation": gaze_context.observation.model_dump(mode="json"),
        "attempted_frames": gaze_context.attempted_frames,
        "successful_readings": gaze_context.successful_readings,
        "missing_readings": gaze_context.missing_readings,
        "grounded_readings": gaze_context.grounded_readings,
        "dominant_target_hits": gaze_context.dominant_target_hits,
    }
    events.append(loop.update_gaze_context(gaze_context.observation, gaze_context.target))

    speech_provider = MacOSLiveSpeechProvider(store.paths)
    live_capture: dict[str, Any] | None = None
    confirmation_capture: dict[str, Any] | None = None
    events.append(
        FusionFeedbackEvent(
            phase=FusionLoopPhase.TRACKING,
            message=f"Starting live macOS speech capture for {args.duration:.1f}s.",
        )
    )
    try:
        capture = speech_provider.capture_turn(duration_s=args.duration, locale=args.locale)
    except SpeechCaptureError as error:
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=error.message))
        events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle."))
        live_capture = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
        return _build_payload(
            source="fusion",
            events=events,
            logger=logger,
            overlay=overlay,
            renderer=renderer,
            extra={
                "settings": _settings_payload(settings, effective_dry_run),
                "camera_index": camera_index,
                "live_gaze_context": live_gaze_context,
                "live_capture": live_capture,
                "confirmation_capture": confirmation_capture,
                "metrics": loop.metrics.to_dict(),
                "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
                "calibration_profile": _profile_to_payload(profile),
            },
        )

    live_capture = {
        "status": "success",
        "transcript": capture.transcript,
        "confidence": capture.confidence,
        "locale": capture.locale,
        "duration_s": capture.duration_s,
        "provider": capture.provider,
        "used_on_device": capture.used_on_device,
        "permission_state": capture.permission_state,
    }
    events.append(
        FusionFeedbackEvent(
            phase=FusionLoopPhase.INTERPRETING,
            transcript=capture.transcript,
            message="Live macOS speech capture completed for fusion.",
            target=gaze_context.target,
        )
    )
    events.extend(
        loop.run_voice_turn(
            capture.transcript,
            environment,
            final_confidence=capture.confidence,
        )
    )

    if loop.pending_decision is not None:
        events.append(
            FusionFeedbackEvent(
                phase=FusionLoopPhase.CONFIRMING,
                message=f"Starting live confirmation capture for {args.confirm_duration:.1f}s.",
                target=gaze_context.target,
            )
        )
        try:
            confirmation = speech_provider.capture_turn(duration_s=args.confirm_duration, locale=args.locale)
        except SpeechCaptureError as error:
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.RECOVERING, message=error.message, target=gaze_context.target))
            events.append(FusionFeedbackEvent(phase=FusionLoopPhase.IDLE, message="Fusion loop is idle.", target=gaze_context.target))
            confirmation_capture = {
                "status": "error",
                "error": error.code,
                "message": error.message,
                "payload": error.payload,
            }
            return _build_payload(
                source="fusion",
                events=events,
                logger=logger,
                overlay=overlay,
                renderer=renderer,
                extra={
                    "settings": _settings_payload(settings, effective_dry_run),
                    "camera_index": camera_index,
                    "live_gaze_context": live_gaze_context,
                    "live_capture": live_capture,
                    "confirmation_capture": confirmation_capture,
                    "metrics": loop.metrics.to_dict(),
                    "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
                    "calibration_profile": _profile_to_payload(profile),
                },
            )

        confirmation_capture = {
            "status": "success",
            "transcript": confirmation.transcript,
            "confidence": confirmation.confidence,
            "locale": confirmation.locale,
            "duration_s": confirmation.duration_s,
            "provider": confirmation.provider,
            "used_on_device": confirmation.used_on_device,
            "permission_state": confirmation.permission_state,
        }
        events.append(
            FusionFeedbackEvent(
                phase=FusionLoopPhase.INTERPRETING,
                transcript=confirmation.transcript,
                message="Live macOS confirmation capture completed.",
                target=gaze_context.target,
            )
        )
        events.extend(
            loop.run_voice_turn(
                confirmation.transcript,
                environment,
                final_confidence=confirmation.confidence,
            )
        )

    return _build_payload(
        source="fusion",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run),
            "camera_index": camera_index,
            "live_gaze_context": live_gaze_context,
            "live_capture": live_capture,
            "confirmation_capture": confirmation_capture,
            "metrics": loop.metrics.to_dict(),
            "platform_capabilities": loop.adapter.describe_capabilities().__dict__,
            "calibration_profile": _profile_to_payload(profile),
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
    snapshots: list[dict[str, Any]] = []
    serialized_events: list[dict[str, Any]] = []
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


def _append_calibration_prompt(events: list[object], label: str, frames_per_step: int, settle_ms: int) -> None:
    events.append(
        GazeFeedbackEvent(
            phase=GazeLoopPhase.CALIBRATING,
            message=f'Look at the {label} reference point. Capturing {frames_per_step} stable webcam readings.',
        )
    )
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)


def _effective_dry_run(settings, args: argparse.Namespace) -> bool:
    return False if getattr(args, "execute", False) else settings.dry_run


def _settings_payload(settings, effective_dry_run: bool) -> dict[str, Any]:
    payload = dict(settings.__dict__)
    payload["effective_dry_run"] = effective_dry_run
    return payload


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


def _profile_to_payload(profile) -> dict[str, float] | None:
    if profile is None:
        return None
    return {
        "x_scale": profile.x_scale,
        "y_scale": profile.y_scale,
        "x_offset": profile.x_offset,
        "y_offset": profile.y_offset,
    }
