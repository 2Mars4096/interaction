"""Small CLI entry point for repeatable local testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
from interaction.platform import MacOSPlatformAdapter, current_position_normalized
from interaction.runtime import (
    GazeTrackingLoop,
    MultimodalFusionLoop,
    VoiceCommandLoop,
    WebcamCalibrationError,
    capture_live_gaze_context,
    collect_webcam_calibration,
    collect_webcam_validation,
    default_webcam_targets,
    generate_single_screen_calibration_targets,
    generate_single_screen_validation_targets,
    run_live_cursor_follow,
    run_live_webcam_trace,
)
from interaction.session import SessionLogger, SessionReplay, serialize_feedback_event
from interaction.ui import CalibrationDotOverlay, ConsoleOverlayRenderer, LiveWebcamDebugPreview, LiveGazeDotOverlay, OverlayController
from interaction.vision import (
    CalibrationProfile,
    CalibrationSample,
    DwellTrigger,
    GazeSample,
    GazeSmoother,
    MediaPipeWebcamGazeProvider,
    NormalizedPoint,
    NormalizedScreenTarget,
    OpenCVWebcamGazeProvider,
    WebcamProviderError,
)


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
    elif args.command == "gaze-validate":
        payload = _run_gaze_validate(args, store)
    elif args.command == "gaze-refine":
        payload = _run_gaze_refine(args, store)
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
    gaze_calibrate.add_argument("--tracker", choices=["opencv", "mediapipe"], default="opencv")
    gaze_calibrate.add_argument("--tracker-model-path", default=None)
    gaze_calibrate.add_argument("--frames-per-step", type=int, default=6)
    gaze_calibrate.add_argument("--delta-ms", type=int, default=100)
    gaze_calibrate.add_argument("--settle-ms", type=int, default=800)
    gaze_calibrate.add_argument("--dot-size", type=int, default=40)
    gaze_calibrate.add_argument("--random-seed", type=int, default=None)
    gaze_calibrate.add_argument("--runtime-dir", default=".interaction")
    gaze_calibrate.add_argument("--session-name", default="gaze-calibrate")

    gaze_validate = subparsers.add_parser("gaze-validate")
    gaze_validate.add_argument("--camera-index", type=int, default=None)
    gaze_validate.add_argument("--tracker", choices=["opencv", "mediapipe"], default="opencv")
    gaze_validate.add_argument("--tracker-model-path", default=None)
    gaze_validate.add_argument("--points", type=int, default=6)
    gaze_validate.add_argument("--frames-per-step", type=int, default=4)
    gaze_validate.add_argument("--delta-ms", type=int, default=100)
    gaze_validate.add_argument("--dot-size", type=int, default=40)
    gaze_validate.add_argument("--random-seed", type=int, default=None)
    gaze_validate.add_argument("--runtime-dir", default=".interaction")
    gaze_validate.add_argument("--session-name", default="gaze-validate")

    gaze_refine = subparsers.add_parser("gaze-refine")
    gaze_refine.add_argument("--camera-index", type=int, default=None)
    gaze_refine.add_argument("--tracker", choices=["opencv", "mediapipe"], default="opencv")
    gaze_refine.add_argument("--tracker-model-path", default=None)
    gaze_refine.add_argument("--points", type=int, default=6)
    gaze_refine.add_argument("--delta-ms", type=int, default=100)
    gaze_refine.add_argument("--dot-size", type=int, default=40)
    gaze_refine.add_argument("--random-seed", type=int, default=None)
    gaze_refine.add_argument("--recalibrate-first", action="store_true")
    gaze_refine.add_argument("--calibration-frames-per-step", type=int, default=4)
    gaze_refine.add_argument("--calibration-settle-ms", type=int, default=800)
    gaze_refine.add_argument("--runtime-dir", default=".interaction")
    gaze_refine.add_argument("--session-name", default="gaze-refine")

    gaze = subparsers.add_parser("gaze-smoke")
    gaze.add_argument("--action", choices=["highlight", "move", "click", "right-click", "double-click", "drag", "cursor"], default="highlight")
    gaze.add_argument("--action-cooldown-ms", type=int, default=900)
    gaze.add_argument("--drag-timeout-ms", type=int, default=3000)
    gaze.add_argument("--cursor-deadzone", type=float, default=0.05)
    gaze.add_argument("--cursor-smoothing", type=float, default=0.1)
    gaze.add_argument("--cursor-edge-padding", type=float, default=0.05)
    gaze.add_argument("--cursor-max-step", type=float, default=0.04)
    gaze.add_argument("--cursor-gain", type=float, default=0.35)
    gaze.add_argument("--cursor-gain-x", type=float, default=None)
    gaze.add_argument("--cursor-gain-y", type=float, default=None)
    gaze.add_argument("--runtime-dir", default=".interaction")
    gaze.add_argument("--session-name", default="gaze-smoke")
    gaze.add_argument("--execute", action="store_true")

    gaze_live = subparsers.add_parser("gaze-live")
    gaze_live.add_argument("--camera-index", type=int, default=None)
    gaze_live.add_argument("--tracker", choices=["opencv", "mediapipe"], default="opencv")
    gaze_live.add_argument("--tracker-model-path", default=None)
    gaze_live.add_argument("--frames", type=int, default=18)
    gaze_live.add_argument("--delta-ms", type=int, default=100)
    gaze_live.add_argument("--action", choices=["highlight", "move", "click", "right-click", "double-click", "drag", "cursor"], default="highlight")
    gaze_live.add_argument("--action-cooldown-ms", type=int, default=900)
    gaze_live.add_argument("--drag-timeout-ms", type=int, default=3000)
    gaze_live.add_argument("--cursor-deadzone", type=float, default=0.05)
    gaze_live.add_argument("--cursor-smoothing", type=float, default=0.1)
    gaze_live.add_argument("--cursor-edge-padding", type=float, default=0.05)
    gaze_live.add_argument("--cursor-max-step", type=float, default=0.04)
    gaze_live.add_argument("--cursor-gain", type=float, default=0.35)
    gaze_live.add_argument("--cursor-gain-x", type=float, default=None)
    gaze_live.add_argument("--cursor-gain-y", type=float, default=None)
    gaze_live.add_argument("--show-dot", action="store_true")
    gaze_live.add_argument("--show-camera", action="store_true")
    gaze_live.add_argument("--dot-size", type=int, default=28)
    gaze_live.add_argument("--camera-preview-width", type=int, default=420)
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
    fusion_live.add_argument("--tracker", choices=["opencv", "mediapipe"], default="opencv")
    fusion_live.add_argument("--tracker-model-path", default=None)
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


def _perform_live_gaze_calibration(
    args: argparse.Namespace,
    store: JsonStateStore,
    *,
    camera_index: int,
    tracker_model_path: str | None,
    frames_per_step: int,
    settle_ms: int,
) -> dict[str, Any]:
    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    events: list[object] = []
    calibration_targets = generate_single_screen_calibration_targets(seed=args.random_seed)
    interactive_calibration = _is_interactive_calibration_session()
    calibration_visualization = _calibration_visualization_payload(args, interactive=interactive_calibration)
    calibration_quality = None
    point_visualizer = None
    profile = None

    if interactive_calibration:
        try:
            point_visualizer = CalibrationDotOverlay(size=max(16, args.dot_size), fill="#0a84ff")
            point_visualizer.open()
            calibration_visualization["status"] = "enabled"
        except Exception as error:
            calibration_visualization["status"] = "error"
            calibration_visualization["error"] = str(error)

    try:
        provider.open()
        profile, step_results, calibration_quality = collect_webcam_calibration(
            provider,
            targets=calibration_targets,
            frames_per_step=frames_per_step,
            delta_ms=args.delta_ms,
            before_step=lambda step, step_index, total_steps: _append_calibration_prompt(
                events,
                step,
                step_index=step_index,
                total_steps=total_steps,
                frames_per_step=frames_per_step,
                settle_ms=settle_ms,
                interactive=interactive_calibration,
                point_visualizer=point_visualizer,
            ),
        )
    except WebcamProviderError as error:
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message))
        calibration = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
        step_results = []
        profile_payload = None
    except WebcamCalibrationError as error:
        _write_live_calibration_invalid_marker(
            store.paths,
            code=error.code,
            message=error.message,
            payload=error.payload,
        )
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message))
        calibration = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
        step_results = list(error.payload.get("steps", []))
        profile_payload = None
        calibration_quality = error.payload.get("quality")
        profile = None
    else:
        _clear_live_calibration_invalid_marker(store.paths)
        store.save_calibration_profile(profile, name="webcam-live")
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.CALIBRATING, message="Saved live webcam calibration profile."))
        calibration = {"status": "success"}
        profile_payload = _profile_to_payload(profile)
    finally:
        provider.close()
        if point_visualizer is not None:
            point_visualizer.close()

    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
    return {
        "events": events,
        "calibration": calibration,
        "calibration_steps": step_results,
        "calibration_targets": [
            {
                "label": step.label,
                "expected": {"x": step.expected.x, "y": step.expected.y},
            }
            for step in calibration_targets
        ],
        "calibration_visualization": calibration_visualization,
        "calibration_quality": calibration_quality,
        "calibration_profile": profile_payload,
        "profile": profile,
    }


def _run_gaze_calibrate(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    tracker_model_path = _resolve_tracker_model_path(args, store.paths)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    calibration_run = _perform_live_gaze_calibration(
        args,
        store,
        camera_index=camera_index,
        tracker_model_path=tracker_model_path,
        frames_per_step=args.frames_per_step,
        settle_ms=args.settle_ms,
    )
    return _build_payload(
        source="gaze",
        events=calibration_run["events"],
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": settings.__dict__,
            "camera_index": camera_index,
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "calibration": calibration_run["calibration"],
            "calibration_steps": calibration_run["calibration_steps"],
            "calibration_targets": calibration_run["calibration_targets"],
            "calibration_visualization": calibration_run["calibration_visualization"],
            "calibration_quality": calibration_run["calibration_quality"],
            "calibration_profile": calibration_run["calibration_profile"],
        },
    )


def _run_gaze_validate(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    tracker_model_path = _resolve_tracker_model_path(args, store.paths)
    invalid_calibration = _load_live_calibration_invalid_marker(store.paths)
    profile = store.load_calibration_profile("webcam-live")
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events: list[object] = []

    if invalid_calibration is not None:
        events = [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING,
                message=(
                    f'{invalid_calibration["message"]} '
                    "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before validation."
                ),
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
                "settings": settings.__dict__,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "validation": {
                    "status": "error",
                    "error": "invalid_calibration_profile",
                    "message": (
                        f'{invalid_calibration["message"]} '
                        "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before validation."
                    ),
                    "payload": invalid_calibration.get("payload", {}),
                },
                "validation_points": [],
                "validation_targets": [],
                "validation_visualization": _calibration_visualization_payload(
                    args,
                    interactive=_is_interactive_calibration_session(),
                ),
                "calibration_profile": None,
            },
        )

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
                "settings": settings.__dict__,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "validation": {
                    "status": "error",
                    "error": "missing_calibration_profile",
                    "message": "No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
                },
                "validation_points": [],
                "validation_targets": [],
                "validation_visualization": _calibration_visualization_payload(
                    args,
                    interactive=_is_interactive_calibration_session(),
                ),
                "calibration_profile": None,
            },
        )

    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    validation_targets = generate_single_screen_validation_targets(count=args.points, seed=args.random_seed)
    interactive_validation = _is_interactive_calibration_session()
    validation_visualization = _calibration_visualization_payload(args, interactive=interactive_validation)
    point_visualizer = None
    if interactive_validation:
        try:
            point_visualizer = CalibrationDotOverlay(size=max(16, args.dot_size), fill="#0a84ff")
            point_visualizer.open()
            validation_visualization["status"] = "enabled"
        except Exception as error:
            validation_visualization["status"] = "error"
            validation_visualization["error"] = str(error)

    try:
        provider.open()
        if interactive_validation and point_visualizer is not None:
            validation_points = []
            for step_index, step in enumerate(validation_targets, start=1):
                _append_validation_prompt(
                    events,
                    step,
                    step_index=step_index,
                    total_steps=len(validation_targets),
                    frames_per_step=args.frames_per_step,
                    interactive=True,
                    point_visualizer=None,
                    live_tracking=True,
                )
                validation_points.append(
                    _capture_interactive_validation_point(
                        provider,
                        profile,
                        step,
                        step_index=step_index,
                        total_steps=len(validation_targets),
                        delta_ms=args.delta_ms,
                        smoothing_window=max(6, min(10, args.frames_per_step * 2)),
                        point_visualizer=point_visualizer,
                    )
                )
            summary = _summarize_validation_points(validation_points)
        else:
            validation_points, summary = collect_webcam_validation(
                provider,
                profile,
                targets=validation_targets,
                frames_per_step=args.frames_per_step,
                delta_ms=args.delta_ms,
                before_step=lambda step, step_index, total_steps: _append_validation_prompt(
                    events,
                    step,
                    step_index=step_index,
                    total_steps=total_steps,
                    frames_per_step=args.frames_per_step,
                    interactive=interactive_validation,
                    point_visualizer=point_visualizer,
                ),
            )
    except WebcamProviderError as error:
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message))
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
        validation = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
        validation_points = []
        summary = {
            "attempted_points": len(validation_targets),
            "successful_points": 0,
            "missing_points": len(validation_targets),
            "mean_error_distance": 0.0,
            "max_error_distance": 0.0,
        }
    else:
        if int(summary["successful_points"]) == 0:
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="No usable gaze validation points were captured.",
                )
            )
            validation = {
                "status": "error",
                "error": "missing_validation_samples",
                "message": "No usable gaze validation points were captured.",
                **summary,
            }
        else:
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.CALIBRATING,
                    message=(
                        f'Validation captured {summary["successful_points"]}/{summary["attempted_points"]} points. '
                        f'Mean normalized error: {summary["mean_error_distance"]:.3f}.'
                    ),
                )
            )
            validation = {
                "status": "success",
                **summary,
            }
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
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
            "settings": settings.__dict__,
            "camera_index": camera_index,
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "validation": validation,
            "validation_points": validation_points,
            "validation_targets": [
                {
                    "label": step.label,
                    "expected": {"x": step.expected.x, "y": step.expected.y},
                }
                for step in validation_targets
            ],
            "validation_visualization": validation_visualization,
            "calibration_profile": _profile_to_payload(profile),
        },
    )


def _run_gaze_refine(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    tracker_model_path = _resolve_tracker_model_path(args, store.paths)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    events: list[object] = []
    calibration_stage = None

    if args.recalibrate_first:
        calibration_stage = _perform_live_gaze_calibration(
            args,
            store,
            camera_index=camera_index,
            tracker_model_path=tracker_model_path,
            frames_per_step=max(1, args.calibration_frames_per_step),
            settle_ms=max(0, args.calibration_settle_ms),
        )
        events.extend(calibration_stage["events"])
        if calibration_stage["calibration"]["status"] != "success":
            return _build_payload(
                source="gaze",
                events=events,
                logger=logger,
                overlay=overlay,
                renderer=renderer,
                extra={
                    "settings": settings.__dict__,
                    "camera_index": camera_index,
                    "tracker": args.tracker,
                    "tracker_model_path": tracker_model_path,
                    "refinement": {
                        "status": "error",
                        "error": "calibration_failed",
                        "message": "The initial calibration stage failed, so refinement was not started.",
                        "payload": calibration_stage["calibration"],
                    },
                    "initial_calibration": {
                        "calibration": calibration_stage["calibration"],
                        "calibration_steps": calibration_stage["calibration_steps"],
                        "calibration_targets": calibration_stage["calibration_targets"],
                        "calibration_visualization": calibration_stage["calibration_visualization"],
                        "calibration_quality": calibration_stage["calibration_quality"],
                        "calibration_profile": calibration_stage["calibration_profile"],
                    },
                    "refinement_points": [],
                    "refinement_targets": [],
                    "refinement_visualization": _calibration_visualization_payload(
                        args,
                        interactive=_is_interactive_calibration_session(),
                    ),
                    "refinement_quality": None,
                    "calibration_profile": calibration_stage["calibration_profile"],
                    "refined_profile": None,
                    "correction_profile": None,
                },
            )

    invalid_calibration = _load_live_calibration_invalid_marker(store.paths)
    base_profile = calibration_stage["profile"] if calibration_stage is not None else store.load_calibration_profile("webcam-live")

    if invalid_calibration is not None:
        events = [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING,
                message=(
                    f'{invalid_calibration["message"]} '
                    "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before refinement."
                ),
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
                "settings": settings.__dict__,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "initial_calibration": (
                    {
                        "calibration": calibration_stage["calibration"],
                        "calibration_steps": calibration_stage["calibration_steps"],
                        "calibration_targets": calibration_stage["calibration_targets"],
                        "calibration_visualization": calibration_stage["calibration_visualization"],
                        "calibration_quality": calibration_stage["calibration_quality"],
                        "calibration_profile": calibration_stage["calibration_profile"],
                    }
                    if calibration_stage is not None
                    else None
                ),
                "refinement": {
                    "status": "error",
                    "error": "invalid_calibration_profile",
                    "message": (
                        f'{invalid_calibration["message"]} '
                        "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before refinement."
                    ),
                    "payload": invalid_calibration.get("payload", {}),
                },
                "refinement_points": [],
                "refinement_targets": [],
                "refinement_visualization": _calibration_visualization_payload(
                    args,
                    interactive=_is_interactive_calibration_session(),
                ),
                "refinement_quality": None,
                "calibration_profile": None,
                "refined_profile": None,
                "correction_profile": None,
            },
        )

    if base_profile is None:
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
                "settings": settings.__dict__,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "initial_calibration": (
                    {
                        "calibration": calibration_stage["calibration"],
                        "calibration_steps": calibration_stage["calibration_steps"],
                        "calibration_targets": calibration_stage["calibration_targets"],
                        "calibration_visualization": calibration_stage["calibration_visualization"],
                        "calibration_quality": calibration_stage["calibration_quality"],
                        "calibration_profile": calibration_stage["calibration_profile"],
                    }
                    if calibration_stage is not None
                    else None
                ),
                "refinement": {
                    "status": "error",
                    "error": "missing_calibration_profile",
                    "message": "No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
                },
                "refinement_points": [],
                "refinement_targets": [],
                "refinement_visualization": _calibration_visualization_payload(
                    args,
                    interactive=_is_interactive_calibration_session(),
                ),
                "refinement_quality": None,
                "calibration_profile": None,
                "refined_profile": None,
                "correction_profile": None,
            },
        )

    interactive_refinement = _is_interactive_calibration_session()
    if not interactive_refinement:
        events = [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING,
                message="Gaze refinement requires an interactive overlay session so you can click the point you were looking at.",
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
                "settings": settings.__dict__,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "initial_calibration": (
                    {
                        "calibration": calibration_stage["calibration"],
                        "calibration_steps": calibration_stage["calibration_steps"],
                        "calibration_targets": calibration_stage["calibration_targets"],
                        "calibration_visualization": calibration_stage["calibration_visualization"],
                        "calibration_quality": calibration_stage["calibration_quality"],
                        "calibration_profile": calibration_stage["calibration_profile"],
                    }
                    if calibration_stage is not None
                    else None
                ),
                "refinement": {
                    "status": "error",
                    "error": "interactive_overlay_required",
                    "message": "Gaze refinement requires an interactive overlay session so you can click the point you were looking at.",
                },
                "refinement_points": [],
                "refinement_targets": [],
                "refinement_visualization": _calibration_visualization_payload(args, interactive=False),
                "refinement_quality": None,
                "calibration_profile": _profile_to_payload(base_profile),
                "refined_profile": None,
                "correction_profile": None,
            },
        )

    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    refinement_targets = generate_single_screen_validation_targets(count=args.points, seed=args.random_seed)
    refinement_visualization = _calibration_visualization_payload(args, interactive=True)
    point_visualizer = None
    try:
        point_visualizer = CalibrationDotOverlay(size=max(16, args.dot_size), fill="#0a84ff")
        point_visualizer.open()
        refinement_visualization["status"] = "enabled"
    except Exception as error:
        refinement_visualization["status"] = "error"
        refinement_visualization["error"] = str(error)

    refinement_points: list[dict[str, Any]] = []
    correction_profile = None
    refined_profile = None
    refinement_quality = None
    try:
        provider.open()
        if point_visualizer is None:
            raise WebcamProviderError(
                "overlay_unavailable",
                "Could not open the interactive refinement overlay.",
            )
        for step_index, step in enumerate(refinement_targets, start=1):
            _append_refinement_prompt(
                events,
                step,
                step_index=step_index,
                total_steps=len(refinement_targets),
            )
            refinement_points.append(
                _capture_interactive_refinement_point(
                    provider,
                    base_profile,
                    step,
                    step_index=step_index,
                    total_steps=len(refinement_targets),
                    delta_ms=args.delta_ms,
                    smoothing_window=8,
                    point_visualizer=point_visualizer,
                )
            )
    except WebcamProviderError as error:
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message))
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
        refinement = {
            "status": "error",
            "error": error.code,
            "message": error.message,
            "payload": error.payload,
        }
    else:
        successful_points = [point for point in refinement_points if point.get("success")]
        if len(successful_points) < 3:
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.RECOVERING,
                    message="Not enough refinement clicks were captured. Retry and click the blue target while keeping your gaze fixed.",
                )
            )
            refinement = {
                "status": "error",
                "error": "insufficient_refinement_samples",
                "message": "Not enough refinement clicks were captured. Retry and click the blue target while keeping your gaze fixed.",
                **_summarize_refinement_points(refinement_points),
            }
        else:
            summary = _summarize_refinement_points(refinement_points)
            refinement_quality = _refinement_signal_quality(successful_points)
            correction_samples = [
                CalibrationSample(
                    raw=NormalizedPoint(point["tracked"]["x"], point["tracked"]["y"]),
                    expected=NormalizedPoint(point["clicked"]["x"], point["clicked"]["y"]),
                )
                for point in successful_points
            ]
            correction_profile = CalibrationProfile.fit(correction_samples)
            suppressed_axes = _suppressed_refinement_axes(refinement_quality)
            if "x" in suppressed_axes or "y" in suppressed_axes:
                correction_profile = CalibrationProfile(
                    x_scale=1.0 if "x" in suppressed_axes else correction_profile.x_scale,
                    y_scale=1.0 if "y" in suppressed_axes else correction_profile.y_scale,
                    x_offset=0.0 if "x" in suppressed_axes else correction_profile.x_offset,
                    y_offset=0.0 if "y" in suppressed_axes else correction_profile.y_offset,
                )

            if len(suppressed_axes) == 2:
                message = (
                    "The refinement clicks did not contain enough usable tracked movement on either axis. "
                    "The tracked point was too compressed or clipped to learn a correction."
                )
                _write_live_calibration_invalid_marker(
                    store.paths,
                    code="invalid_refinement_signal",
                    message=message,
                    payload={"summary": summary, "quality": refinement_quality},
                )
                correction_profile = None
                refined_profile = None
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.RECOVERING,
                        message=message,
                    )
                )
                refinement = {
                    "status": "error",
                    "error": "invalid_refinement_signal",
                    "message": message,
                    "payload": {"summary": summary, "quality": refinement_quality},
                    **summary,
                }
            else:
                refined_profile = base_profile.compose(correction_profile)
                store.save_calibration_profile(refined_profile, name="webcam-live")
                _clear_live_calibration_invalid_marker(store.paths)
                if suppressed_axes:
                    axis_text = "vertical" if suppressed_axes == ["y"] else "horizontal" if suppressed_axes == ["x"] else "one-axis"
                    message = (
                        f'Refinement captured {summary["successful_points"]}/{summary["attempted_points"]} points. '
                        f'Mean click error: {summary["mean_error_distance"]:.3f}. '
                        f'Saved updated live webcam calibration profile with {axis_text} correction disabled because the tracked axis signal was saturated.'
                    )
                else:
                    message = (
                        f'Refinement captured {summary["successful_points"]}/{summary["attempted_points"]} points. '
                        f'Mean click error: {summary["mean_error_distance"]:.3f}. Saved updated live webcam calibration profile.'
                    )
                events.append(
                    GazeFeedbackEvent(
                        phase=GazeLoopPhase.CALIBRATING,
                        message=message,
                    )
                )
                refinement = {
                    "status": "success",
                    "suppressed_axes": suppressed_axes,
                    **summary,
                }
        events.append(GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."))
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
            "settings": settings.__dict__,
            "camera_index": camera_index,
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "initial_calibration": (
                {
                    "calibration": calibration_stage["calibration"],
                    "calibration_steps": calibration_stage["calibration_steps"],
                    "calibration_targets": calibration_stage["calibration_targets"],
                    "calibration_visualization": calibration_stage["calibration_visualization"],
                    "calibration_quality": calibration_stage["calibration_quality"],
                    "calibration_profile": calibration_stage["calibration_profile"],
                }
                if calibration_stage is not None
                else None
            ),
            "refinement": refinement,
            "refinement_points": refinement_points,
            "refinement_targets": [
                {
                    "label": step.label,
                    "expected": {"x": step.expected.x, "y": step.expected.y},
                }
                for step in refinement_targets
            ],
            "refinement_visualization": refinement_visualization,
            "refinement_quality": refinement_quality,
            "calibration_profile": _profile_to_payload(base_profile),
            "refined_profile": _profile_to_payload(refined_profile),
            "correction_profile": _profile_to_payload(correction_profile),
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
    tracker_model_path = _resolve_tracker_model_path(args, store.paths)
    invalid_calibration = _load_live_calibration_invalid_marker(store.paths)
    profile = store.load_calibration_profile("webcam-live")
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    visualization = _gaze_visualization_payload(args)
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
                "cursor_settings": _cursor_settings_payload(args),
                "gaze_control_settings": _gaze_control_settings_payload(args),
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "live_gaze": {
                    "status": "error",
                    "error": "missing_calibration_profile",
                    "message": "No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
                },
                "gaze_visualization": visualization,
                "calibration_profile": None,
            },
        )

    if invalid_calibration is not None:
        events = [
            GazeFeedbackEvent(
                phase=GazeLoopPhase.RECOVERING,
                message=(
                    f'{invalid_calibration["message"]} '
                    "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before live tracking."
                ),
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
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "live_gaze": {
                    "status": "error",
                    "error": "invalid_calibration_profile",
                    "message": (
                        f'{invalid_calibration["message"]} '
                        "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before live tracking."
                    ),
                    "payload": invalid_calibration.get("payload", {}),
                },
                "gaze_visualization": visualization,
                "calibration_profile": None,
            },
        )

    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    point_visualizer = None
    camera_preview = None
    initial_cursor_point = _current_cursor_point() if gaze_mode == "cursor" else None
    if args.show_dot:
        try:
            point_visualizer = LiveGazeDotOverlay(size=max(12, args.dot_size))
            point_visualizer.open()
            visualization["status"] = "enabled"
        except Exception as error:
            visualization["status"] = "error"
            visualization["error"] = str(error)
    if args.show_camera:
        try:
            camera_preview = LiveWebcamDebugPreview(max_width=max(240, args.camera_preview_width))
            camera_preview.open()
            visualization["camera_status"] = "enabled"
        except Exception as error:
            visualization["camera_status"] = "error"
            visualization["camera_error"] = str(error)
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
                response_gain_x=_cursor_axis_gain(args, axis="x"),
                response_gain_y=_cursor_axis_gain(args, axis="y"),
                point_visualizer=point_visualizer,
                initial_cursor_point=initial_cursor_point,
                camera_preview=camera_preview,
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
                camera_preview=camera_preview,
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
        if camera_preview is not None:
            camera_preview.close()
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
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "live_gaze": live_gaze,
            "initial_cursor_point": (
                {"x": initial_cursor_point[0], "y": initial_cursor_point[1]}
                if initial_cursor_point is not None
                else None
            ),
            "cursor_settings": _cursor_settings_payload(args),
            "gaze_control_settings": _gaze_control_settings_payload(args),
            "gaze_visualization": visualization,
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
    tracker_model_path = _resolve_tracker_model_path(args, store.paths)
    invalid_calibration = _load_live_calibration_invalid_marker(store.paths)
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
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
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

    if invalid_calibration is not None:
        events = [
            FusionFeedbackEvent(
                phase=FusionLoopPhase.RECOVERING,
                message=(
                    f'{invalid_calibration["message"]} '
                    "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before fusion."
                ),
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
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "live_gaze_context": {
                    "status": "error",
                    "error": "invalid_calibration_profile",
                    "message": (
                        f'{invalid_calibration["message"]} '
                        "The previous saved profile has been marked stale. Run `interaction gaze-calibrate` again before fusion."
                    ),
                    "payload": invalid_calibration.get("payload", {}),
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
    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
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
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
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
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
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
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
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
                    "tracker": args.tracker,
                    "tracker_model_path": tracker_model_path,
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
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
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


def _append_calibration_prompt(
    events: list[object],
    step,
    *,
    step_index: int,
    total_steps: int,
    frames_per_step: int,
    settle_ms: int,
    interactive: bool,
    point_visualizer: Any | None,
) -> None:
    if point_visualizer is not None:
        try:
            point_visualizer.show_point(
                step.expected.x,
                step.expected.y,
                step_index=step_index,
                total_steps=total_steps,
            )
        except TypeError:
            point_visualizer.show_point(step.expected.x, step.expected.y)
    if point_visualizer is not None:
        mode_message = "Fixate on the on-screen dot, then press Enter, Space, or click on the overlay to capture."
    elif interactive:
        mode_message = "Fixate on the dot and press Enter in the terminal to capture."
    else:
        mode_message = "Auto-capturing this point without terminal confirmation."
    events.append(
        GazeFeedbackEvent(
            phase=GazeLoopPhase.CALIBRATING,
            message=(
                f'Calibration point {step_index}/{total_steps} is ready near the {step.label} of the primary display. '
                f'{mode_message} Capturing {frames_per_step} stable webcam readings.'
            ),
        )
    )
    if interactive:
        _wait_for_calibration_confirmation(
            step_index=step_index,
            total_steps=total_steps,
            label=step.label,
            point_visualizer=point_visualizer,
        )
    elif settle_ms > 0:
        time.sleep(settle_ms / 1000.0)


def _wait_for_calibration_confirmation(
    *,
    step_index: int,
    total_steps: int,
    label: str,
    point_visualizer: Any | None,
    on_iteration: Any | None = None,
) -> None:
    if point_visualizer is not None and hasattr(point_visualizer, "wait_for_confirmation"):
        try:
            point_visualizer.wait_for_confirmation(on_iteration=on_iteration)
        except TypeError:
            point_visualizer.wait_for_confirmation()
        return
    print(
        (
            f"[interaction] Calibration {step_index}/{total_steps}: look at the dot near the {label} "
            "of the primary display, then press Enter to capture."
        ),
        file=sys.stderr,
        flush=True,
    )
    try:
        input()
    except EOFError:
        return


def _is_interactive_calibration_session() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _append_validation_prompt(
    events: list[object],
    step,
    *,
    step_index: int,
    total_steps: int,
    frames_per_step: int,
    interactive: bool,
    point_visualizer: Any | None,
    live_tracking: bool = False,
) -> None:
    if point_visualizer is not None:
        try:
            point_visualizer.show_point(
                step.expected.x,
                step.expected.y,
                step_index=step_index,
                total_steps=total_steps,
            )
        except TypeError:
            point_visualizer.show_point(step.expected.x, step.expected.y)
    if live_tracking:
        mode_message = "Move your gaze until the red tracked dot overlaps the blue target, then click, press Space, or press Enter."
    elif point_visualizer is not None:
        mode_message = "Fixate on the on-screen dot, then press Enter, Space, or click on the overlay to capture."
    elif interactive:
        mode_message = "Fixate on the dot and press Enter in the terminal to capture."
    else:
        mode_message = "Auto-capturing this test point without terminal confirmation."
    events.append(
        GazeFeedbackEvent(
            phase=GazeLoopPhase.CALIBRATING,
            message=(
                f"Validation point {step_index}/{total_steps} is ready at {step.label}. "
                f"{mode_message} Capturing {frames_per_step} stable webcam readings."
            ),
        )
    )
    if interactive and not live_tracking:
        _wait_for_calibration_confirmation(
            step_index=step_index,
            total_steps=total_steps,
            label=step.label,
            point_visualizer=point_visualizer,
        )


def _capture_interactive_validation_point(
    provider,
    profile,
    step,
    *,
    step_index: int,
    total_steps: int,
    delta_ms: int,
    smoothing_window: int,
    point_visualizer: Any,
) -> dict[str, Any]:
    smoother = GazeSmoother(window_size=max(2, smoothing_window))
    latest_raw: NormalizedPoint | None = None
    latest_tracked: NormalizedPoint | None = None
    latest_confidence = 0.0
    attempted_frames = 0
    used_frames = 0
    sample_interval_s = max(0.05, delta_ms / 1000.0)
    last_sample_at = 0.0

    def _update_live_tracking() -> None:
        nonlocal latest_raw, latest_tracked, latest_confidence, attempted_frames, used_frames, last_sample_at
        now = time.monotonic()
        if now - last_sample_at < sample_interval_s:
            return
        last_sample_at = now
        reading = provider.read(delta_ms=delta_ms)
        attempted_frames += 1
        if reading is None:
            try:
                point_visualizer.show_validation_point(
                    step.expected.x,
                    step.expected.y,
                    tracked_x=None,
                    tracked_y=None,
                    step_index=step_index,
                    total_steps=total_steps,
                )
            except Exception:
                pass
            return

        tracked_point = profile.apply(reading.sample.point)
        observation = smoother.smooth(
            GazeSample(
                point=tracked_point,
                confidence=reading.sample.confidence,
                delta_ms=reading.sample.delta_ms,
            )
        )
        if observation.x_norm is None or observation.y_norm is None:
            return
        latest_raw = reading.sample.point
        latest_tracked = NormalizedPoint(observation.x_norm, observation.y_norm)
        latest_confidence = observation.confidence
        used_frames += 1
        try:
            point_visualizer.show_validation_point(
                step.expected.x,
                step.expected.y,
                tracked_x=latest_tracked.x,
                tracked_y=latest_tracked.y,
                step_index=step_index,
                total_steps=total_steps,
            )
        except Exception:
            pass

    try:
        point_visualizer.show_validation_point(
            step.expected.x,
            step.expected.y,
            tracked_x=None,
            tracked_y=None,
            step_index=step_index,
            total_steps=total_steps,
        )
    except Exception:
        pass

    _wait_for_calibration_confirmation(
        step_index=step_index,
        total_steps=total_steps,
        label=step.label,
        point_visualizer=point_visualizer,
        on_iteration=_update_live_tracking,
    )

    if latest_raw is None or latest_tracked is None:
        return {
            "label": step.label,
            "expected": {"x": step.expected.x, "y": step.expected.y},
            "success": False,
            "used_frames": used_frames,
            "attempted_frames": attempted_frames,
        }

    error_x = latest_tracked.x - step.expected.x
    error_y = latest_tracked.y - step.expected.y
    error_distance = (error_x**2 + error_y**2) ** 0.5
    return {
        "label": step.label,
        "expected": {"x": step.expected.x, "y": step.expected.y},
        "raw": {"x": latest_raw.x, "y": latest_raw.y},
        "tracked": {"x": latest_tracked.x, "y": latest_tracked.y},
        "error": {"dx": error_x, "dy": error_y, "distance": error_distance},
        "success": True,
        "confidence": latest_confidence,
        "used_frames": used_frames,
        "attempted_frames": attempted_frames,
        "captured_on_confirmation": True,
    }


def _append_refinement_prompt(
    events: list[object],
    step,
    *,
    step_index: int,
    total_steps: int,
) -> None:
    events.append(
        GazeFeedbackEvent(
            phase=GazeLoopPhase.CALIBRATING,
            message=(
                f"Refinement point {step_index}/{total_steps} is ready at {step.label}. "
                "Look at the blue target, then click exactly where you were looking. "
                "The red dot is the current tracked point and will be used only as feedback."
            ),
        )
    )


def _capture_interactive_refinement_point(
    provider,
    profile,
    step,
    *,
    step_index: int,
    total_steps: int,
    delta_ms: int,
    smoothing_window: int,
    point_visualizer: Any,
) -> dict[str, Any]:
    smoother = GazeSmoother(window_size=max(2, smoothing_window))
    latest_raw: NormalizedPoint | None = None
    latest_tracked: NormalizedPoint | None = None
    latest_confidence = 0.0
    attempted_frames = 0
    used_frames = 0
    sample_interval_s = max(0.05, delta_ms / 1000.0)
    last_sample_at = 0.0

    def _update_live_tracking() -> None:
        nonlocal latest_raw, latest_tracked, latest_confidence, attempted_frames, used_frames, last_sample_at
        now = time.monotonic()
        if now - last_sample_at < sample_interval_s:
            return
        last_sample_at = now
        reading = provider.read(delta_ms=delta_ms)
        attempted_frames += 1
        if reading is None:
            try:
                point_visualizer.show_validation_point(
                    step.expected.x,
                    step.expected.y,
                    tracked_x=None,
                    tracked_y=None,
                    step_index=step_index,
                    total_steps=total_steps,
                )
            except Exception:
                pass
            return

        tracked_point = profile.apply(reading.sample.point)
        observation = smoother.smooth(
            GazeSample(
                point=tracked_point,
                confidence=reading.sample.confidence,
                delta_ms=reading.sample.delta_ms,
            )
        )
        if observation.x_norm is None or observation.y_norm is None:
            return
        latest_raw = reading.sample.point
        latest_tracked = NormalizedPoint(observation.x_norm, observation.y_norm)
        latest_confidence = observation.confidence
        used_frames += 1
        try:
            point_visualizer.show_validation_point(
                step.expected.x,
                step.expected.y,
                tracked_x=latest_tracked.x,
                tracked_y=latest_tracked.y,
                step_index=step_index,
                total_steps=total_steps,
            )
        except Exception:
            pass

    try:
        point_visualizer.show_validation_point(
            step.expected.x,
            step.expected.y,
            tracked_x=None,
            tracked_y=None,
            step_index=step_index,
            total_steps=total_steps,
        )
    except Exception:
        pass

    clicked_x, clicked_y = point_visualizer.wait_for_click_point(on_iteration=_update_live_tracking)
    clicked_point = NormalizedPoint(clicked_x, clicked_y)

    if latest_raw is None or latest_tracked is None:
        return {
            "label": step.label,
            "expected": {"x": step.expected.x, "y": step.expected.y},
            "clicked": {"x": clicked_point.x, "y": clicked_point.y},
            "success": False,
            "used_frames": used_frames,
            "attempted_frames": attempted_frames,
        }

    error_x = latest_tracked.x - clicked_point.x
    error_y = latest_tracked.y - clicked_point.y
    error_distance = (error_x**2 + error_y**2) ** 0.5
    return {
        "label": step.label,
        "expected": {"x": step.expected.x, "y": step.expected.y},
        "clicked": {"x": clicked_point.x, "y": clicked_point.y},
        "raw": {"x": latest_raw.x, "y": latest_raw.y},
        "tracked": {"x": latest_tracked.x, "y": latest_tracked.y},
        "error": {"dx": error_x, "dy": error_y, "distance": error_distance},
        "success": True,
        "confidence": latest_confidence,
        "used_frames": used_frames,
        "attempted_frames": attempted_frames,
        "captured_on_click": True,
    }


def _summarize_validation_points(points: list[dict[str, Any]]) -> dict[str, float | int]:
    successful_results = [result for result in points if result.get("success")]
    distances = [float(result["error"]["distance"]) for result in successful_results]
    attempted_points = len(points)
    return {
        "attempted_points": attempted_points,
        "successful_points": len(successful_results),
        "missing_points": attempted_points - len(successful_results),
        "mean_error_distance": (sum(distances) / len(distances)) if distances else 0.0,
        "max_error_distance": max(distances) if distances else 0.0,
    }


def _summarize_refinement_points(points: list[dict[str, Any]]) -> dict[str, float | int]:
    return _summarize_validation_points(points)


def _refinement_signal_quality(points: list[dict[str, Any]]) -> dict[str, float | int]:
    tracked_xs = [float(point["tracked"]["x"]) for point in points]
    tracked_ys = [float(point["tracked"]["y"]) for point in points]
    return {
        "sample_count": len(points),
        "tracked_x_span": max(tracked_xs) - min(tracked_xs),
        "tracked_y_span": max(tracked_ys) - min(tracked_ys),
        "tracked_x_saturated_count": sum(1 for value in tracked_xs if value <= 0.01 or value >= 0.99),
        "tracked_y_saturated_count": sum(1 for value in tracked_ys if value <= 0.01 or value >= 0.99),
    }


def _suppressed_refinement_axes(quality: dict[str, float | int]) -> list[str]:
    sample_count = max(1, int(quality["sample_count"]))
    suppressed: list[str] = []
    if float(quality["tracked_x_span"]) < 0.06 or int(quality["tracked_x_saturated_count"]) >= max(2, sample_count // 2):
        suppressed.append("x")
    if float(quality["tracked_y_span"]) < 0.06 or int(quality["tracked_y_saturated_count"]) >= max(2, sample_count // 2):
        suppressed.append("y")
    return suppressed


def _calibration_visualization_payload(args: argparse.Namespace, *, interactive: bool) -> dict[str, Any]:
    payload = {
        "dot_size": max(16, int(getattr(args, "dot_size", 40))),
        "interactive": interactive,
        "status": "disabled",
    }
    if interactive:
        payload["display"] = "primary-screen"
    return payload


def _current_cursor_point() -> tuple[float, float] | None:
    try:
        x, y = current_position_normalized()
    except Exception:
        return None
    return x, y


def _resolve_tracker_model_path(args: argparse.Namespace, runtime_paths: RuntimePaths) -> str | None:
    raw_path = getattr(args, "tracker_model_path", None)
    if raw_path:
        return str(Path(raw_path).expanduser())
    default_model_path = runtime_paths.base_dir / "models" / "face_landmarker.task"
    if default_model_path.is_file():
        return str(default_model_path)
    return None


def _live_calibration_invalid_marker_path(runtime_paths: RuntimePaths) -> Path:
    return runtime_paths.profiles_dir / "webcam-live.invalid.json"


def _load_live_calibration_invalid_marker(runtime_paths: RuntimePaths) -> dict[str, Any] | None:
    path = _live_calibration_invalid_marker_path(runtime_paths)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"code": "invalid_calibration_profile", "message": "The live calibration profile is marked stale.", "payload": {}}


def _write_live_calibration_invalid_marker(
    runtime_paths: RuntimePaths,
    *,
    code: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    runtime_paths.ensure()
    marker_payload = {"code": code, "message": message, "payload": payload or {}}
    _live_calibration_invalid_marker_path(runtime_paths).write_text(json.dumps(marker_payload, indent=2) + "\n")


def _clear_live_calibration_invalid_marker(runtime_paths: RuntimePaths) -> None:
    path = _live_calibration_invalid_marker_path(runtime_paths)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _build_webcam_provider(tracker: str, camera_index: int, *, tracker_model_path: str | None = None):
    if tracker == "mediapipe":
        return MediaPipeWebcamGazeProvider(camera_index=camera_index, model_asset_path=tracker_model_path)
    return OpenCVWebcamGazeProvider(camera_index=camera_index)


def _effective_dry_run(settings, args: argparse.Namespace) -> bool:
    return False if getattr(args, "execute", False) else settings.dry_run


def _settings_payload(settings, effective_dry_run: bool) -> dict[str, Any]:
    payload = dict(settings.__dict__)
    payload["effective_dry_run"] = effective_dry_run
    return payload


def _cursor_settings_payload(args: argparse.Namespace) -> dict[str, float]:
    return {
        "deadzone": max(0.0, float(getattr(args, "cursor_deadzone", 0.05))),
        "smoothing": max(0.0, min(1.0, float(getattr(args, "cursor_smoothing", 0.1)))),
        "edge_padding": max(0.0, min(0.45, float(getattr(args, "cursor_edge_padding", 0.05)))),
        "max_step": max(0.01, min(1.0, float(getattr(args, "cursor_max_step", 0.04)))),
        "gain": max(0.05, min(1.0, float(getattr(args, "cursor_gain", 0.35)))),
        "gain_x": _cursor_axis_gain(args, axis="x"),
        "gain_y": _cursor_axis_gain(args, axis="y"),
    }


def _cursor_axis_gain(args: argparse.Namespace, *, axis: str) -> float:
    base_gain = max(0.05, min(1.0, float(getattr(args, "cursor_gain", 0.35))))
    axis_value = getattr(args, f"cursor_gain_{axis}", None)
    if axis_value is not None:
        return max(0.05, min(1.0, float(axis_value)))
    if axis == "x":
        return max(0.05, min(1.0, base_gain * 0.7))
    return max(0.05, min(1.0, base_gain * 1.45))


def _gaze_control_settings_payload(args: argparse.Namespace) -> dict[str, int]:
    return {
        "action_cooldown_ms": max(0, int(getattr(args, "action_cooldown_ms", 900))),
        "drag_timeout_ms": max(0, int(getattr(args, "drag_timeout_ms", 3000))),
    }


def _gaze_visualization_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "show_dot": bool(getattr(args, "show_dot", False)),
        "show_camera": bool(getattr(args, "show_camera", False)),
        "dot_size": max(12, int(getattr(args, "dot_size", 28))),
        "camera_preview_width": max(240, int(getattr(args, "camera_preview_width", 420))),
        "status": "disabled",
        "camera_status": "disabled",
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


def _profile_to_payload(profile) -> dict[str, float] | None:
    if profile is None:
        return None
    return {
        "x_scale": profile.x_scale,
        "y_scale": profile.y_scale,
        "x_offset": profile.x_offset,
        "y_offset": profile.y_offset,
    }
