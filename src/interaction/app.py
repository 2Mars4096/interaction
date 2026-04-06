"""Small CLI entry point for repeatable local testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from typing import Any, cast

from interaction.audio import MacOSLiveSpeechProvider, SpeechCaptureError
from interaction.contracts import ActionName, BoundingBox, DesktopTargetCandidate, EnvironmentSnapshot, GazeObservation, GroundedTarget, SessionMode
from interaction.feedback import (
    FusionFeedbackEvent,
    FusionLoopPhase,
    GazeFeedbackEvent,
    GazeLoopPhase,
    VoiceFeedbackEvent,
    VoiceLoopPhase,
)
from interaction.intent import GemmaIntentInterpreter, InterpreterStrategy, VoiceInterpretationRouter
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.platform import MacOSPlatformAdapter, capture_frontmost_environment, current_position_normalized, normalized_to_screen_point
from interaction.platform.base import TargetCandidateQuery
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
    run_live_assisted_cursor_follow,
    run_live_cursor_follow,
    run_live_cursor_snap,
    run_live_relative_cursor_follow,
    run_live_webcam_benchmark,
    run_live_wink_click_test,
    run_live_webcam_trace,
)
from interaction.session import SessionLogger, SessionReplay, serialize_feedback_event
from interaction.ui import CalibrationDotOverlay, ConsoleOverlayRenderer, LiveGazeDotOverlay, LiveSnapDotsOverlay, LiveWebcamDebugPreview, OverlayController
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
    elif args.command in {"voice-live", "voice-dictate"}:
        payload = _run_voice_live(args, store)
    elif args.command == "gaze-calibrate":
        payload = _run_gaze_calibrate(args, store)
    elif args.command == "gaze-validate":
        payload = _run_gaze_validate(args, store)
    elif args.command == "gaze-refine":
        payload = _run_gaze_refine(args, store)
    elif args.command == "gaze-smoke":
        payload = _run_gaze_smoke(args, store)
    elif args.command in {"gaze-live", "gaze-live-easy"}:
        payload = _run_gaze_live(args, store)
    elif args.command == "gaze-wink-test":
        payload = _run_gaze_wink_test(args, store)
    elif args.command == "gaze-benchmark":
        payload = _run_gaze_benchmark(args, store)
    elif args.command in {"gaze-benchmark-report", "gaze-benchmark-brief"}:
        payload = _run_gaze_benchmark_report(args)
    elif args.command == "fusion-smoke":
        payload = _run_fusion_smoke(args, store)
    elif args.command == "fusion-live":
        payload = _run_fusion_live(args, store)
    else:
        payload = _run_replay(args)
    if args.command in {"gaze-benchmark-report", "gaze-benchmark-brief"}:
        rendering = _benchmark_report_rendering_payload(args)
        rendered_output = _render_benchmark_report_output(
            payload,
            output_format=str(rendering["format"]),
            brief=bool(rendering["brief"]),
            top=rendering["top"],
        )
        output_path = rendering["output"]
        if output_path:
            output_text = rendered_output if rendered_output.endswith("\n") else f"{rendered_output}\n"
            Path(output_path).write_text(output_text)
        else:
            print(rendered_output)
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interaction", description="Repeatable local smoke and replay CLI.")
    subparsers = parser.add_subparsers(dest="command")

    voice = subparsers.add_parser("voice-smoke")
    voice.add_argument("--utterance", default="open Safari")
    voice.add_argument("--follow-up", default=None)
    voice.add_argument("--lane", choices=["command", "dictation", "hybrid"], default="command")
    voice.add_argument("--runtime-dir", default=".interaction")
    voice.add_argument("--session-name", default="voice-smoke")
    voice.add_argument("--execute", action="store_true")
    _add_interpreter_argument(voice)

    voice_live = subparsers.add_parser("voice-live")
    voice_live.add_argument("--duration", type=float, default=4.0)
    voice_live.add_argument("--locale", default="en-US")
    voice_live.add_argument("--lane", choices=["command", "dictation", "hybrid"], default="command")
    voice_live.add_argument("--streaming", action="store_true")
    voice_live.add_argument("--pause-ms", type=int, default=700)
    voice_live.add_argument("--save-audio", action="store_true")
    voice_live.add_argument("--audio-path", default=None)
    voice_live.add_argument("--runtime-dir", default=".interaction")
    voice_live.add_argument("--session-name", default="voice-live")
    voice_live.add_argument("--execute", action="store_true")
    _add_interpreter_argument(voice_live)

    voice_dictate = subparsers.add_parser("voice-dictate")
    voice_dictate.add_argument("--duration", type=float, default=10.0)
    voice_dictate.add_argument("--locale", default="en-US")
    voice_dictate.add_argument("--pause-ms", type=int, default=700)
    voice_dictate.add_argument("--save-audio", action="store_true")
    voice_dictate.add_argument("--audio-path", default=None)
    voice_dictate.add_argument("--runtime-dir", default=".interaction")
    voice_dictate.add_argument("--session-name", default="voice-dictate")
    voice_dictate.add_argument("--execute", action="store_true")
    voice_dictate.set_defaults(lane="dictation", streaming=True)
    _add_interpreter_argument(voice_dictate)

    gaze_calibrate = subparsers.add_parser("gaze-calibrate")
    gaze_calibrate.add_argument("--camera-index", type=int, default=None)
    gaze_calibrate.add_argument("--tracker", choices=["opencv", "mediapipe"], default="opencv")
    gaze_calibrate.add_argument("--tracker-model-path", default=None)
    gaze_calibrate.add_argument("--frames-per-step", type=int, default=6)
    gaze_calibrate.add_argument("--delta-ms", type=int, default=100)
    gaze_calibrate.add_argument("--settle-ms", type=int, default=800)
    gaze_calibrate.add_argument("--dot-size", type=int, default=40)
    gaze_calibrate.add_argument("--calibration-grid-rows", type=int, choices=[3, 5], default=3)
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
    gaze_refine.add_argument("--calibration-grid-rows", type=int, choices=[3, 5], default=3)
    gaze_refine.add_argument("--random-seed", type=int, default=None)
    gaze_refine.add_argument("--recalibrate-first", action="store_true")
    gaze_refine.add_argument("--calibration-frames-per-step", type=int, default=4)
    gaze_refine.add_argument("--calibration-settle-ms", type=int, default=800)
    gaze_refine.add_argument("--runtime-dir", default=".interaction")
    gaze_refine.add_argument("--session-name", default="gaze-refine")

    gaze = subparsers.add_parser("gaze-smoke")
    gaze.add_argument("--action", choices=["highlight", "move", "click", "right-click", "double-click", "drag", "cursor", "cursor-relative", "cursor-relative-click", "cursor-assist", "cursor-assist-click", "cursor-snap", "cursor-snap-click", "cursor-snap-right-click", "cursor-snap-wink"], default="highlight")
    gaze.add_argument("--action-cooldown-ms", type=int, default=900)
    gaze.add_argument("--drag-timeout-ms", type=int, default=3000)
    gaze.add_argument("--dwell-ms", type=int, default=None)
    gaze.add_argument("--dwell-min-confidence", type=float, default=None)
    gaze.add_argument("--cursor-deadzone", type=float, default=None)
    gaze.add_argument("--cursor-smoothing", type=float, default=None)
    gaze.add_argument("--cursor-edge-padding", type=float, default=None)
    gaze.add_argument("--cursor-max-step", type=float, default=None)
    gaze.add_argument("--cursor-gain", type=float, default=None)
    gaze.add_argument("--cursor-gain-x", type=float, default=None)
    gaze.add_argument("--cursor-gain-y", type=float, default=None)
    gaze.add_argument("--runtime-dir", default=".interaction")
    gaze.add_argument("--session-name", default="gaze-smoke")
    gaze.add_argument("--execute", action="store_true")

    gaze_live = subparsers.add_parser("gaze-live")
    _add_gaze_live_arguments(
        gaze_live,
        calibration_dot_size_default=44,
        calibration_frames_per_step_default=6,
    )

    gaze_live_easy = subparsers.add_parser("gaze-live-easy")
    _add_gaze_live_arguments(
        gaze_live_easy,
        tracker_default="mediapipe",
        frames_default=10000,
        action_default="cursor-snap",
        show_dot_default=True,
        show_camera_default=False,
        dot_size_default=24,
        session_name_default="gaze-live-easy",
        recalibrate_first_default=True,
        calibration_dot_size_default=44,
        calibration_frames_per_step_default=6,
    )

    gaze_wink_test = subparsers.add_parser("gaze-wink-test")
    gaze_wink_test.add_argument("--camera-index", type=int, default=None)
    gaze_wink_test.add_argument("--tracker", choices=["opencv", "mediapipe"], default="mediapipe")
    gaze_wink_test.add_argument("--tracker-model-path", default=None)
    gaze_wink_test.add_argument("--frames", type=int, default=300)
    gaze_wink_test.add_argument("--delta-ms", type=int, default=41)
    gaze_wink_test.add_argument("--wink-hold-ms", type=int, default=180)
    gaze_wink_test.add_argument("--wink-closed-threshold", type=float, default=0.45)
    gaze_wink_test.add_argument("--wink-open-threshold", type=float, default=0.25)
    gaze_wink_test.add_argument("--stop-after-clicks", type=int, default=2)
    gaze_wink_test.add_argument("--action-cooldown-ms", type=int, default=900)
    gaze_wink_test.add_argument("--min-confidence", "--dwell-min-confidence", dest="dwell_min_confidence", type=float, default=None)
    gaze_wink_test.set_defaults(show_dot=False, show_camera=False)
    gaze_wink_test.add_argument("--show-dot", dest="show_dot", action="store_true")
    gaze_wink_test.add_argument("--hide-dot", dest="show_dot", action="store_false")
    gaze_wink_test.add_argument("--show-camera", dest="show_camera", action="store_true")
    gaze_wink_test.add_argument("--hide-camera", dest="show_camera", action="store_false")
    gaze_wink_test.add_argument("--dot-size", type=int, default=18)
    gaze_wink_test.add_argument("--camera-preview-width", type=int, default=420)
    gaze_wink_test.add_argument("--runtime-dir", default=".interaction")
    gaze_wink_test.add_argument("--session-name", default="gaze-wink-test")
    gaze_wink_test.add_argument("--execute", action="store_true")

    gaze_benchmark = subparsers.add_parser(
        "gaze-benchmark",
        help="Capture a read-only webcam benchmark summary.",
        description=(
            "Capture a read-only webcam benchmark run that records signal quality, jitter, "
            "drift, confidence, and wink episodes without moving the cursor or clicking."
        ),
    )
    gaze_benchmark.add_argument("--camera-index", type=int, default=None)
    gaze_benchmark.add_argument("--tracker", choices=["opencv", "mediapipe"], default="mediapipe")
    gaze_benchmark.add_argument("--tracker-model-path", default=None)
    gaze_benchmark.add_argument("--frames", type=int, default=180)
    gaze_benchmark.add_argument("--delta-ms", type=int, default=41)
    gaze_benchmark.add_argument("--wink-closed-threshold", type=float, default=0.45)
    gaze_benchmark.add_argument("--wink-open-threshold", type=float, default=0.25)
    gaze_benchmark.set_defaults(show_dot=False, show_camera=False)
    gaze_benchmark.add_argument("--show-dot", dest="show_dot", action="store_true")
    gaze_benchmark.add_argument("--hide-dot", dest="show_dot", action="store_false")
    gaze_benchmark.add_argument("--show-camera", dest="show_camera", action="store_true")
    gaze_benchmark.add_argument("--hide-camera", dest="show_camera", action="store_false")
    gaze_benchmark.add_argument("--dot-size", type=int, default=18)
    gaze_benchmark.add_argument("--camera-preview-width", type=int, default=420)
    gaze_benchmark.add_argument("--runtime-dir", default=".interaction")
    gaze_benchmark.add_argument("--session-name", default="gaze-benchmark")

    gaze_benchmark_report = subparsers.add_parser(
        "gaze-benchmark-report",
        help="Summarize saved benchmark runs in JSON or text form.",
        description=(
            "Summarize saved gaze benchmark summaries. Use JSON for machine-readable output, "
            "full text for terminal review, or brief text for quick scans."
        ),
        epilog=(
            "Examples:\n"
            "  interaction gaze-benchmark-report --runtime-dir .interaction --limit 10\n"
            "  interaction gaze-benchmark-report --runtime-dir .interaction --preset full-text\n"
            "  interaction gaze-benchmark-report --runtime-dir .interaction --preset brief-text --top 1\n"
            "  interaction gaze-benchmark-report --runtime-dir .interaction --preset json --output benchmark-report.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gaze_benchmark_report.add_argument("--runtime-dir", default=".interaction")
    gaze_benchmark_report.add_argument("--summary-path", action="append", default=None)
    gaze_benchmark_report.add_argument("--tracker", choices=["opencv", "mediapipe"], default=None)
    gaze_benchmark_report.add_argument("--limit", type=int, default=10)
    gaze_benchmark_report.add_argument("--preset", choices=["json", "full-text", "brief-text"], default=None)
    gaze_benchmark_report.add_argument("--format", choices=["json", "text"], default="json")
    gaze_benchmark_report.add_argument("--brief", action="store_true")
    gaze_benchmark_report.add_argument("--output", default=None)
    gaze_benchmark_report.add_argument("--top", type=int, default=None)

    gaze_benchmark_brief = subparsers.add_parser(
        "gaze-benchmark-brief",
        help="Print the common brief text benchmark scan.",
        description=(
            "Shortcut for the common brief benchmark scan. This is equivalent to "
            "`gaze-benchmark-report --preset brief-text`."
        ),
    )
    gaze_benchmark_brief.add_argument("--runtime-dir", default=".interaction")
    gaze_benchmark_brief.add_argument("--summary-path", action="append", default=None)
    gaze_benchmark_brief.add_argument("--tracker", choices=["opencv", "mediapipe"], default=None)
    gaze_benchmark_brief.add_argument("--limit", type=int, default=10)
    gaze_benchmark_brief.add_argument("--output", default=None)
    gaze_benchmark_brief.add_argument("--top", type=int, default=None)
    gaze_benchmark_brief.set_defaults(preset="brief-text", format="text", brief=True)

    fusion = subparsers.add_parser("fusion-smoke")
    fusion.add_argument("--utterance", default="open this")
    fusion.add_argument("--follow-up", default="yes")
    fusion.add_argument("--runtime-dir", default=".interaction")
    fusion.add_argument("--session-name", default="fusion-smoke")
    fusion.add_argument("--execute", action="store_true")
    _add_interpreter_argument(fusion)

    fusion_live = subparsers.add_parser("fusion-live")
    fusion_live.add_argument("--camera-index", type=int, default=None)
    fusion_live.add_argument("--tracker", choices=["opencv", "mediapipe"], default="opencv")
    fusion_live.add_argument("--tracker-model-path", default=None)
    fusion_live.add_argument("--gaze-frames", type=int, default=12)
    fusion_live.add_argument("--gaze-delta-ms", type=int, default=41)
    fusion_live.add_argument("--duration", type=float, default=4.0)
    fusion_live.add_argument("--confirm-duration", type=float, default=2.5)
    fusion_live.add_argument("--locale", default="en-US")
    fusion_live.add_argument("--runtime-dir", default=".interaction")
    fusion_live.add_argument("--session-name", default="fusion-live")
    fusion_live.add_argument("--execute", action="store_true")
    _add_interpreter_argument(fusion_live)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--session-log", required=True)
    replay.add_argument("--runtime-dir", default=".interaction")
    return parser


def _add_gaze_live_arguments(
    parser: argparse.ArgumentParser,
    *,
    tracker_default: str = "opencv",
    frames_default: int = 10000,
    action_default: str = "highlight",
    show_dot_default: bool = False,
    show_camera_default: bool = False,
    dot_size_default: int = 28,
    session_name_default: str = "gaze-live",
    recalibrate_first_default: bool = False,
    calibration_dot_size_default: int | None = None,
    calibration_frames_per_step_default: int = 4,
) -> None:
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--tracker", choices=["opencv", "mediapipe"], default=tracker_default)
    parser.add_argument("--tracker-model-path", default=None)
    parser.add_argument("--frames", type=int, default=frames_default)
    parser.add_argument("--delta-ms", type=int, default=41)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--calibration-grid-rows", type=int, choices=[3, 5], default=3)
    parser.add_argument(
        "--action",
        choices=[
            "highlight",
            "move",
            "click",
            "right-click",
            "double-click",
            "drag",
            "cursor",
            "cursor-relative",
            "cursor-relative-click",
            "cursor-assist",
            "cursor-assist-click",
            "cursor-snap",
            "cursor-snap-click",
            "cursor-snap-right-click",
            "cursor-snap-wink",
        ],
        default=action_default,
    )
    parser.add_argument("--action-cooldown-ms", type=int, default=900)
    parser.add_argument("--wink-closed-threshold", type=float, default=0.45)
    parser.add_argument("--wink-open-threshold", type=float, default=0.25)
    parser.add_argument("--drag-timeout-ms", type=int, default=3000)
    parser.add_argument("--dwell-ms", type=int, default=None)
    parser.add_argument("--dwell-min-confidence", type=float, default=None)
    parser.add_argument("--cursor-deadzone", type=float, default=None)
    parser.add_argument("--cursor-smoothing", type=float, default=None)
    parser.add_argument("--cursor-edge-padding", type=float, default=None)
    parser.add_argument("--cursor-max-step", type=float, default=None)
    parser.add_argument("--cursor-gain", type=float, default=None)
    parser.add_argument("--cursor-gain-x", type=float, default=None)
    parser.add_argument("--cursor-gain-y", type=float, default=None)
    parser.set_defaults(show_dot=show_dot_default, show_camera=show_camera_default, recalibrate_first=recalibrate_first_default)
    parser.add_argument("--show-dot", dest="show_dot", action="store_true")
    parser.add_argument("--hide-dot", dest="show_dot", action="store_false")
    parser.add_argument("--show-camera", dest="show_camera", action="store_true")
    parser.add_argument("--hide-camera", dest="show_camera", action="store_false")
    parser.add_argument("--recalibrate-first", dest="recalibrate_first", action="store_true")
    parser.add_argument("--skip-recalibration", dest="recalibrate_first", action="store_false")
    parser.add_argument("--calibration-frames-per-step", type=int, default=calibration_frames_per_step_default)
    parser.add_argument("--calibration-settle-ms", type=int, default=800)
    parser.add_argument(
        "--calibration-dot-size",
        type=int,
        default=calibration_dot_size_default if calibration_dot_size_default is not None else max(16, dot_size_default),
    )
    parser.add_argument("--dot-size", type=int, default=dot_size_default)
    parser.add_argument("--camera-preview-width", type=int, default=420)
    parser.add_argument("--runtime-dir", default=".interaction")
    parser.add_argument("--session-name", default=session_name_default)
    parser.add_argument("--execute", action="store_true")


def _add_interpreter_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interpreter",
        choices=[strategy.value for strategy in InterpreterStrategy],
        default=InterpreterStrategy.HYBRID.value,
    )


def _build_gemma_backend(_args: argparse.Namespace) -> object | None:
    return None


def _build_router(args: argparse.Namespace) -> VoiceInterpretationRouter:
    backend = _build_gemma_backend(args)
    return VoiceInterpretationRouter(
        strategy=InterpreterStrategy(args.interpreter),
        gemma_interpreter=GemmaIntentInterpreter(cast(Any, backend)),
    )


def _build_voice_loop(
    args: argparse.Namespace,
    *,
    effective_dry_run: bool,
    default_mode: SessionMode,
) -> VoiceCommandLoop:
    return VoiceCommandLoop(
        adapter=MacOSPlatformAdapter(dry_run=effective_dry_run),
        router=_build_router(args),
        default_mode=default_mode,
    )


def _build_fusion_loop(args: argparse.Namespace, *, effective_dry_run: bool) -> MultimodalFusionLoop:
    return MultimodalFusionLoop(
        adapter=MacOSPlatformAdapter(dry_run=effective_dry_run),
        router=_build_router(args),
    )


def _capture_runtime_environment(*, fallback_app: str, fallback_window: str | None = None) -> EnvironmentSnapshot:
    try:
        environment = capture_frontmost_environment()
    except Exception:
        environment = None
    else:
        if environment.active_app or environment.active_bundle_id or environment.active_window_title:
            if environment.active_window_title is None and fallback_window is not None:
                return environment.model_copy(update={"active_window_title": fallback_window})
            return environment
    return EnvironmentSnapshot(active_app=fallback_app, active_window_title=fallback_window)


def _run_voice_smoke(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    lane_mode, fallback_to_dictation = _voice_lane_settings(str(args.lane))
    loop = _build_voice_loop(args, effective_dry_run=effective_dry_run, default_mode=lane_mode)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    environment = _capture_runtime_environment(fallback_app="Finder", fallback_window="Voice Smoke")
    events = loop.run_text_turn(
        args.utterance,
        environment,
        mode=lane_mode,
        fallback_to_dictation=fallback_to_dictation,
    )
    if args.follow_up:
        events.extend(
            loop.run_text_turn(
                args.follow_up,
                environment,
                mode=lane_mode,
                fallback_to_dictation=fallback_to_dictation,
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
            "interpreter": args.interpreter,
            "environment": environment.model_dump(mode="json"),
            "voice_lane": str(args.lane),
            "voice_session_state": loop.state_snapshot(),
        },
    )


def _voice_lane_settings(lane: str) -> tuple[SessionMode, bool]:
    normalized_lane = lane.strip().lower()
    if normalized_lane == "dictation":
        return SessionMode.DICTATION, False
    if normalized_lane == "hybrid":
        return SessionMode.COMMAND, True
    return SessionMode.COMMAND, False


def _run_voice_live(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    provider = MacOSLiveSpeechProvider(store.paths)
    recording_path = _resolve_voice_recording_path(args, store.paths)
    lane_mode, fallback_to_dictation = _voice_lane_settings(str(args.lane))
    loop = _build_voice_loop(args, effective_dry_run=effective_dry_run, default_mode=lane_mode)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    fallback_window = "Live Dictation" if getattr(args, "command", None) == "voice-dictate" else "Live Voice"
    stream_environment = _capture_runtime_environment(fallback_app="Finder", fallback_window=fallback_window)
    events: list[object] = [
        VoiceFeedbackEvent(
            phase=VoiceLoopPhase.LISTENING,
            message=f"Starting live macOS speech capture for {args.duration:.1f}s.",
        )
    ]
    terminal_logger = _build_live_voice_terminal_logger()
    seen_stream_partials: set[str] = set()
    seen_stream_clauses: set[int] = set()
    stream_final_logged = False

    def emit_live_voice_event(payload: dict[str, object]) -> None:
        nonlocal stream_final_logged
        terminal_logger(payload)
        if not args.streaming or not isinstance(payload, dict):
            return
        transcript = str(payload.get("transcript", "") or "").strip()
        if not transcript:
            return
        event = str(payload.get("event", "") or "").strip().lower()
        status = str(payload.get("status", "") or "").strip().lower()
        if event == "partial":
            if transcript in seen_stream_partials:
                return
            seen_stream_partials.add(transcript)
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.LISTENING,
                    transcript=transcript,
                    message="Live streaming partial transcript update.",
                )
            )
            return
        if event == "clause":
            clause_index_value = payload.get("clause_index")
            clause_index = clause_index_value if isinstance(clause_index_value, int) else None
            if clause_index is not None and clause_index in seen_stream_clauses:
                return
            if clause_index is not None:
                seen_stream_clauses.add(clause_index)
            confidence_value = payload.get("confidence")
            final_confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else None
            display_index = clause_index + 1 if clause_index is not None else len(seen_stream_clauses) + 1
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.INTERPRETING,
                    transcript=transcript,
                    message=f"Detected streamed speech clause {display_index}.",
                )
            )
            events.extend(
                loop.run_text_turn(
                    transcript,
                    stream_environment,
                    final_confidence=final_confidence,
                    begin_turn=False,
                    mode=lane_mode,
                    fallback_to_dictation=fallback_to_dictation,
                )
            )
            return
        if event == "final" and status != "error" and not stream_final_logged:
            stream_final_logged = True
            events.append(
                VoiceFeedbackEvent(
                    phase=VoiceLoopPhase.INTERPRETING,
                    transcript=transcript,
                    message="Live macOS streaming speech capture completed.",
                )
            )

    provider.live_event_callback = emit_live_voice_event
    live_capture: dict[str, Any] | None = None
    try:
        if args.streaming:
            stream_turn_kwargs: dict[str, Any] = {
                "duration_s": args.duration,
                "locale": args.locale,
                "pause_ms": args.pause_ms,
            }
            if recording_path:
                stream_turn_kwargs["recording_path"] = recording_path
            stream_capture = provider.stream_turn(
                **stream_turn_kwargs,
            )
            capture = None
        else:
            capture_turn_kwargs: dict[str, Any] = {
                "duration_s": args.duration,
                "locale": args.locale,
            }
            if recording_path:
                capture_turn_kwargs["recording_path"] = recording_path
            capture = provider.capture_turn(**capture_turn_kwargs)
            stream_capture = None
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
        if stream_capture is not None:
            live_capture = {
                "status": "success",
                "mode": "streaming",
                "transcript": stream_capture.final_transcript,
                "partials": stream_capture.partials,
                "clauses": [
                    {
                        "transcript": clause.transcript,
                        "confidence": clause.confidence,
                        "clause_index": clause.clause_index,
                    }
                    for clause in stream_capture.clauses
                ],
                "locale": stream_capture.locale,
                "duration_s": stream_capture.duration_s,
                "provider": stream_capture.provider,
                "used_on_device": stream_capture.used_on_device,
                "permission_state": stream_capture.permission_state,
                "pause_ms": max(0, int(args.pause_ms)),
                "recording_path": stream_capture.recording_path,
            }
            for partial in stream_capture.partials:
                if partial in seen_stream_partials:
                    continue
                seen_stream_partials.add(partial)
                events.append(
                    VoiceFeedbackEvent(
                        phase=VoiceLoopPhase.LISTENING,
                        transcript=partial,
                        message="Live streaming partial transcript update.",
                    )
                )
            if not stream_final_logged:
                events.append(
                    VoiceFeedbackEvent(
                        phase=VoiceLoopPhase.INTERPRETING,
                        transcript=stream_capture.final_transcript,
                        message="Live macOS streaming speech capture completed.",
                    )
                )
            for clause in stream_capture.clauses:
                if clause.clause_index in seen_stream_clauses:
                    continue
                seen_stream_clauses.add(clause.clause_index)
                events.append(
                    VoiceFeedbackEvent(
                        phase=VoiceLoopPhase.INTERPRETING,
                        transcript=clause.transcript,
                        message=f"Detected streamed speech clause {clause.clause_index + 1}.",
                    )
                )
                events.extend(
                    loop.run_text_turn(
                        clause.transcript,
                        stream_environment,
                        final_confidence=clause.confidence,
                        begin_turn=False,
                        mode=lane_mode,
                        fallback_to_dictation=fallback_to_dictation,
                    )
                )
        else:
            assert capture is not None
            live_capture = {
                "status": "success",
                "mode": "one_shot",
                "transcript": capture.transcript,
                "confidence": capture.confidence,
                "locale": capture.locale,
                "duration_s": capture.duration_s,
                "provider": capture.provider,
                "used_on_device": capture.used_on_device,
                "permission_state": capture.permission_state,
                "recording_path": capture.recording_path,
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
                    stream_environment,
                    final_confidence=capture.confidence,
                    begin_turn=False,
                    mode=lane_mode,
                    fallback_to_dictation=fallback_to_dictation,
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
            "interpreter": args.interpreter,
            "environment": stream_environment.model_dump(mode="json"),
            "live_capture": live_capture,
            "voice_lane": str(args.lane),
            "voice_session_state": loop.state_snapshot(),
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
    provider: Any | None = None,
    keep_provider_open_on_success: bool = False,
) -> dict[str, Any]:
    provider = provider or _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    events: list[object] = []
    calibration_targets = generate_single_screen_calibration_targets(
        seed=args.random_seed,
        rows=max(3, int(getattr(args, "calibration_grid_rows", 3))),
    )
    interactive_calibration = _is_interactive_calibration_session()
    calibration_visualization = _calibration_visualization_payload(args, interactive=interactive_calibration)
    calibration_quality = None
    point_visualizer = None
    profile = None
    calibration: dict[str, Any] = {"status": "error"}

    if interactive_calibration:
        try:
            point_visualizer = CalibrationDotOverlay(size=_calibration_dot_size(args), fill="#0a84ff")
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
            after_step=lambda step_result, step_index, total_steps: _append_calibration_capture_feedback(
                events,
                step_result,
                step_index=step_index,
                total_steps=total_steps,
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
        fallback_axes = (
            list(calibration_quality.get("fallback_axes", []))
            if isinstance(calibration_quality, dict)
            else []
        )
        if fallback_axes:
            axis_label = ", ".join(str(axis).upper() for axis in fallback_axes)
            events.append(
                GazeFeedbackEvent(
                    phase=GazeLoopPhase.CALIBRATING,
                    message=f"Saved live webcam calibration profile with degraded fallback on the blocked {axis_label} axis.",
                )
            )
            calibration = {
                "status": "success",
                "degraded": True,
                "fallback_axes": fallback_axes,
            }
        else:
            events.append(GazeFeedbackEvent(phase=GazeLoopPhase.CALIBRATING, message="Saved live webcam calibration profile."))
            calibration = {"status": "success"}
        profile_payload = _profile_to_payload(profile)
        if keep_provider_open_on_success:
            _reset_webcam_provider_tracking_state(provider)
    finally:
        if not (calibration.get("status") == "success" and keep_provider_open_on_success):
            provider.close()
        if point_visualizer is not None:
            point_visualizer.close()
            time.sleep(0.12)

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


def _reset_webcam_provider_tracking_state(provider: Any) -> None:
    reset_tracking_state = getattr(provider, "reset_tracking_state", None)
    if callable(reset_tracking_state):
        reset_tracking_state()


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
    gaze_control_settings = _gaze_control_settings_payload(settings, args)
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=effective_dry_run),
        dwell_trigger=DwellTrigger(
            dwell_ms=gaze_control_settings["dwell_ms"],
            min_confidence=gaze_control_settings["dwell_min_confidence"],
            action=gaze_action,
            cooldown_ms=gaze_control_settings["action_cooldown_ms"],
        ),
        auto_confirm_actions=_gaze_auto_confirm_actions(gaze_action),
        drag_timeout_ms=gaze_control_settings["drag_timeout_ms"],
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
            "gaze_control_settings": gaze_control_settings,
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
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    visualization = _gaze_visualization_payload(args)
    calibration_events: list[object] = []
    startup_events: list[object] = []
    initial_calibration = None
    dry_run_warning = _gaze_live_dry_run_warning(effective_dry_run)
    provider = None
    tracker_capabilities = _tracker_capabilities_payload(args.tracker)
    mode_support = _gaze_mode_support_payload(gaze_mode, tracker_capabilities=tracker_capabilities)
    cursor_settings = _effective_cursor_settings(args, gaze_mode=gaze_mode)
    gaze_control_settings = _gaze_control_settings_payload(settings, args)

    if not mode_support["tracker_supported"]:
        error = _unsupported_tracker_mode_error(tracker=args.tracker, gaze_mode=gaze_mode, mode_support=mode_support)
        events = [
            GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error["message"]),
            GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."),
        ]
        if startup_events:
            events = [*startup_events, *events]
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
                "gaze_mode_support": mode_support,
                "tracker_capabilities": tracker_capabilities,
                "cursor_settings": cursor_settings,
                "gaze_control_settings": gaze_control_settings,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "live_gaze": {"status": "error", **error},
                "initial_calibration": initial_calibration,
                "gaze_visualization": visualization,
                "calibration_profile": None,
                "warnings": [dry_run_warning] if dry_run_warning is not None else [],
            },
        )

    if not mode_support["runtime_supported"]:
        error = _unsupported_gaze_mode_error(gaze_mode=gaze_mode, mode_support=mode_support)
        events = [
            GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error["message"]),
            GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."),
        ]
        if startup_events:
            events = [*startup_events, *events]
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
                "gaze_mode_support": mode_support,
                "tracker_capabilities": tracker_capabilities,
                "cursor_settings": cursor_settings,
                "gaze_control_settings": gaze_control_settings,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "live_gaze": {"status": "error", **error},
                "initial_calibration": initial_calibration,
                "gaze_visualization": visualization,
                "calibration_profile": None,
                "warnings": [dry_run_warning] if dry_run_warning is not None else [],
            },
        )

    if dry_run_warning is not None:
        startup_events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message=dry_run_warning,
            )
        )

    if args.recalibrate_first:
        provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
        calibration_stage = _perform_live_gaze_calibration(
            args,
            store,
            camera_index=camera_index,
            tracker_model_path=tracker_model_path,
            frames_per_step=max(1, int(args.calibration_frames_per_step)),
            settle_ms=max(0, int(args.calibration_settle_ms)),
            provider=provider,
            keep_provider_open_on_success=True,
        )
        initial_calibration = {
            "calibration": calibration_stage["calibration"],
            "calibration_steps": calibration_stage["calibration_steps"],
            "calibration_targets": calibration_stage["calibration_targets"],
            "calibration_visualization": calibration_stage["calibration_visualization"],
            "calibration_quality": calibration_stage["calibration_quality"],
            "calibration_profile": calibration_stage["calibration_profile"],
        }
        calibration_events = list(calibration_stage["events"])
        if calibration_stage["calibration"]["status"] != "success":
            return _build_payload(
                source="gaze",
                events=calibration_events,
                logger=logger,
                overlay=overlay,
                renderer=renderer,
                extra={
                    "settings": _settings_payload(settings, effective_dry_run),
                    "gaze_action": gaze_action.value,
                    "gaze_mode": gaze_mode,
                    "gaze_mode_support": mode_support,
                    "tracker_capabilities": tracker_capabilities,
                    "cursor_settings": cursor_settings,
                    "gaze_control_settings": gaze_control_settings,
                    "camera_index": camera_index,
                    "tracker": args.tracker,
                    "tracker_model_path": tracker_model_path,
                    "live_gaze": {
                        "status": "error",
                        "error": "initial_calibration_failed",
                        "message": initial_calibration["calibration"]["message"],
                        "payload": initial_calibration["calibration"].get("payload", {}),
                    },
                    "initial_calibration": initial_calibration,
                    "gaze_visualization": visualization,
                    "calibration_profile": None,
                },
            )
        if calibration_events and getattr(calibration_events[-1], "phase", None) == GazeLoopPhase.IDLE:
            calibration_events.pop()
        profile = calibration_stage["profile"]
        invalid_calibration = None
    else:
        invalid_calibration = _load_live_calibration_invalid_marker(store.paths)
        profile = store.load_calibration_profile("webcam-live")

    if mode_support["requires_calibration_profile"] and profile is None:
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
                "gaze_mode_support": mode_support,
                "tracker_capabilities": tracker_capabilities,
                "cursor_settings": cursor_settings,
                "gaze_control_settings": gaze_control_settings,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "live_gaze": {
                    "status": "error",
                    "error": "missing_calibration_profile",
                    "message": "No live webcam calibration profile is saved yet. Run `interaction gaze-calibrate` first.",
                },
                "initial_calibration": initial_calibration,
                "gaze_visualization": visualization,
                "calibration_profile": None,
            },
        )

    if mode_support["requires_calibration_profile"] and invalid_calibration is not None:
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
                "gaze_mode_support": mode_support,
                "tracker_capabilities": tracker_capabilities,
                "cursor_settings": cursor_settings,
                "gaze_control_settings": gaze_control_settings,
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
                "initial_calibration": initial_calibration,
                "gaze_visualization": visualization,
                "calibration_profile": None,
            },
        )

    if provider is None:
        provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    point_visualizer = None
    camera_preview = None
    initial_cursor_point = _current_cursor_point() if gaze_mode in {"cursor", "cursor-relative", "cursor-relative-click", "cursor-assist", "cursor-assist-click"} else None
    if args.show_dot:
        try:
            if gaze_mode in {"cursor-snap", "cursor-snap-click", "cursor-snap-right-click", "cursor-snap-wink"}:
                point_visualizer = LiveSnapDotsOverlay(size=max(12, args.dot_size))
            else:
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
        dwell_trigger=DwellTrigger(
            dwell_ms=gaze_control_settings["dwell_ms"],
            min_confidence=gaze_control_settings["dwell_min_confidence"],
            action=gaze_action,
            cooldown_ms=gaze_control_settings["action_cooldown_ms"],
        ),
        auto_confirm_actions=_gaze_auto_confirm_actions(gaze_action),
        drag_timeout_ms=gaze_control_settings["drag_timeout_ms"],
    )
    runtime_profile = profile if profile is not None else CalibrationProfile()
    loop.calibration_profile = runtime_profile
    live_event_sink = _build_live_gaze_terminal_logger() if gaze_mode == "cursor-snap-wink" else None
    try:
        provider.open()
        if gaze_mode == "cursor":
            events, summary = run_live_cursor_follow(
                provider,
                runtime_profile,
                adapter=loop.adapter,
                frames=args.frames,
                delta_ms=args.delta_ms,
                environment=EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera"),
                targets=default_webcam_targets(),
                min_move_distance=cursor_settings["deadzone"],
                smoothing=cursor_settings["smoothing"],
                edge_padding=cursor_settings["edge_padding"],
                max_step=cursor_settings["max_step"],
                response_gain_x=cursor_settings["gain_x"],
                response_gain_y=cursor_settings["gain_y"],
                point_visualizer=point_visualizer,
                initial_cursor_point=initial_cursor_point,
                camera_preview=camera_preview,
            )
        elif gaze_mode in {"cursor-relative", "cursor-relative-click"}:
            events, summary = run_live_relative_cursor_follow(
                provider,
                runtime_profile,
                adapter=loop.adapter,
                frames=args.frames,
                delta_ms=args.delta_ms,
                environment=EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera"),
                min_move_distance=cursor_settings["deadzone"],
                edge_padding=cursor_settings["edge_padding"],
                max_step=cursor_settings["max_step"],
                response_gain_x=cursor_settings["gain_x"],
                response_gain_y=cursor_settings["gain_y"],
                neutral_frames=6,
                click_on_dwell=(gaze_mode == "cursor-relative-click"),
                dwell_ms=gaze_control_settings["dwell_ms"],
                click_cooldown_ms=gaze_control_settings["action_cooldown_ms"],
                point_visualizer=point_visualizer,
                initial_cursor_point=initial_cursor_point,
                camera_preview=camera_preview,
            )
        elif gaze_mode in {"cursor-assist", "cursor-assist-click"}:
            events, summary = run_live_assisted_cursor_follow(
                provider,
                runtime_profile,
                adapter=loop.adapter,
                frames=args.frames,
                delta_ms=args.delta_ms,
                environment=EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera"),
                min_move_distance=cursor_settings["deadzone"],
                edge_padding=cursor_settings["edge_padding"],
                max_step=cursor_settings["max_step"],
                response_gain_x=cursor_settings["gain_x"],
                response_gain_y=cursor_settings["gain_y"],
                click_on_dwell=(gaze_mode == "cursor-assist-click"),
                dwell_ms=gaze_control_settings["dwell_ms"],
                click_cooldown_ms=gaze_control_settings["action_cooldown_ms"],
                point_visualizer=point_visualizer,
                initial_cursor_point=initial_cursor_point,
                camera_preview=camera_preview,
            )
        elif gaze_mode in {"cursor-snap", "cursor-snap-click", "cursor-snap-right-click", "cursor-snap-wink"}:
            events, summary = run_live_cursor_snap(
                provider,
                runtime_profile,
                adapter=loop.adapter,
                frames=args.frames,
                delta_ms=args.delta_ms,
                environment=EnvironmentSnapshot(active_app="Webcam Live", active_window_title="Live Camera"),
                dwell_ms=gaze_control_settings["dwell_ms"],
                second_dwell_action=(
                    ActionName.CLICK_TARGET
                    if gaze_mode == "cursor-snap-click"
                    else ActionName.RIGHT_CLICK_TARGET if gaze_mode == "cursor-snap-right-click" else None
                ),
                wink_confirm=(gaze_mode == "cursor-snap-wink"),
                wink_closed_threshold=max(0.0, min(1.0, float(args.wink_closed_threshold))),
                wink_open_threshold=max(0.0, min(1.0, float(args.wink_open_threshold))),
                cooldown_ms=max(0, args.action_cooldown_ms),
                point_visualizer=point_visualizer,
                camera_preview=camera_preview,
                event_sink=live_event_sink,
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
    if calibration_events:
        events = [*calibration_events, *events]
    if startup_events:
        events = [*startup_events, *events]
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
            "gaze_mode_support": mode_support,
            "camera_index": camera_index,
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "tracker_capabilities": tracker_capabilities,
            "live_gaze": live_gaze,
            "initial_calibration": initial_calibration,
            "initial_cursor_point": (
                {"x": initial_cursor_point[0], "y": initial_cursor_point[1]}
                if initial_cursor_point is not None
                else None
            ),
            "cursor_settings": cursor_settings,
            "gaze_control_settings": gaze_control_settings,
            "gaze_visualization": visualization,
            "calibration_profile": _profile_to_payload(profile) if mode_support["requires_calibration_profile"] else None,
            "warnings": [dry_run_warning] if dry_run_warning is not None else [],
        },
    )


def _run_gaze_wink_test(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    tracker_model_path = _resolve_tracker_model_path(args, store.paths)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    visualization = _gaze_visualization_payload(args)
    dry_run_warning = _gaze_live_dry_run_warning(effective_dry_run)
    tracker_capabilities = _tracker_capabilities_payload(args.tracker)
    mode_support = _gaze_mode_support_payload("wink-click-test", tracker_capabilities=tracker_capabilities)

    if not mode_support["tracker_supported"]:
        error = _unsupported_tracker_mode_error(
            tracker=args.tracker,
            gaze_mode="wink-click-test",
            mode_support=mode_support,
        )
        events = [
            GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error["message"]),
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
                "gaze_mode": "wink-click-test",
                "gaze_action": ActionName.CLICK_TARGET.value,
                "gaze_mode_support": mode_support,
                "tracker_capabilities": tracker_capabilities,
                "camera_index": camera_index,
                "tracker": args.tracker,
                "tracker_model_path": tracker_model_path,
                "wink_test": {"status": "error", **error},
                "gaze_visualization": visualization,
                "warnings": [dry_run_warning] if dry_run_warning is not None else [],
            },
        )

    events: list[object] = []
    if dry_run_warning is not None:
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.TRACKING,
                message=dry_run_warning,
            )
        )

    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    live_event_sink = _build_live_gaze_terminal_logger()
    point_visualizer = None
    camera_preview = None
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

    adapter = MacOSPlatformAdapter(dry_run=effective_dry_run)
    try:
        provider.open()
        runtime_events, summary = run_live_wink_click_test(
            provider,
            adapter=adapter,
            frames=args.frames,
            delta_ms=args.delta_ms,
            environment=EnvironmentSnapshot(active_app="Wink Click Test", active_window_title="Wink Click Test"),
            min_confidence=_effective_dwell_min_confidence(args),
            wink_hold_ms=max(0, int(args.wink_hold_ms)),
            wink_closed_threshold=max(0.0, min(1.0, float(args.wink_closed_threshold))),
            wink_open_threshold=max(0.0, min(1.0, float(args.wink_open_threshold))),
            cooldown_ms=max(0, int(args.action_cooldown_ms)),
            stop_after_clicks=max(0, int(args.stop_after_clicks)),
            point_visualizer=point_visualizer,
            camera_preview=camera_preview,
            event_sink=live_event_sink,
        )
        events.extend(runtime_events)
        wink_test = {"status": "success", **summary}
    except WebcamProviderError as error:
        events.extend(
            [
                GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message),
                GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."),
            ]
        )
        wink_test = {
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
            "gaze_mode": "wink-click-test",
            "gaze_action": ActionName.CLICK_TARGET.value,
            "gaze_mode_support": mode_support,
            "tracker_capabilities": tracker_capabilities,
            "camera_index": camera_index,
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "wink_test": wink_test,
            "gaze_visualization": visualization,
            "warnings": [dry_run_warning] if dry_run_warning is not None else [],
        },
    )


def _run_gaze_benchmark(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    camera_index = settings.camera_index if args.camera_index is None else args.camera_index
    tracker_model_path = _resolve_tracker_model_path(args, store.paths)
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    visualization = _gaze_visualization_payload(args)
    tracker_capabilities = _tracker_capabilities_payload(args.tracker)
    mode_support = _gaze_mode_support_payload("benchmark", tracker_capabilities=tracker_capabilities)

    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    point_visualizer = None
    camera_preview = None
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

    try:
        provider.open()
        events, summary = run_live_webcam_benchmark(
            provider,
            frames=args.frames,
            delta_ms=args.delta_ms,
            wink_closed_threshold=max(0.0, min(1.0, float(args.wink_closed_threshold))),
            wink_open_threshold=max(0.0, min(1.0, float(args.wink_open_threshold))),
            point_visualizer=point_visualizer,
            camera_preview=camera_preview,
        )
        benchmark = {"status": "success", **summary}
    except WebcamProviderError as error:
        events = [
            GazeFeedbackEvent(phase=GazeLoopPhase.RECOVERING, message=error.message),
            GazeFeedbackEvent(phase=GazeLoopPhase.IDLE, message="Gaze loop is idle."),
        ]
        benchmark = {
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

    payload = _build_payload(
        source="gaze",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run=settings.dry_run),
            "gaze_mode": "benchmark",
            "gaze_action": ActionName.HIGHLIGHT_TARGET.value,
            "gaze_mode_support": mode_support,
            "tracker_capabilities": tracker_capabilities,
            "camera_index": camera_index,
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "benchmark": benchmark,
            "gaze_visualization": visualization,
        },
    )
    summary_path = _benchmark_summary_path_for_log(logger.path)
    _write_benchmark_summary(
        summary_path,
        {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_log": str(logger.path),
            "summary_path": str(summary_path),
            "camera_index": camera_index,
            "tracker": args.tracker,
            "tracker_model_path": tracker_model_path,
            "benchmark": benchmark,
        },
    )
    payload["summary_path"] = str(summary_path)
    return payload


def _run_gaze_benchmark_report(args: argparse.Namespace) -> dict[str, Any]:
    runtime_paths = RuntimePaths(Path(args.runtime_dir))
    rendering = _benchmark_report_rendering_payload(args)
    summary_paths = _resolve_benchmark_summary_paths(
        runtime_paths=runtime_paths,
        explicit_paths=args.summary_path,
        tracker=args.tracker,
        limit=max(0, int(args.limit)),
    )
    if not summary_paths:
        return {
            "runtime_dir": str(runtime_paths.base_dir),
            "summary_paths": [],
            "benchmark_report": {
                "status": "error",
                "error": "no_benchmark_summaries_found",
                "message": "No gaze benchmark summary files were found for the requested filter.",
                "rendering": rendering,
            },
        }

    summaries: list[dict[str, Any]] = []
    skipped_paths: list[dict[str, str]] = []
    for path in summary_paths:
        try:
            summaries.append(_load_benchmark_summary(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            skipped_paths.append({"summary_path": str(path), "error": str(error)})

    if not summaries:
        return {
            "runtime_dir": str(runtime_paths.base_dir),
            "summary_paths": [str(path) for path in summary_paths],
            "benchmark_report": {
                "status": "error",
                "error": "benchmark_summary_parse_failed",
                "message": "Benchmark summary files were found, but none could be parsed successfully.",
                "skipped_paths": skipped_paths,
                "rendering": rendering,
            },
        }

    runs = [_condense_benchmark_summary(summary) for summary in summaries]
    successful_runs = [run for run in runs if run["status"] == "success"]
    latest = runs[-1]
    previous = runs[-2] if len(runs) > 1 else None
    latest_successful = successful_runs[-1] if successful_runs else None
    previous_successful = successful_runs[-2] if len(successful_runs) > 1 else None
    aggregate = _aggregate_benchmark_runs(successful_runs)
    quality_summary = _benchmark_quality_summary(successful_runs)
    trend = _benchmark_trend_summary(successful_runs)
    notes = _benchmark_report_notes(
        aggregate=aggregate,
        latest_successful=latest_successful,
        previous_successful=previous_successful,
    )
    best_runs = _select_best_benchmark_runs(successful_runs)
    recommendations = _benchmark_report_recommendations(
        aggregate=aggregate,
        latest_successful=latest_successful,
        trend=trend,
    )
    dominant_bottleneck = _benchmark_dominant_bottleneck(
        aggregate=aggregate,
        latest_successful=latest_successful,
        trend=trend,
    )
    groups_by_tracker = _group_benchmark_runs(successful_runs, group_by="tracker")
    groups_by_session_prefix = _group_benchmark_runs(successful_runs, group_by="session_prefix")
    group_comparisons = {
        "tracker": _benchmark_group_comparison(groups_by_tracker, group_by="tracker"),
        "session_prefix": _benchmark_group_comparison(groups_by_session_prefix, group_by="session_prefix"),
    }
    group_metric_hotspots = {
        "tracker": _benchmark_group_metric_hotspots(groups_by_tracker, group_by="tracker"),
        "session_prefix": _benchmark_group_metric_hotspots(groups_by_session_prefix, group_by="session_prefix"),
    }
    metric_leaderboards = _benchmark_metric_leaderboards(successful_runs)
    recent_changes = _benchmark_recent_changes(successful_runs)
    return {
        "runtime_dir": str(runtime_paths.base_dir),
        "summary_paths": [str(path) for path in summary_paths],
        "benchmark_report": {
            "status": "success",
            "rendering": rendering,
            "run_count": len(runs),
            "successful_run_count": len(successful_runs),
            "tracker_filter": args.tracker,
            "latest": latest,
            "previous": previous,
            "latest_successful": latest_successful,
            "previous_successful": previous_successful,
            "delta_from_previous_successful": (
                _benchmark_run_delta(previous_successful, latest_successful)
                if previous_successful is not None and latest_successful is not None
                else None
            ),
            "aggregate": aggregate,
            "quality_summary": quality_summary,
            "trend": trend,
            "best_runs": best_runs,
            "notes": notes,
            "dominant_bottleneck": dominant_bottleneck,
            "recommendations": recommendations,
            "groups_by_tracker": groups_by_tracker,
            "groups_by_session_prefix": groups_by_session_prefix,
            "group_comparisons": group_comparisons,
            "group_metric_hotspots": group_metric_hotspots,
            "metric_leaderboards": metric_leaderboards,
            "recent_changes": recent_changes,
            "runs": runs,
            "skipped_paths": skipped_paths,
        },
    }


def _benchmark_report_rendering_payload(args: argparse.Namespace) -> dict[str, Any]:
    preset = getattr(args, "preset", None)
    output_format = str(getattr(args, "format", "json"))
    brief = bool(getattr(args, "brief", False))
    if preset == "json":
        output_format = "json"
        brief = False
    elif preset == "full-text":
        output_format = "text"
        brief = False
    elif preset == "brief-text":
        output_format = "text"
        brief = True
    return {
        "preset": preset,
        "format": output_format,
        "brief": brief,
        "top": _render_benchmark_report_top_limit(getattr(args, "top", None)),
        "output": str(getattr(args, "output", "") or "") or None,
    }


def _collect_fusion_candidate_targets(
    *,
    adapter: MacOSPlatformAdapter,
    observation: GazeObservation,
    environment: EnvironmentSnapshot,
    limit: int = 5,
) -> tuple[list[GroundedTarget], list[dict[str, Any]]]:
    if observation.x_norm is None or observation.y_norm is None:
        return [], []
    capabilities = adapter.describe_capabilities()
    if not capabilities.candidate_collection:
        return [], []
    try:
        screen_point = normalized_to_screen_point(observation.x_norm, observation.y_norm)
        candidates = adapter.collect_target_candidates(
            TargetCandidateQuery(
                screen_point=screen_point,
                limit=max(1, limit),
                active_app=environment.active_app,
                active_window_title=environment.active_window_title,
            )
        )
    except Exception:
        return [], []
    grounded = [_candidate_to_grounded_target(candidate) for candidate in candidates]
    grounded = [candidate for candidate in grounded if candidate is not None]
    return grounded, [candidate.model_dump(mode="json") for candidate in candidates]


def _candidate_to_grounded_target(candidate: DesktopTargetCandidate) -> GroundedTarget | None:
    label = candidate.display_label or candidate.label or candidate.role or "desktop target"
    role = candidate.role or "ui_element"
    bounds = None
    if candidate.normalized_bounds is not None:
        normalized_bounds = candidate.normalized_bounds
        bounds = BoundingBox(
            x=normalized_bounds.x,
            y=normalized_bounds.y,
            width=normalized_bounds.width,
            height=normalized_bounds.height,
        )
    try:
        return GroundedTarget(
            target_id=candidate.candidate_id,
            label=label,
            role=role,
            confidence=0.92 if candidate.invocable is not False else 0.82,
            screen_region=None,
            bounds=bounds,
            source=candidate.source,
            candidate=candidate,
        )
    except Exception:
        return None


def _run_fusion_smoke(args: argparse.Namespace, store: JsonStateStore) -> dict[str, Any]:
    settings = store.load_settings()
    effective_dry_run = _effective_dry_run(settings, args)
    loop = _build_fusion_loop(args, effective_dry_run=effective_dry_run)
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
    environment = _capture_runtime_environment(fallback_app="Mail", fallback_window="Inbox")
    events: list[object] = []
    events.append(loop.update_gaze_context(GazeObservation(confidence=0.88, x_norm=0.56, y_norm=0.20, fixation_ms=250), target))
    events.extend(loop.run_voice_turn(args.utterance, environment))
    if args.follow_up:
        events.extend(loop.run_voice_turn(args.follow_up, environment))
    return _build_payload(
        source="fusion",
        events=events,
        logger=logger,
        overlay=overlay,
        renderer=renderer,
        extra={
            "settings": _settings_payload(settings, effective_dry_run),
            "interpreter": args.interpreter,
            "environment": environment.model_dump(mode="json"),
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
    loop = _build_fusion_loop(args, effective_dry_run=effective_dry_run)
    loop.state.gaze_context_window_ms = settings.gaze_context_window_ms
    logger = SessionLogger(store.paths.next_session_log_path(args.session_name))
    overlay = OverlayController()
    renderer = ConsoleOverlayRenderer()
    environment = _capture_runtime_environment(fallback_app="Finder", fallback_window="Live Fusion")

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
                "interpreter": args.interpreter,
                "environment": environment.model_dump(mode="json"),
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
                "interpreter": args.interpreter,
                "environment": environment.model_dump(mode="json"),
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

    events = [
        FusionFeedbackEvent(
            phase=FusionLoopPhase.TRACKING,
            message=f"Capturing live gaze context from webcam for {args.gaze_frames} frames.",
        )
    ]
    provider = _build_webcam_provider(args.tracker, camera_index, tracker_model_path=tracker_model_path)
    live_gaze_context: dict[str, Any]
    explicit_candidate_targets: list[GroundedTarget] = []
    candidate_payload: list[dict[str, Any]] = []
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
                "interpreter": args.interpreter,
                "environment": environment.model_dump(mode="json"),
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
                "interpreter": args.interpreter,
                "environment": environment.model_dump(mode="json"),
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

    explicit_candidate_targets, candidate_payload = _collect_fusion_candidate_targets(
        adapter=loop.adapter,
        observation=gaze_context.observation,
        environment=environment,
    )
    live_gaze_context = {
        "status": "success",
        "target": gaze_context.target.model_dump(mode="json"),
        "selected_target": (
            explicit_candidate_targets[0].model_dump(mode="json")
            if explicit_candidate_targets
            else gaze_context.target.model_dump(mode="json")
        ),
        "candidate_targets": candidate_payload,
        "observation": gaze_context.observation.model_dump(mode="json"),
        "attempted_frames": gaze_context.attempted_frames,
        "successful_readings": gaze_context.successful_readings,
        "missing_readings": gaze_context.missing_readings,
        "grounded_readings": gaze_context.grounded_readings,
        "dominant_target_hits": gaze_context.dominant_target_hits,
        "explicit_candidate_count": len(explicit_candidate_targets),
    }
    events.append(
        loop.update_gaze_context(
            gaze_context.observation,
            gaze_context.target,
            candidate_targets=explicit_candidate_targets,
        )
    )

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
                "interpreter": args.interpreter,
                "environment": environment.model_dump(mode="json"),
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
                "interpreter": args.interpreter,
                "environment": environment.model_dump(mode="json"),
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
            "interpreter": args.interpreter,
            "environment": environment.model_dump(mode="json"),
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


def _build_live_gaze_terminal_logger() -> Any:
    last_signal: tuple[str, str] | None = None

    def emit(event: object) -> None:
        nonlocal last_signal
        message = str(getattr(event, "message", "") or "")
        signal: tuple[str, str] | None = None
        terminal_message: str | None = None
        if message.startswith("Left wink detected"):
            signal = ("detected", "left")
            terminal_message = "WINK DETECTED: LEFT"
        elif message.startswith("Right wink detected"):
            signal = ("detected", "right")
            terminal_message = "WINK DETECTED: RIGHT"
        elif message.startswith("Left wink candidate"):
            signal = ("candidate", "left")
            terminal_message = "WINK CANDIDATE: LEFT"
        elif message.startswith("Right wink candidate"):
            signal = ("candidate", "right")
            terminal_message = "WINK CANDIDATE: RIGHT"
        elif "with a left wink" in message or "left wink and issued a left click" in message:
            signal = ("triggered", "left")
            terminal_message = "WINK TRIGGERED: LEFT"
        elif "with a right wink" in message or "right wink and issued a right click" in message:
            signal = ("triggered", "right")
            terminal_message = "WINK TRIGGERED: RIGHT"
        elif message.startswith("Snapped to ") and "Left wink to click" in message:
            signal = ("armed", "snap")
            terminal_message = "TARGET ARMED: wink to confirm"
        elif "looked away" in message and "Snap preview is active again" in message:
            signal = ("canceled", "snap")
            terminal_message = "TARGET CANCELED"
        elif message.startswith("Watching for deliberate left and right winks") or (
            message.startswith("Previewing armed target") and "Left wink to click" in message
        ) or message == "Gaze loop is idle.":
            last_signal = None
            return
        else:
            return
        if signal != last_signal and terminal_message is not None:
            print(f"[gaze] {terminal_message}", file=sys.stderr, flush=True)
        last_signal = signal

    return emit


def _build_live_voice_terminal_logger() -> Any:
    last_message: str | None = None
    last_audio_message: str | None = None

    def emit(payload: dict[str, object]) -> None:
        nonlocal last_message, last_audio_message
        if not isinstance(payload, dict):
            return
        transcript = str(payload.get("transcript", "") or "").strip()
        event = str(payload.get("event", "") or "").strip().lower()
        status = str(payload.get("status", "") or "").strip().lower()
        message = str(payload.get("message", "") or "").strip()
        audio_peak_dbfs = payload.get("audio_peak_dbfs")
        audio_rms_dbfs = payload.get("audio_rms_dbfs")
        audio_signal_detected = payload.get("audio_signal_detected")
        if (
            event == "final"
            and isinstance(audio_peak_dbfs, (int, float))
            and isinstance(audio_rms_dbfs, (int, float))
        ):
            state = "audio detected" if bool(audio_signal_detected) else "no usable audio detected"
            audio_message = f"mic: {state} (peak {float(audio_peak_dbfs):.1f} dBFS, rms {float(audio_rms_dbfs):.1f} dBFS)"
            if audio_message != last_audio_message:
                print(f"[voice] {audio_message}", file=sys.stderr, flush=True)
                last_audio_message = audio_message
        terminal_message: str
        if transcript and event == "partial":
            terminal_message = f"partial: {transcript}"
        elif transcript and event == "clause":
            clause_index = payload.get("clause_index")
            if isinstance(clause_index, int):
                terminal_message = f"clause {clause_index + 1}: {transcript}"
            else:
                terminal_message = f"clause: {transcript}"
        elif transcript:
            terminal_message = f"final: {transcript}"
        elif status == "error" and message:
            terminal_message = f"error: {message}"
        else:
            return
        if terminal_message == last_message:
            return
        print(f"[voice] {terminal_message}", file=sys.stderr, flush=True)
        last_message = terminal_message

    return emit


def _resolve_voice_recording_path(args: argparse.Namespace, runtime_paths: RuntimePaths) -> str | None:
    explicit_path = str(getattr(args, "audio_path", "") or "").strip()
    if explicit_path:
        return str(Path(explicit_path).expanduser())
    if not bool(getattr(args, "save_audio", False)):
        return None
    runtime_paths.ensure()
    stem = re.sub(r"[^a-z0-9]+", "-", str(getattr(args, "session_name", "voice-live")).lower()).strip("-") or "voice"
    return str(runtime_paths.logs_dir / f"{stem}-audio.caf")


def _benchmark_summary_path_for_log(log_path: Path) -> Path:
    return log_path.with_suffix(".summary.json")


def _write_benchmark_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _resolve_benchmark_summary_paths(
    *,
    runtime_paths: RuntimePaths,
    explicit_paths: list[str] | None,
    tracker: str | None,
    limit: int,
) -> list[Path]:
    if explicit_paths:
        candidates = [Path(value) for value in explicit_paths]
    else:
        candidates = sorted(
            runtime_paths.logs_dir.glob("gaze-benchmark*.summary.json"),
            key=lambda path: path.stat().st_mtime,
        )
    filtered: list[Path] = []
    for path in candidates:
        if tracker is None:
            filtered.append(path)
            continue
        try:
            payload = _load_benchmark_summary(path)
        except (OSError, ValueError, json.JSONDecodeError):
            filtered.append(path)
            continue
        if payload.get("tracker") == tracker:
            filtered.append(path)
    if limit > 0:
        return filtered[-limit:]
    return filtered


def _load_benchmark_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark summary at {path} is not a JSON object.")
    return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _condense_benchmark_summary(payload: dict[str, Any]) -> dict[str, Any]:
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        benchmark = {}
    attempted_frames = _as_int(benchmark.get("attempted_frames"))
    successful_readings = _as_int(benchmark.get("successful_readings"))
    missing_readings = _as_int(benchmark.get("missing_readings"))
    fused_jitter_x = _as_float(benchmark.get("fused_jitter_x"))
    fused_jitter_y = _as_float(benchmark.get("fused_jitter_y"))
    head_yaw_abs_mean = _as_float(benchmark.get("head_yaw_abs_mean"))
    head_pitch_abs_mean = _as_float(benchmark.get("head_pitch_abs_mean"))
    head_roll_abs_mean = _as_float(benchmark.get("head_roll_abs_mean"))
    condensed = {
        "created_at": payload.get("created_at"),
        "summary_path": payload.get("summary_path"),
        "session_log": payload.get("session_log"),
        "tracker": payload.get("tracker"),
        "status": benchmark.get("status", "unknown"),
        "attempted_frames": attempted_frames,
        "successful_readings": successful_readings,
        "missing_readings": missing_readings,
        "dropout_rate": (missing_readings / attempted_frames) if attempted_frames > 0 else 0.0,
        "sample_rate_hz": _as_float(benchmark.get("sample_rate_hz")),
        "mean_confidence": _as_float(benchmark.get("mean_confidence")),
        "fused_jitter_x": fused_jitter_x,
        "fused_jitter_y": fused_jitter_y,
        "fused_jitter_distance": (fused_jitter_x**2 + fused_jitter_y**2) ** 0.5,
        "fused_drift_distance": _as_float(benchmark.get("fused_drift_distance")),
        "raw_fused_mean_distance": _as_float(benchmark.get("raw_fused_mean_distance")),
        "x_axis_confidence_mean": _as_float(benchmark.get("x_axis_confidence_mean")),
        "y_axis_confidence_mean": _as_float(benchmark.get("y_axis_confidence_mean")),
        "head_motion_mean_magnitude": (head_yaw_abs_mean**2 + head_pitch_abs_mean**2 + head_roll_abs_mean**2) ** 0.5,
        "temporal_fusion_active_count": _as_int(benchmark.get("temporal_fusion_active_count")),
        "fusion_wink_freeze_count": _as_int(benchmark.get("fusion_wink_freeze_count")),
        "left_wink_episodes": _as_int(benchmark.get("left_wink_episodes")),
        "right_wink_episodes": _as_int(benchmark.get("right_wink_episodes")),
        "wink_signal_available": bool(benchmark.get("wink_signal_available", False)),
        "error": benchmark.get("error"),
        "message": benchmark.get("message"),
    }
    quality = _benchmark_quality_score(condensed)
    condensed["quality_score"] = quality["score"]
    condensed["quality_grade"] = quality["grade"]
    condensed["quality_components"] = quality["components"]
    return condensed


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _aggregate_benchmark_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    tracker_counts: dict[str, int] = {}
    for run in runs:
        tracker = str(run.get("tracker") or "unknown")
        tracker_counts[tracker] = tracker_counts.get(tracker, 0) + 1
    aggregate = {
        "tracker_counts": tracker_counts,
        "mean_dropout_rate": _mean([_as_float(run.get("dropout_rate")) for run in runs]),
        "mean_sample_rate_hz": _mean([_as_float(run.get("sample_rate_hz")) for run in runs]),
        "mean_confidence": _mean([_as_float(run.get("mean_confidence")) for run in runs]),
        "mean_fused_jitter_x": _mean([_as_float(run.get("fused_jitter_x")) for run in runs]),
        "mean_fused_jitter_y": _mean([_as_float(run.get("fused_jitter_y")) for run in runs]),
        "mean_fused_jitter_distance": _mean([_as_float(run.get("fused_jitter_distance")) for run in runs]),
        "mean_fused_drift_distance": _mean([_as_float(run.get("fused_drift_distance")) for run in runs]),
        "mean_raw_fused_mean_distance": _mean([_as_float(run.get("raw_fused_mean_distance")) for run in runs]),
        "mean_x_axis_confidence": _mean([_as_float(run.get("x_axis_confidence_mean")) for run in runs]),
        "mean_y_axis_confidence": _mean([_as_float(run.get("y_axis_confidence_mean")) for run in runs]),
        "mean_head_motion_magnitude": _mean([_as_float(run.get("head_motion_mean_magnitude")) for run in runs]),
        "total_left_wink_episodes": sum(_as_int(run.get("left_wink_episodes")) for run in runs),
        "total_right_wink_episodes": sum(_as_int(run.get("right_wink_episodes")) for run in runs),
    }
    quality = _benchmark_quality_score(
        {
            "dropout_rate": aggregate["mean_dropout_rate"],
            "mean_confidence": aggregate["mean_confidence"],
            "fused_jitter_distance": aggregate["mean_fused_jitter_distance"],
            "fused_drift_distance": aggregate["mean_fused_drift_distance"],
            "raw_fused_mean_distance": aggregate["mean_raw_fused_mean_distance"],
            "x_axis_confidence_mean": aggregate["mean_x_axis_confidence"],
            "y_axis_confidence_mean": aggregate["mean_y_axis_confidence"],
        }
    )
    aggregate["mean_quality_score"] = quality["score"]
    aggregate["mean_quality_grade"] = quality["grade"]
    return aggregate


def _benchmark_run_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    metric_keys = (
        "dropout_rate",
        "mean_confidence",
        "fused_jitter_x",
        "fused_jitter_y",
        "fused_jitter_distance",
        "fused_drift_distance",
        "raw_fused_mean_distance",
        "x_axis_confidence_mean",
        "y_axis_confidence_mean",
        "head_motion_mean_magnitude",
    )
    return {f"{key}_delta": _as_float(current.get(key)) - _as_float(previous.get(key)) for key in metric_keys}


def _select_best_benchmark_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    if not runs:
        return {
            "lowest_fused_jitter_distance": None,
            "lowest_dropout_rate": None,
            "highest_mean_confidence": None,
            "highest_quality_score": None,
        }
    return {
        "lowest_fused_jitter_distance": min(runs, key=lambda run: _as_float(run.get("fused_jitter_distance"))),
        "lowest_dropout_rate": min(runs, key=lambda run: _as_float(run.get("dropout_rate"))),
        "highest_mean_confidence": max(runs, key=lambda run: _as_float(run.get("mean_confidence"))),
        "highest_quality_score": max(runs, key=lambda run: _as_float(run.get("quality_score"))),
    }


def _benchmark_report_notes(
    *,
    aggregate: dict[str, Any],
    latest_successful: dict[str, Any] | None,
    previous_successful: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    if not aggregate:
        return notes
    if _as_float(aggregate.get("mean_dropout_rate")) > 0.05:
        notes.append("Dropout rate is still high enough to distort live control; prioritize camera stability and provider robustness.")
    if _as_float(aggregate.get("mean_y_axis_confidence")) + 0.05 < _as_float(aggregate.get("mean_x_axis_confidence")):
        notes.append("Vertical confidence remains materially weaker than horizontal confidence; keep treating Y as the riskier axis.")
    if _as_float(aggregate.get("mean_raw_fused_mean_distance")) < 0.02:
        notes.append("Temporal fusion is smoothing the raw signal without introducing a large average lag offset.")
    if latest_successful is not None and previous_successful is not None:
        delta = _benchmark_run_delta(previous_successful, latest_successful)
        if _as_float(delta.get("fused_jitter_distance_delta")) < -0.002:
            notes.append("Latest successful run reduced fused jitter versus the prior successful run.")
        elif _as_float(delta.get("fused_jitter_distance_delta")) > 0.002:
            notes.append("Latest successful run increased fused jitter versus the prior successful run.")
    return notes


def _score_inverse(value: float, *, good: float, bad: float) -> float:
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((value - good) / max(1e-6, bad - good))))


def _score_direct(value: float, *, good: float, bad: float) -> float:
    if value >= good:
        return 1.0
    if value <= bad:
        return 0.0
    return max(0.0, min(1.0, (value - bad) / max(1e-6, good - bad)))


def _benchmark_quality_score(metrics: dict[str, Any]) -> dict[str, Any]:
    components = {
        "dropout": _score_inverse(_as_float(metrics.get("dropout_rate")), good=0.01, bad=0.12),
        "confidence": _score_direct(_as_float(metrics.get("mean_confidence")), good=0.92, bad=0.65),
        "jitter": _score_inverse(_as_float(metrics.get("fused_jitter_distance")), good=0.010, bad=0.060),
        "drift": _score_inverse(_as_float(metrics.get("fused_drift_distance")), good=0.010, bad=0.080),
        "fusion_lag": _score_inverse(_as_float(metrics.get("raw_fused_mean_distance")), good=0.015, bad=0.080),
        "x_axis_confidence": _score_direct(_as_float(metrics.get("x_axis_confidence_mean")), good=0.85, bad=0.55),
        "y_axis_confidence": _score_direct(_as_float(metrics.get("y_axis_confidence_mean")), good=0.75, bad=0.42),
    }
    weights = {
        "dropout": 0.18,
        "confidence": 0.18,
        "jitter": 0.22,
        "drift": 0.14,
        "fusion_lag": 0.10,
        "x_axis_confidence": 0.08,
        "y_axis_confidence": 0.10,
    }
    weighted_score = sum(components[key] * weights[key] for key in components)
    score = round(weighted_score * 100.0, 1)
    if score >= 85.0:
        grade = "A"
    elif score >= 70.0:
        grade = "B"
    elif score >= 55.0:
        grade = "C"
    elif score >= 40.0:
        grade = "D"
    else:
        grade = "F"
    return {
        "score": score,
        "grade": grade,
        "components": {key: round(value, 4) for key, value in components.items()},
    }


def _benchmark_quality_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    scores = [_as_float(run.get("quality_score")) for run in runs]
    mean_score = round(_mean(scores), 1)
    latest_score = _as_float(runs[-1].get("quality_score"))
    best_score = max(scores) if scores else 0.0
    return {
        "mean_quality_score": mean_score,
        "mean_quality_grade": _quality_grade_from_score(mean_score),
        "latest_quality_score": latest_score,
        "latest_quality_grade": _quality_grade_from_score(latest_score),
        "best_quality_score": best_score,
        "best_quality_grade": _quality_grade_from_score(best_score),
    }


def _quality_grade_from_score(score: float) -> str:
    score = _as_float(score)
    if score >= 85.0:
        return "A"
    if score >= 70.0:
        return "B"
    if score >= 55.0:
        return "C"
    if score >= 40.0:
        return "D"
    return "F"


def _metric_trend(*, previous: float, current: float, higher_is_better: bool, epsilon: float) -> dict[str, Any]:
    delta = current - previous
    if abs(delta) <= epsilon:
        status = "stable"
    elif (delta > 0.0 and higher_is_better) or (delta < 0.0 and not higher_is_better):
        status = "improved"
    else:
        status = "regressed"
    return {
        "status": status,
        "delta": delta,
    }


def _benchmark_metric_specs() -> dict[str, dict[str, Any]]:
    return {
        "quality_score": {
            "label": "quality",
            "higher_is_better": True,
            "epsilon": 1.0,
            "digits": 1,
        },
        "dropout_rate": {
            "label": "dropout",
            "higher_is_better": False,
            "epsilon": 0.01,
            "digits": 3,
        },
        "fused_jitter_distance": {
            "label": "jitter",
            "higher_is_better": False,
            "epsilon": 0.002,
            "digits": 3,
        },
        "fused_drift_distance": {
            "label": "drift",
            "higher_is_better": False,
            "epsilon": 0.003,
            "digits": 3,
        },
        "mean_confidence": {
            "label": "confidence",
            "higher_is_better": True,
            "epsilon": 0.02,
            "digits": 3,
        },
        "x_axis_confidence_mean": {
            "label": "x-axis",
            "higher_is_better": True,
            "epsilon": 0.03,
            "digits": 3,
        },
        "y_axis_confidence_mean": {
            "label": "y-axis",
            "higher_is_better": True,
            "epsilon": 0.03,
            "digits": 3,
        },
        "raw_fused_mean_distance": {
            "label": "fusion lag",
            "higher_is_better": False,
            "epsilon": 0.003,
            "digits": 3,
        },
    }


def _benchmark_metric_trends(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name, spec in _benchmark_metric_specs().items():
        metric = _metric_trend(
            previous=_as_float(previous.get(metric_name)),
            current=_as_float(current.get(metric_name)),
            higher_is_better=bool(spec["higher_is_better"]),
            epsilon=_as_float(spec["epsilon"]),
        )
        metric["label"] = spec["label"]
        metric["epsilon"] = _as_float(spec["epsilon"])
        metric["higher_is_better"] = bool(spec["higher_is_better"])
        metrics[metric_name] = metric
    return metrics


def _mean_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    keys = (
        "dropout_rate",
        "mean_confidence",
        "fused_jitter_distance",
        "fused_drift_distance",
        "raw_fused_mean_distance",
        "x_axis_confidence_mean",
        "y_axis_confidence_mean",
        "head_motion_mean_magnitude",
        "quality_score",
    )
    averaged = {key: _mean([_as_float(run.get(key)) for run in runs]) for key in keys}
    averaged["status"] = "success"
    return averaged


def _benchmark_trend_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        return {}
    latest = runs[-1]
    baseline_runs = runs[:-1][-4:] or runs[:-1]
    baseline = _mean_run(baseline_runs)
    if baseline is None:
        return {}
    metrics = _benchmark_metric_trends(baseline, latest)
    improving = sum(1 for metric in metrics.values() if metric["status"] == "improved")
    regressing = sum(1 for metric in metrics.values() if metric["status"] == "regressed")
    if improving >= regressing + 2:
        overall = "improving"
    elif regressing >= improving + 2:
        overall = "regressing"
    else:
        overall = "mixed"
    return {
        "overall": overall,
        "baseline_run_count": len(baseline_runs),
        "improving_metric_count": improving,
        "regressing_metric_count": regressing,
        "baseline": baseline,
        "metrics": metrics,
    }


def _benchmark_recent_changes(
    runs: list[dict[str, Any]],
    *,
    transition_limit: int = 4,
    item_limit: int = 5,
) -> dict[str, Any]:
    if len(runs) < 2:
        return {}
    transitions: list[dict[str, Any]] = []
    recent_runs = runs[-(transition_limit + 1) :]
    for previous_run, current_run in zip(recent_runs, recent_runs[1:]):
        metric_changes = _benchmark_metric_trends(previous_run, current_run)
        changes: list[dict[str, Any]] = []
        for metric_name, metric in metric_changes.items():
            status = metric.get("status")
            if status == "stable":
                continue
            epsilon = max(1e-6, _as_float(metric.get("epsilon")))
            changes.append(
                {
                    "metric": metric_name,
                    "label": metric.get("label"),
                    "status": status,
                    "delta": _as_float(metric.get("delta")),
                    "magnitude": round(abs(_as_float(metric.get("delta"))) / epsilon, 2),
                    "from_session_log": previous_run.get("session_log"),
                    "to_session_log": current_run.get("session_log"),
                }
            )
        changes.sort(key=lambda change: (-_as_float(change.get("magnitude")), str(change.get("metric") or "")))
        transitions.append(
            {
                "from_session_log": previous_run.get("session_log"),
                "to_session_log": current_run.get("session_log"),
                "changes": changes,
            }
        )
    improvement_entries = [change for transition in transitions for change in transition["changes"] if change["status"] == "improved"]
    regression_entries = [change for transition in transitions for change in transition["changes"] if change["status"] == "regressed"]
    improvement_entries.sort(key=lambda change: (-_as_float(change.get("magnitude")), str(change.get("metric") or "")))
    regression_entries.sort(key=lambda change: (-_as_float(change.get("magnitude")), str(change.get("metric") or "")))
    latest_transition = transitions[-1]
    return {
        "transition_count": len(transitions),
        "latest_transition": latest_transition,
        "top_improvements": improvement_entries[:item_limit],
        "top_regressions": regression_entries[:item_limit],
    }


def _benchmark_report_recommendations(
    *,
    aggregate: dict[str, Any],
    latest_successful: dict[str, Any] | None,
    trend: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if latest_successful is None:
        return recommendations
    if _as_float(latest_successful.get("dropout_rate")) > 0.05:
        recommendations.append("Reduce dropouts first: improve lighting, camera framing, or provider robustness before tuning control behavior.")
    if _as_float(latest_successful.get("y_axis_confidence_mean")) + 0.05 < _as_float(latest_successful.get("x_axis_confidence_mean")):
        recommendations.append("Vertical tracking is still the weaker axis; prioritize vertical feature extraction and calibration quality over cursor tuning.")
    if _as_float(latest_successful.get("fused_jitter_distance")) > 0.03:
        recommendations.append("Fixation jitter is still high enough to affect small targets; prioritize temporal stability and semantic target grounding.")
    if _as_float(latest_successful.get("fused_drift_distance")) > 0.04:
        recommendations.append("Slow drift is still visible; keep bounded online bias correction and posture-change handling on the critical path.")
    if _as_float(latest_successful.get("raw_fused_mean_distance")) > 0.04:
        recommendations.append("Fusion lag is becoming noticeable; avoid adding more smoothing before measuring task latency.")
    if trend.get("overall") == "regressing":
        recommendations.append("The recent trend is regressing overall; compare the latest run against the best-quality run before layering more control changes.")
    elif trend.get("overall") == "improving" and _as_float(aggregate.get("mean_quality_score")) >= 70.0:
        recommendations.append("Recent benchmark quality is improving; the next gains are more likely to come from desktop target grounding than more raw pointer tuning.")
    return recommendations


def _benchmark_dominant_bottleneck(
    *,
    aggregate: dict[str, Any],
    latest_successful: dict[str, Any] | None,
    trend: dict[str, Any],
) -> dict[str, Any]:
    if latest_successful is None:
        return {
            "category": "no_successful_runs",
            "message": "There are no successful benchmark runs to classify yet.",
            "next_focus": "Collect a successful baseline run before drawing conclusions from the report.",
        }
    dropout_rate = _as_float(latest_successful.get("dropout_rate"))
    x_conf = _as_float(latest_successful.get("x_axis_confidence_mean"))
    y_conf = _as_float(latest_successful.get("y_axis_confidence_mean"))
    jitter = _as_float(latest_successful.get("fused_jitter_distance"))
    drift = _as_float(latest_successful.get("fused_drift_distance"))
    fusion_lag = _as_float(latest_successful.get("raw_fused_mean_distance"))
    if dropout_rate > 0.05:
        return {
            "category": "dropout_rate",
            "message": "Too many frames are still being lost for stable live control.",
            "next_focus": "Improve lighting, camera framing, or provider robustness before more control tuning.",
        }
    if y_conf + 0.05 < x_conf:
        return {
            "category": "vertical_axis",
            "message": "Vertical confidence is materially weaker than horizontal confidence.",
            "next_focus": "Prioritize vertical feature extraction and calibration robustness over cursor tuning.",
        }
    if jitter > 0.03:
        return {
            "category": "fixation_jitter",
            "message": "Fixation jitter is still high enough to hurt small-target precision.",
            "next_focus": "Keep temporal stability and target grounding ahead of more aggressive pointer behavior.",
        }
    if drift > 0.04:
        return {
            "category": "slow_drift",
            "message": "Slow drift is still strong enough to bias targeting during live use.",
            "next_focus": "Keep online bias correction and posture handling on the critical path.",
        }
    if fusion_lag > 0.04:
        return {
            "category": "fusion_lag",
            "message": "The fused signal is drifting too far from the raw signal, which suggests noticeable lag.",
            "next_focus": "Avoid stacking more smoothing before re-measuring end-to-end latency.",
        }
    if trend.get("overall") == "regressing":
        return {
            "category": "recent_regression",
            "message": "The recent benchmark trend is regressing overall.",
            "next_focus": "Compare the latest run with the best-quality run before layering more control changes.",
        }
    if _as_float(aggregate.get("mean_quality_score")) >= 70.0:
        return {
            "category": "desktop_grounding",
            "message": "Raw signal quality is good enough that desktop target grounding is the next limiting factor.",
            "next_focus": "Shift effort from cursor tuning to real UI target grounding and locked local precision.",
        }
    return {
        "category": "general_quality",
        "message": "Overall benchmark quality is still below the target level, but no single dominant failure stands out.",
        "next_focus": "Keep improving raw signal quality and calibration consistency before expanding interaction features.",
    }


def _group_benchmark_runs(runs: list[dict[str, Any]], *, group_by: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group_key = _benchmark_group_key(run, group_by=group_by)
        groups.setdefault(group_key, []).append(run)
    grouped_summary: dict[str, dict[str, Any]] = {}
    for group_key, group_runs in sorted(groups.items()):
        aggregate = _aggregate_benchmark_runs(group_runs)
        latest = group_runs[-1]
        previous = group_runs[-2] if len(group_runs) > 1 else None
        trend = _benchmark_trend_summary(group_runs)
        grouped_summary[group_key] = {
            "group": group_key,
            "group_by": group_by,
            "run_count": len(group_runs),
            "latest": latest,
            "previous": previous,
            "aggregate": aggregate,
            "quality_summary": _benchmark_quality_summary(group_runs),
            "trend": trend,
            "best_runs": _select_best_benchmark_runs(group_runs),
            "dominant_bottleneck": _benchmark_dominant_bottleneck(
                aggregate=aggregate,
                latest_successful=latest,
                trend=trend,
            ),
            "recommendations": _benchmark_report_recommendations(
                aggregate=aggregate,
                latest_successful=latest,
                trend=trend,
            ),
        }
    return grouped_summary


def _benchmark_group_comparison(groups: dict[str, dict[str, Any]], *, group_by: str) -> dict[str, Any]:
    if not groups:
        return {}
    entries: list[dict[str, Any]] = []
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        quality_summary = group.get("quality_summary")
        trend = group.get("trend")
        bottleneck = group.get("dominant_bottleneck")
        if not isinstance(quality_summary, dict):
            quality_summary = {}
        if not isinstance(trend, dict):
            trend = {}
        if not isinstance(bottleneck, dict):
            bottleneck = {}
        entries.append(
            {
                "group": str(group.get("group") or "unknown"),
                "group_by": group_by,
                "run_count": _as_int(group.get("run_count")),
                "latest_quality_score": _as_float(quality_summary.get("latest_quality_score")),
                "latest_quality_grade": quality_summary.get("latest_quality_grade"),
                "mean_quality_score": _as_float(quality_summary.get("mean_quality_score")),
                "mean_quality_grade": quality_summary.get("mean_quality_grade"),
                "trend_overall": trend.get("overall"),
                "improving_metric_count": _as_int(trend.get("improving_metric_count")),
                "regressing_metric_count": _as_int(trend.get("regressing_metric_count")),
                "dominant_bottleneck": bottleneck.get("category"),
            }
        )
    entries.sort(
        key=lambda entry: (
            -_as_float(entry.get("mean_quality_score")),
            -_as_float(entry.get("latest_quality_score")),
            -_as_int(entry.get("run_count")),
            str(entry.get("group") or ""),
        )
    )
    weakest_group = min(
        entries,
        key=lambda entry: (
            _as_float(entry.get("mean_quality_score")),
            _as_float(entry.get("latest_quality_score")),
            str(entry.get("group") or ""),
        ),
    )
    improving_entries = [entry for entry in entries if entry.get("trend_overall") == "improving"]
    most_improved_group = None
    if improving_entries:
        most_improved_group = max(
            improving_entries,
            key=lambda entry: (
                _as_int(entry.get("improving_metric_count")) - _as_int(entry.get("regressing_metric_count")),
                _as_float(entry.get("latest_quality_score")),
                -_as_int(entry.get("regressing_metric_count")),
            ),
        )
    return {
        "group_by": group_by,
        "group_count": len(entries),
        "groups": entries,
        "best_group": entries[0],
        "weakest_group": weakest_group,
        "most_improved_group": most_improved_group,
        "quality_spread": round(
            _as_float(entries[0].get("mean_quality_score")) - _as_float(weakest_group.get("mean_quality_score")),
            1,
        ),
    }


def _benchmark_group_metric_specs() -> dict[str, dict[str, Any]]:
    return {
        "mean_quality_score": {
            "label": "quality",
            "higher_is_better": True,
            "digits": 1,
        },
        "mean_dropout_rate": {
            "label": "dropout",
            "higher_is_better": False,
            "digits": 3,
        },
        "mean_fused_jitter_distance": {
            "label": "jitter",
            "higher_is_better": False,
            "digits": 3,
        },
        "mean_fused_drift_distance": {
            "label": "drift",
            "higher_is_better": False,
            "digits": 3,
        },
        "mean_y_axis_confidence": {
            "label": "y-axis",
            "higher_is_better": True,
            "digits": 3,
        },
        "mean_confidence": {
            "label": "confidence",
            "higher_is_better": True,
            "digits": 3,
        },
    }


def _benchmark_leaderboard_metric_specs() -> dict[str, dict[str, Any]]:
    return {
        "quality_score": {
            "label": "quality",
            "higher_is_better": True,
            "digits": 1,
        },
        "dropout_rate": {
            "label": "dropout",
            "higher_is_better": False,
            "digits": 3,
        },
        "fused_jitter_distance": {
            "label": "jitter",
            "higher_is_better": False,
            "digits": 3,
        },
        "fused_drift_distance": {
            "label": "drift",
            "higher_is_better": False,
            "digits": 3,
        },
        "y_axis_confidence_mean": {
            "label": "y-axis",
            "higher_is_better": True,
            "digits": 3,
        },
        "mean_confidence": {
            "label": "confidence",
            "higher_is_better": True,
            "digits": 3,
        },
    }


def _benchmark_metric_leaderboards(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name, spec in _benchmark_leaderboard_metric_specs().items():
        ranked = sorted(
            runs,
            key=lambda run: (
                -_as_float(run.get(metric_name)) if bool(spec["higher_is_better"]) else _as_float(run.get(metric_name)),
                _benchmark_run_label(run),
            ),
        )
        best_run = ranked[0]
        weakest_run = ranked[-1]
        best_value = _as_float(best_run.get(metric_name))
        weakest_value = _as_float(weakest_run.get(metric_name))
        metrics[metric_name] = {
            "metric": metric_name,
            "label": spec["label"],
            "higher_is_better": spec["higher_is_better"],
            "best_run": {
                "session_log": best_run.get("session_log"),
                "tracker": best_run.get("tracker"),
                "value": best_value,
                "quality_score": _as_float(best_run.get("quality_score")),
                "quality_grade": best_run.get("quality_grade"),
            },
            "weakest_run": {
                "session_log": weakest_run.get("session_log"),
                "tracker": weakest_run.get("tracker"),
                "value": weakest_value,
                "quality_score": _as_float(weakest_run.get("quality_score")),
                "quality_grade": weakest_run.get("quality_grade"),
            },
            "spread": round(abs(best_value - weakest_value), int(spec["digits"])),
        }
    return {
        "run_count": len(runs),
        "metrics": metrics,
    }


def _benchmark_group_metric_hotspots(groups: dict[str, dict[str, Any]], *, group_by: str) -> dict[str, Any]:
    if not groups:
        return {}
    group_entries: list[dict[str, Any]] = []
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        aggregate = group.get("aggregate")
        if not isinstance(aggregate, dict):
            aggregate = {}
        group_entries.append(
            {
                "group": str(group.get("group") or "unknown"),
                "aggregate": aggregate,
            }
        )
    metric_summaries: dict[str, dict[str, Any]] = {}
    for metric_name, spec in _benchmark_group_metric_specs().items():
        ranked = sorted(
            group_entries,
            key=lambda entry: (
                -_as_float(entry["aggregate"].get(metric_name))
                if bool(spec["higher_is_better"])
                else _as_float(entry["aggregate"].get(metric_name)),
                str(entry.get("group") or ""),
            ),
        )
        best_group = ranked[0]
        weakest_group = ranked[-1]
        best_value = _as_float(best_group["aggregate"].get(metric_name))
        weakest_value = _as_float(weakest_group["aggregate"].get(metric_name))
        metric_summaries[metric_name] = {
            "metric": metric_name,
            "label": spec["label"],
            "higher_is_better": spec["higher_is_better"],
            "best_group": {"group": best_group["group"], "value": best_value},
            "weakest_group": {"group": weakest_group["group"], "value": weakest_value},
            "spread": round(abs(best_value - weakest_value), int(spec["digits"])),
        }
    return {
        "group_by": group_by,
        "group_count": len(group_entries),
        "metrics": metric_summaries,
    }


def _render_benchmark_report_text(payload: dict[str, Any], *, brief: bool = False, top: int | None = None) -> str:
    report = payload.get("benchmark_report")
    if not isinstance(report, dict):
        return "Gaze Benchmark Report\nInvalid benchmark report payload."

    lines = [
        "Gaze Benchmark Report",
        f"Runtime Dir: {payload.get('runtime_dir', '-')}",
    ]
    tracker_filter = report.get("tracker_filter")
    if tracker_filter:
        lines.append(f"Tracker Filter: {tracker_filter}")

    status = str(report.get("status") or "unknown")
    lines.append(f"Status: {status}")
    if status != "success":
        error = report.get("error")
        message = report.get("message")
        if error:
            lines.append(f"Error: {error}")
        if message:
            lines.append(f"Message: {message}")
        skipped_paths = report.get("skipped_paths")
        if isinstance(skipped_paths, list) and skipped_paths:
            lines.append(f"Skipped Summaries: {len(skipped_paths)}")
        return "\n".join(lines)

    lines.append(
        f"Runs: {report.get('run_count', 0)} total, {report.get('successful_run_count', 0)} successful"
    )

    if brief:
        lines.extend(_render_benchmark_report_brief_lines(report, top=top))
        return "\n".join(lines)

    latest_successful = report.get("latest_successful")
    if isinstance(latest_successful, dict):
        lines.extend(
            [
                "",
                "Latest Successful Run",
                _render_benchmark_run_line(
                    latest_successful,
                    prefix="- Latest",
                    include_tracker=True,
                ),
            ]
        )

    aggregate = report.get("aggregate")
    if isinstance(aggregate, dict) and aggregate:
        lines.extend(
            [
                "",
                "Aggregate",
                _render_benchmark_aggregate_line(aggregate, prefix="- Mean"),
            ]
        )

    trend = report.get("trend")
    if isinstance(trend, dict) and trend:
        lines.extend(
            [
                "",
                "Trend",
                _render_benchmark_trend_line(trend),
            ]
        )

    delta = report.get("delta_from_previous_successful")
    if isinstance(delta, dict) and delta:
        lines.extend(
            [
                "",
                "Latest Delta",
                (
                    "- "
                    f"jitter {_format_signed_float(delta.get('fused_jitter_distance_delta'))}, "
                    f"drift {_format_signed_float(delta.get('fused_drift_distance_delta'))}, "
                    f"dropout {_format_signed_percent(delta.get('dropout_rate_delta'))}, "
                    f"y-axis {_format_signed_float(delta.get('y_axis_confidence_mean_delta'), digits=2)}"
                ),
            ]
        )

    best_runs = report.get("best_runs")
    if isinstance(best_runs, dict) and best_runs:
        best_quality = best_runs.get("highest_quality_score")
        best_jitter = best_runs.get("lowest_fused_jitter_distance")
        best_dropout = best_runs.get("lowest_dropout_rate")
        lines.extend(["", "Best Runs"])
        if isinstance(best_quality, dict):
            lines.append(_render_benchmark_run_line(best_quality, prefix="- Quality"))
        if isinstance(best_jitter, dict):
            lines.append(_render_benchmark_run_line(best_jitter, prefix="- Jitter"))
        if isinstance(best_dropout, dict):
            lines.append(_render_benchmark_run_line(best_dropout, prefix="- Dropout"))

    notes = report.get("notes")
    if isinstance(notes, list) and notes:
        lines.extend(["", "Notes"])
        lines.extend(f"- {note}" for note in notes if note)

    recommendations = report.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        lines.extend(["", "Recommendations"])
        lines.extend(f"- {recommendation}" for recommendation in recommendations if recommendation)

    bottleneck = report.get("dominant_bottleneck")
    if isinstance(bottleneck, dict) and bottleneck:
        lines.extend(["", "Bottleneck", _render_benchmark_bottleneck_line(bottleneck)])

    groups_by_tracker = report.get("groups_by_tracker")
    if isinstance(groups_by_tracker, dict) and groups_by_tracker:
        lines.extend(["", "By Tracker"])
        lines.extend(_render_benchmark_group_line(group) for group in groups_by_tracker.values() if isinstance(group, dict))

    groups_by_session_prefix = report.get("groups_by_session_prefix")
    if isinstance(groups_by_session_prefix, dict) and groups_by_session_prefix:
        lines.extend(["", "By Session Prefix"])
        lines.extend(
            _render_benchmark_group_line(group)
            for group in groups_by_session_prefix.values()
            if isinstance(group, dict)
        )

    group_comparisons = report.get("group_comparisons")
    if isinstance(group_comparisons, dict) and group_comparisons:
        tracker_comparison = group_comparisons.get("tracker")
        session_comparison = group_comparisons.get("session_prefix")
        if isinstance(tracker_comparison, dict) and tracker_comparison:
            lines.extend(["", "Tracker Comparison", _render_benchmark_group_comparison_line(tracker_comparison)])
        if isinstance(session_comparison, dict) and session_comparison:
            lines.extend(["", "Session Comparison", _render_benchmark_group_comparison_line(session_comparison)])

    group_metric_hotspots = report.get("group_metric_hotspots")
    if isinstance(group_metric_hotspots, dict) and group_metric_hotspots:
        tracker_hotspots = group_metric_hotspots.get("tracker")
        session_hotspots = group_metric_hotspots.get("session_prefix")
        if isinstance(tracker_hotspots, dict) and tracker_hotspots:
            lines.extend(["", "Tracker Hotspots"])
            lines.extend(_render_benchmark_group_hotspots_lines(tracker_hotspots, top=top))
        if isinstance(session_hotspots, dict) and session_hotspots:
            lines.extend(["", "Session Hotspots"])
            lines.extend(_render_benchmark_group_hotspots_lines(session_hotspots, top=top))

    metric_leaderboards = report.get("metric_leaderboards")
    if isinstance(metric_leaderboards, dict) and metric_leaderboards:
        lines.extend(["", "Metric Leaderboards"])
        lines.extend(_render_benchmark_metric_leaderboards_lines(metric_leaderboards, top=top))

    recent_changes = report.get("recent_changes")
    if isinstance(recent_changes, dict) and recent_changes:
        lines.extend(["", "Recent Changes"])
        lines.extend(_render_benchmark_recent_changes_lines(recent_changes, top=top))

    return "\n".join(lines)


def _render_benchmark_run_line(
    run: dict[str, Any],
    *,
    prefix: str,
    include_tracker: bool = False,
) -> str:
    label = _benchmark_run_label(run)
    parts = [prefix, label]
    tracker = run.get("tracker")
    if include_tracker and tracker:
        parts.append(f"tracker {tracker}")
    parts.extend(
        [
            f"grade {run.get('quality_grade', '-')} {_format_float(run.get('quality_score'), digits=1)}",
            f"dropout {_format_percent(run.get('dropout_rate'))}",
            f"conf {_format_float(run.get('mean_confidence'), digits=2)}",
            f"jitter {_format_float(run.get('fused_jitter_distance'))}",
            f"drift {_format_float(run.get('fused_drift_distance'))}",
            f"y-axis {_format_float(run.get('y_axis_confidence_mean'), digits=2)}",
        ]
    )
    return " | ".join(parts)


def _render_benchmark_aggregate_line(aggregate: dict[str, Any], *, prefix: str) -> str:
    tracker_counts = aggregate.get("tracker_counts")
    tracker_summary = ""
    if isinstance(tracker_counts, dict) and tracker_counts:
        tracker_summary = ", ".join(f"{tracker}={count}" for tracker, count in sorted(tracker_counts.items()))
    parts = [
        prefix,
        f"grade {aggregate.get('mean_quality_grade', '-')} {_format_float(aggregate.get('mean_quality_score'), digits=1)}",
        f"dropout {_format_percent(aggregate.get('mean_dropout_rate'))}",
        f"conf {_format_float(aggregate.get('mean_confidence'), digits=2)}",
        f"jitter {_format_float(aggregate.get('mean_fused_jitter_distance'))}",
        f"drift {_format_float(aggregate.get('mean_fused_drift_distance'))}",
        f"y-axis {_format_float(aggregate.get('mean_y_axis_confidence'), digits=2)}",
    ]
    if tracker_summary:
        parts.append(f"trackers {tracker_summary}")
    return " | ".join(parts)


def _render_benchmark_trend_line(trend: dict[str, Any]) -> str:
    metrics = trend.get("metrics")
    metric_parts: list[str] = []
    if isinstance(metrics, dict):
        for metric_name, short_label in (
            ("quality_score", "quality"),
            ("fused_jitter_distance", "jitter"),
            ("fused_drift_distance", "drift"),
            ("mean_confidence", "conf"),
            ("y_axis_confidence_mean", "y-axis"),
        ):
            metric = metrics.get(metric_name)
            if isinstance(metric, dict):
                metric_parts.append(f"{short_label}={metric.get('status', 'unknown')}")
    parts = [
        "-",
        f"overall {trend.get('overall', 'unknown')}",
        f"baseline {trend.get('baseline_run_count', 0)} runs",
        f"improving {trend.get('improving_metric_count', 0)}",
        f"regressing {trend.get('regressing_metric_count', 0)}",
    ]
    if metric_parts:
        parts.append(", ".join(metric_parts))
    return " | ".join(parts)


def _render_benchmark_group_line(group: dict[str, Any]) -> str:
    latest = group.get("latest")
    quality_summary = group.get("quality_summary")
    trend = group.get("trend")
    group_name = str(group.get("group") or "unknown")
    latest_grade = "-"
    latest_score = "-"
    if isinstance(quality_summary, dict):
        latest_grade = str(quality_summary.get("latest_quality_grade") or "-")
        latest_score = _format_float(quality_summary.get("latest_quality_score"), digits=1)
    latest_label = _benchmark_run_label(latest) if isinstance(latest, dict) else "-"
    return (
        "- "
        f"{group_name}: {group.get('run_count', 0)} runs | "
        f"latest {latest_label} | "
        f"quality {latest_grade} {latest_score} | "
        f"trend {trend.get('overall', 'unknown') if isinstance(trend, dict) else 'unknown'}"
    )


def _render_benchmark_bottleneck_line(bottleneck: dict[str, Any]) -> str:
    category = str(bottleneck.get("category") or "unknown")
    message = str(bottleneck.get("message") or "")
    next_focus = str(bottleneck.get("next_focus") or "")
    parts = ["-", category]
    if message:
        parts.append(message)
    if next_focus:
        parts.append(f"next: {next_focus}")
    return " | ".join(parts)


def _render_benchmark_group_comparison_line(comparison: dict[str, Any]) -> str:
    best_group = comparison.get("best_group")
    weakest_group = comparison.get("weakest_group")
    most_improved_group = comparison.get("most_improved_group")
    parts = [
        "-",
        f"{comparison.get('group_by', 'group')} groups {comparison.get('group_count', 0)}",
    ]
    if isinstance(best_group, dict):
        parts.append(
            "best "
            f"{best_group.get('group', 'unknown')} "
            f"({best_group.get('mean_quality_grade', '-')} {_format_float(best_group.get('mean_quality_score'), digits=1)})"
        )
    if isinstance(weakest_group, dict):
        parts.append(
            "weakest "
            f"{weakest_group.get('group', 'unknown')} "
            f"({weakest_group.get('mean_quality_grade', '-')} {_format_float(weakest_group.get('mean_quality_score'), digits=1)})"
        )
    if isinstance(most_improved_group, dict):
        parts.append(f"most improved {most_improved_group.get('group', 'unknown')}")
    return " | ".join(parts)


def _render_benchmark_group_hotspots_lines(hotspots: dict[str, Any], *, top: int | None = None) -> list[str]:
    metrics = hotspots.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return [f"- {hotspots.get('group_by', 'group')} groups {hotspots.get('group_count', 0)}"]
    lines = [f"- {hotspots.get('group_by', 'group')} groups {hotspots.get('group_count', 0)}"]
    preferred_order = (
        "mean_quality_score",
        "mean_fused_jitter_distance",
        "mean_dropout_rate",
        "mean_fused_drift_distance",
        "mean_y_axis_confidence",
    )
    metric_limit = len(preferred_order) if top is None else max(1, min(top, len(preferred_order)))
    for metric_name in preferred_order[:metric_limit]:
        metric = metrics.get(metric_name)
        if isinstance(metric, dict):
            lines.append(_render_benchmark_hotspot_metric_line(metric))
    return lines


def _render_benchmark_report_top_limit(value: Any) -> int | None:
    if value is None:
        return None
    top = _as_int(value)
    if top <= 0:
        return None
    return top


def _render_benchmark_report_output(
    payload: dict[str, Any],
    *,
    output_format: str,
    brief: bool = False,
    top: int | None = None,
) -> str:
    if output_format == "text":
        return _render_benchmark_report_text(payload, brief=brief, top=top)
    return json.dumps(payload, indent=2)


def _render_benchmark_hotspot_metric_line(metric: dict[str, Any]) -> str:
    label = str(metric.get("label") or metric.get("metric") or "metric")
    best_group = metric.get("best_group")
    weakest_group = metric.get("weakest_group")
    best_text = "best -"
    weakest_text = "weakest -"
    if isinstance(best_group, dict):
        best_text = (
            f"best {best_group.get('group', 'unknown')} "
            f"({_format_float(best_group.get('value'), digits=3)})"
        )
    if isinstance(weakest_group, dict):
        weakest_text = (
            f"weakest {weakest_group.get('group', 'unknown')} "
            f"({_format_float(weakest_group.get('value'), digits=3)})"
        )
    return f"- {label}: {best_text} | {weakest_text} | spread {_format_float(metric.get('spread'), digits=3)}"


def _render_benchmark_metric_leaderboards_lines(leaderboards: dict[str, Any], *, top: int | None = None) -> list[str]:
    metrics = leaderboards.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return [f"- runs {leaderboards.get('run_count', 0)}"]
    lines = [f"- runs {leaderboards.get('run_count', 0)}"]
    preferred_order = (
        "quality_score",
        "fused_jitter_distance",
        "dropout_rate",
        "fused_drift_distance",
        "y_axis_confidence_mean",
    )
    metric_limit = len(preferred_order) if top is None else max(1, min(top, len(preferred_order)))
    for metric_name in preferred_order[:metric_limit]:
        metric = metrics.get(metric_name)
        if isinstance(metric, dict):
            lines.append(_render_benchmark_leaderboard_metric_line(metric))
    return lines


def _render_benchmark_leaderboard_metric_line(metric: dict[str, Any]) -> str:
    label = str(metric.get("label") or metric.get("metric") or "metric")
    best_run = metric.get("best_run")
    weakest_run = metric.get("weakest_run")
    best_text = "best -"
    weakest_text = "weakest -"
    if isinstance(best_run, dict):
        best_text = (
            f"best {_benchmark_path_label(best_run.get('session_log'))} "
            f"({_format_float(best_run.get('value'), digits=3)})"
        )
    if isinstance(weakest_run, dict):
        weakest_text = (
            f"weakest {_benchmark_path_label(weakest_run.get('session_log'))} "
            f"({_format_float(weakest_run.get('value'), digits=3)})"
        )
    return f"- {label}: {best_text} | {weakest_text} | spread {_format_float(metric.get('spread'), digits=3)}"


def _render_benchmark_report_brief_lines(report: dict[str, Any], *, top: int | None = None) -> list[str]:
    lines: list[str] = []
    latest_successful = report.get("latest_successful")
    if isinstance(latest_successful, dict):
        lines.extend(
            [
                "",
                "Latest",
                _render_benchmark_run_line(latest_successful, prefix="- Latest", include_tracker=True),
            ]
        )
    dominant_bottleneck = report.get("dominant_bottleneck")
    if isinstance(dominant_bottleneck, dict) and dominant_bottleneck:
        lines.extend(
            [
                "",
                "Bottleneck",
                _render_benchmark_bottleneck_line(dominant_bottleneck),
            ]
        )
    recent_changes = report.get("recent_changes")
    if isinstance(recent_changes, dict) and recent_changes:
        top_regressions = recent_changes.get("top_regressions")
        top_improvements = recent_changes.get("top_improvements")
        change_limit = 2 if top is None else max(1, top)
        lines.extend(["", "Recent"])
        if isinstance(top_regressions, list) and top_regressions:
            change_text = ", ".join(
                _render_benchmark_change_entry(change)
                for change in top_regressions[:change_limit]
                if isinstance(change, dict)
            )
            if change_text:
                lines.append(f"- regressions {change_text}")
        if isinstance(top_improvements, list) and top_improvements:
            change_text = ", ".join(
                _render_benchmark_change_entry(change)
                for change in top_improvements[:change_limit]
                if isinstance(change, dict)
            )
            if change_text:
                lines.append(f"- improvements {change_text}")
    metric_leaderboards = report.get("metric_leaderboards")
    if isinstance(metric_leaderboards, dict) and metric_leaderboards:
        metrics = metric_leaderboards.get("metrics")
        if isinstance(metrics, dict):
            lines.extend(["", "Leaders"])
            metric_names = ("quality_score", "fused_jitter_distance", "dropout_rate")
            metric_limit = len(metric_names) if top is None else max(1, min(top, len(metric_names)))
            for metric_name in metric_names[:metric_limit]:
                metric = metrics.get(metric_name)
                if isinstance(metric, dict):
                    lines.append(_render_benchmark_leaderboard_metric_line(metric))
    return lines


def _render_benchmark_recent_changes_lines(recent_changes: dict[str, Any], *, top: int | None = None) -> list[str]:
    lines = [f"- transitions {recent_changes.get('transition_count', 0)}"]
    change_limit = 3 if top is None else max(1, top)
    latest_transition = recent_changes.get("latest_transition")
    if isinstance(latest_transition, dict):
        from_label = _benchmark_path_label(latest_transition.get("from_session_log"))
        to_label = _benchmark_path_label(latest_transition.get("to_session_log"))
        changes = latest_transition.get("changes")
        lines.append(f"- latest {from_label} -> {to_label}")
        if isinstance(changes, list) and changes:
            top_changes = changes[:change_limit]
            change_text = ", ".join(_render_benchmark_change_entry(change) for change in top_changes if isinstance(change, dict))
            if change_text:
                lines.append(f"- latest changes {change_text}")
    top_improvements = recent_changes.get("top_improvements")
    if isinstance(top_improvements, list) and top_improvements:
        lines.append(
            "- top improvements "
            + ", ".join(_render_benchmark_change_entry(change) for change in top_improvements[:change_limit] if isinstance(change, dict))
        )
    top_regressions = recent_changes.get("top_regressions")
    if isinstance(top_regressions, list) and top_regressions:
        lines.append(
            "- top regressions "
            + ", ".join(_render_benchmark_change_entry(change) for change in top_regressions[:change_limit] if isinstance(change, dict))
        )
    return lines


def _render_benchmark_change_entry(change: dict[str, Any]) -> str:
    label = str(change.get("label") or change.get("metric") or "metric")
    status = str(change.get("status") or "unknown")
    magnitude = _as_float(change.get("magnitude"))
    return f"{label} {status} ({_format_signed_float(change.get('delta'))}, x{magnitude:.1f})"


def _benchmark_run_label(run: dict[str, Any]) -> str:
    session_log = str(run.get("session_log") or "")
    if session_log:
        return Path(session_log).name
    summary_path = str(run.get("summary_path") or "")
    if summary_path:
        return Path(summary_path).name
    return "unknown"


def _benchmark_path_label(path: Any) -> str:
    text = str(path or "")
    if not text:
        return "unknown"
    return Path(text).name


def _format_float(value: Any, *, digits: int = 3) -> str:
    return f"{_as_float(value):.{digits}f}"


def _format_percent(value: Any) -> str:
    return f"{_as_float(value) * 100.0:.1f}%"


def _format_signed_float(value: Any, *, digits: int = 3) -> str:
    return f"{_as_float(value):+.{digits}f}"


def _format_signed_percent(value: Any) -> str:
    return f"{_as_float(value) * 100.0:+.1f}%"


def _benchmark_group_key(run: dict[str, Any], *, group_by: str) -> str:
    if group_by == "tracker":
        return str(run.get("tracker") or "unknown")
    if group_by == "session_prefix":
        session_log = str(run.get("session_log") or "")
        stem = Path(session_log).stem if session_log else ""
        if not stem:
            return "unknown"
        return re.sub(r"-\d+$", "", stem) or stem
    return "all"


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


def _append_calibration_capture_feedback(
    events: list[object],
    step_result: dict[str, Any],
    *,
    step_index: int,
    total_steps: int,
) -> None:
    label = str(step_result.get("label") or "unknown area")
    if not step_result.get("success"):
        events.append(
            GazeFeedbackEvent(
                phase=GazeLoopPhase.CALIBRATING,
                message=(
                    f"Calibration point {step_index}/{total_steps} near the {label} could not capture enough stable webcam readings."
                ),
            )
        )
        return

    message = (
        f"Captured calibration point {step_index}/{total_steps} near the {label} "
        f"with confidence {_format_float(step_result.get('confidence', 0.0))}."
    )
    quality_hint = _calibration_capture_quality_hint(step_result.get("quality"))
    if quality_hint:
        message = f"{message} {quality_hint}"
    events.append(GazeFeedbackEvent(phase=GazeLoopPhase.CALIBRATING, message=message))


def _calibration_capture_quality_hint(quality: Any) -> str:
    if not isinstance(quality, dict):
        return ""
    issues: list[str] = []
    x_confidence = quality.get("x_axis_confidence")
    y_confidence = quality.get("y_axis_confidence")
    if isinstance(x_confidence, (int, float)) and float(x_confidence) < 0.65:
        issues.append("X signal looked weak.")
    if isinstance(y_confidence, (int, float)) and float(y_confidence) < 0.58:
        issues.append("Y signal looked weak.")
    horizontal_disagreement = quality.get("horizontal_disagreement")
    vertical_disagreement = quality.get("vertical_disagreement")
    if isinstance(horizontal_disagreement, (int, float)) and float(horizontal_disagreement) > 0.16:
        issues.append("Eyes disagreed horizontally.")
    if isinstance(vertical_disagreement, (int, float)) and float(vertical_disagreement) > 0.16:
        issues.append("Eyes disagreed vertically.")
    if not issues:
        return "Per-step signal looked stable."
    return " ".join(issues[:2])


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
        mode_message = (
            "Fixate on the blue target. Use the red tracked dot as feedback only, "
            "then click, press Space, or press Enter once the gaze feels stable."
        )
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
        "dot_size": _calibration_dot_size(args),
        "interactive": interactive,
        "status": "disabled",
    }
    if interactive:
        payload["display"] = "primary-screen"
    return payload


def _calibration_dot_size(args: argparse.Namespace) -> int:
    return max(16, int(getattr(args, "calibration_dot_size", getattr(args, "dot_size", 40))))


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


def _gaze_live_dry_run_warning(effective_dry_run: bool) -> str | None:
    if not effective_dry_run:
        return None
    return (
        "Dry-run mode is enabled. Gaze preview is live, but cursor movement and click actions "
        "are only simulated. Pass --execute to control macOS for real."
    )


def _settings_payload(settings, effective_dry_run: bool) -> dict[str, Any]:
    payload = dict(settings.__dict__)
    payload["effective_dry_run"] = effective_dry_run
    return payload


def _tracker_capabilities_payload(tracker: str) -> dict[str, Any]:
    provider_class = MediaPipeWebcamGazeProvider if tracker == "mediapipe" else OpenCVWebcamGazeProvider
    describe_capabilities = getattr(provider_class, "describe_capabilities", None)
    if callable(describe_capabilities):
        return describe_capabilities()
    if tracker == "mediapipe":
        return {
            "tracker": "mediapipe",
            "supported_signals": [
                "mouth_open",
                "left_eye_closed",
                "right_eye_closed",
                "head_yaw",
                "head_pitch",
                "head_roll",
            ],
        }
    return {"tracker": "opencv", "supported_signals": []}


def _gaze_mode_support_payload(gaze_mode: str, *, tracker_capabilities: dict[str, Any]) -> dict[str, Any]:
    supported_signals = set(tracker_capabilities.get("supported_signals", []))
    requirements: dict[str, Any] = {
        "required_signals": [],
        "optional_signals": [],
        "requires_calibration_profile": True,
        "runtime_supported": True,
        "runtime_strategy": "existing_loop",
    }
    if gaze_mode in {"cursor-relative", "cursor-relative-click"}:
        requirements.update(
            {
                "requires_calibration_profile": False,
                "runtime_strategy": "relative_raw",
            }
        )
    elif gaze_mode in {"cursor-assist", "cursor-assist-click"}:
        requirements.update(
            {
                "required_signals": ["mouth_open"],
                "optional_signals": ["head_yaw", "head_pitch", "head_roll"],
                "requires_calibration_profile": False,
                "runtime_strategy": "assist_clutch",
            }
        )
    elif gaze_mode in {"cursor-snap-wink", "wink-click-test"}:
        requirements.update(
            {
                "required_signals": ["left_eye_closed", "right_eye_closed"],
                "requires_calibration_profile": gaze_mode != "wink-click-test",
                "runtime_strategy": "wink_click_test" if gaze_mode == "wink-click-test" else "snap_wink",
            }
        )
    elif gaze_mode == "benchmark":
        requirements.update(
            {
                "requires_calibration_profile": False,
                "runtime_strategy": "benchmark",
            }
        )

    missing_signals = [signal for signal in requirements["required_signals"] if signal not in supported_signals]
    return {
        "gaze_mode": gaze_mode,
        **requirements,
        "tracker_supported": not missing_signals,
        "missing_signals": missing_signals,
    }


def _unsupported_tracker_mode_error(
    *,
    tracker: str,
    gaze_mode: str,
    mode_support: dict[str, Any],
) -> dict[str, Any]:
    required_signals = ", ".join(mode_support["required_signals"])
    missing_signals = ", ".join(mode_support["missing_signals"])
    return {
        "error": "unsupported_tracker_mode",
        "message": (
            f'Tracker "{tracker}" does not expose the control signals required for "{gaze_mode}". '
            f"Required: {required_signals}. Missing: {missing_signals}."
        ),
        "payload": {
            "required_signals": list(mode_support["required_signals"]),
            "missing_signals": list(mode_support["missing_signals"]),
        },
    }


def _unsupported_gaze_mode_error(*, gaze_mode: str, mode_support: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": "unsupported_gaze_mode",
        "message": f'The "{gaze_mode}" control loop is not available in the current runtime.',
        "payload": {
            "required_signals": list(mode_support["required_signals"]),
            "runtime_strategy": mode_support["runtime_strategy"],
        },
    }


def _cursor_settings_payload(args: argparse.Namespace) -> dict[str, float]:
    gaze_mode = str(getattr(args, "action", "cursor"))
    return _effective_cursor_settings(args, gaze_mode=gaze_mode)


def _effective_cursor_settings(args: argparse.Namespace, *, gaze_mode: str) -> dict[str, float]:
    relative_mode = gaze_mode in {"cursor-relative", "cursor-relative-click", "cursor-assist", "cursor-assist-click"}
    deadzone = getattr(args, "cursor_deadzone", None)
    smoothing = getattr(args, "cursor_smoothing", None)
    edge_padding = getattr(args, "cursor_edge_padding", None)
    max_step = getattr(args, "cursor_max_step", None)
    base_gain = getattr(args, "cursor_gain", None)

    if deadzone is None:
        deadzone = 0.025 if relative_mode else 0.05
    if smoothing is None:
        smoothing = 0.1
    if edge_padding is None:
        edge_padding = 0.05
    if max_step is None:
        max_step = 0.05 if relative_mode else 0.04
    if base_gain is None:
        base_gain = 0.6 if relative_mode else 0.35

    return {
        "deadzone": max(0.0, float(deadzone)),
        "smoothing": max(0.0, min(1.0, float(smoothing))),
        "edge_padding": max(0.0, min(0.45, float(edge_padding))),
        "max_step": max(0.01, min(1.0, float(max_step))),
        "gain": max(0.05, min(1.0, float(base_gain))),
        "gain_x": _cursor_axis_gain(args, axis="x", gaze_mode=gaze_mode, base_gain=float(base_gain)),
        "gain_y": _cursor_axis_gain(args, axis="y", gaze_mode=gaze_mode, base_gain=float(base_gain)),
    }


def _cursor_axis_gain(args: argparse.Namespace, *, axis: str, gaze_mode: str, base_gain: float) -> float:
    axis_value = getattr(args, f"cursor_gain_{axis}", None)
    if axis_value is not None:
        return max(0.05, min(1.0, float(axis_value)))
    if gaze_mode in {"cursor-relative", "cursor-relative-click", "cursor-assist", "cursor-assist-click"}:
        return max(0.05, min(1.0, float(base_gain)))
    if axis == "x":
        return max(0.05, min(1.0, float(base_gain) * 0.7))
    return max(0.05, min(1.0, float(base_gain) * 1.45))


def _effective_dwell_ms(settings, args: argparse.Namespace) -> int:
    raw_value = getattr(args, "dwell_ms", None)
    if raw_value is None:
        raw_value = settings.dwell_ms
    return max(0, int(raw_value))


def _effective_dwell_min_confidence(args: argparse.Namespace) -> float:
    raw_value = getattr(args, "dwell_min_confidence", None)
    if raw_value is None:
        raw_value = 0.65
    return max(0.0, min(1.0, float(raw_value)))


def _gaze_control_settings_payload(settings, args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "dwell_ms": _effective_dwell_ms(settings, args),
        "dwell_min_confidence": _effective_dwell_min_confidence(args),
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
        "cursor-relative": ActionName.FOCUS_TARGET,
        "cursor-relative-click": ActionName.CLICK_TARGET,
        "cursor-assist": ActionName.FOCUS_TARGET,
        "cursor-assist-click": ActionName.CLICK_TARGET,
        "cursor-snap": ActionName.FOCUS_TARGET,
        "cursor-snap-click": ActionName.CLICK_TARGET,
        "cursor-snap-right-click": ActionName.RIGHT_CLICK_TARGET,
        "cursor-snap-wink": ActionName.FOCUS_TARGET,
    }
    return mapping[value]


def _gaze_auto_confirm_actions(action: ActionName) -> set[ActionName]:
    if action in {ActionName.CLICK_TARGET, ActionName.RIGHT_CLICK_TARGET, ActionName.DOUBLE_CLICK_TARGET, ActionName.DRAG_TARGET}:
        return {action}
    return set()


def _profile_to_payload(profile) -> dict[str, Any] | None:
    if profile is None:
        return None
    payload: dict[str, Any] = {
        "x_scale": profile.x_scale,
        "y_scale": profile.y_scale,
        "x_offset": profile.x_offset,
        "y_offset": profile.y_offset,
        "fit_mode": getattr(profile, "fit_mode", "linear"),
    }
    if getattr(profile, "x_input_knots", ()):
        payload["x_input_knots"] = list(profile.x_input_knots)
        payload["x_output_knots"] = list(profile.x_output_knots)
    if getattr(profile, "y_input_knots", ()):
        payload["y_input_knots"] = list(profile.y_input_knots)
        payload["y_output_knots"] = list(profile.y_output_knots)
    return payload
