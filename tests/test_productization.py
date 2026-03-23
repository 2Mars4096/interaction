import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.contracts import ActionName, ActionProposal, BrokerDecision, BrokerDecisionType, RiskLevel
from interaction.feedback import FusionFeedbackEvent, FusionLoopPhase
from interaction.persistence import JsonStateStore, RuntimePaths, UserSettings
from interaction.platform import MacOSPlatformAdapter
from interaction.session import SessionLogger, SessionReplay
from interaction.ui import ConsoleOverlayRenderer, OverlayController
from interaction.vision import CalibrationProfile


def test_json_state_store_round_trips_settings_and_calibration(tmp_path: Path) -> None:
    store = JsonStateStore(RuntimePaths(tmp_path / ".interaction"))
    settings = UserSettings(dwell_ms=820, gaze_context_window_ms=1900)
    profile = CalibrationProfile(x_scale=1.1, y_scale=0.95, x_offset=0.02, y_offset=-0.01)

    settings_path = store.save_settings(settings)
    profile_path = store.save_calibration_profile(profile, name="default")

    assert settings_path.exists()
    assert profile_path.exists()
    assert store.load_settings() == settings
    assert store.load_calibration_profile("default") == profile


def test_session_logger_and_replay_round_trip(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    logger = SessionLogger(log_path)
    event = FusionFeedbackEvent(
        phase=FusionLoopPhase.RECOVERING,
        message="I am not confident enough to act on that yet.",
        transcript="open this",
    )

    logger.record_event("fusion", event)
    records = SessionReplay.load(log_path)

    assert len(records) == 1
    assert records[0].source == "fusion"
    assert records[0].phase == "recovering"
    assert records[0].transcript == "open this"


def test_overlay_controller_tracks_pending_confirmation() -> None:
    proposal = ActionProposal(
        action=ActionName.CLICK_TARGET,
        arguments={"target_ref": "compose"},
        confidence=0.9,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Pointer action requires confirmation.",
    )
    decision = BrokerDecision(
        decision=BrokerDecisionType.CONFIRM,
        reason="Need confirmation.",
        proposal=proposal,
        confirmation_prompt="Confirm clicking the grounded target?",
    )
    event = FusionFeedbackEvent(
        phase=FusionLoopPhase.CONFIRMING,
        message="Confirm clicking the grounded target?",
        transcript="click this",
        decision=decision,
    )

    overlay = OverlayController()
    state = overlay.apply_event("fusion", event)
    rendered = ConsoleOverlayRenderer().render(state)

    assert state.pending_confirmation
    assert state.decision == "confirm"
    assert "pending_confirmation: yes" in rendered


def test_macos_adapter_describes_capabilities() -> None:
    capabilities = MacOSPlatformAdapter(dry_run=True).describe_capabilities()

    assert capabilities.platform_name == "macos"
    assert ActionName.OPEN_APP.value in capabilities.supported_actions
    assert ActionName.CLICK_TARGET.value in capabilities.coordinate_actions


def test_app_fusion_smoke_and_replay_create_repeatable_outputs(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(["fusion-smoke", "--runtime-dir", str(runtime_dir), "--session-name", "phase5-test"])
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    session_log = Path(payload["session_log"])
    assert session_log.exists()
    assert payload["metrics"]["executions"] >= 1
    assert payload["final_overlay"]["phase"] == "idle"

    replay_stdout = io.StringIO()
    with redirect_stdout(replay_stdout):
        replay_code = main(["replay", "--runtime-dir", str(runtime_dir), "--session-log", str(session_log)])
    replay_payload = json.loads(replay_stdout.getvalue())

    assert replay_code == 0
    assert replay_payload["records"]
    assert replay_payload["final_overlay"]["source"] == "fusion"
