import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.contracts import EnvironmentSnapshot, SessionMode
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime import VoiceCommandLoop
from interaction.ui.overlay import OverlayState
from interaction.ui.shell import TerminalShellRenderer, build_shell_view_state, infer_shell_surface
from interaction.ui.shell_gui import ShellDemoController


def test_infer_shell_surface_defaults_to_workspace() -> None:
    surface = infer_shell_surface(
        EnvironmentSnapshot(
            active_app="Finder",
            active_bundle_id="com.apple.finder",
        )
    )

    assert surface.value == "workspace"


def test_infer_shell_surface_prefers_browser_apps() -> None:
    surface = infer_shell_surface(
        EnvironmentSnapshot(
            active_app="Safari",
            active_bundle_id="com.apple.Safari",
        )
    )

    assert surface.value == "browser"


def test_infer_shell_surface_detects_notes() -> None:
    surface = infer_shell_surface(
        EnvironmentSnapshot(
            active_app="Notes",
            active_bundle_id="com.apple.Notes",
        )
    )

    assert surface.value == "notes"


def test_terminal_shell_renderer_includes_status_and_panel_commands() -> None:
    state = build_shell_view_state(
        source="voice",
        overlay_state=OverlayState(
            source="voice",
            phase="idle",
            message="Voice loop is idle.",
            transcript="tab 3",
        ),
        environment=EnvironmentSnapshot(active_app="Google Chrome", active_bundle_id="com.google.Chrome"),
        mode=SessionMode.COMMAND,
        active_scope="tab",
    )

    rendered = TerminalShellRenderer().render(state)

    assert "== Interaction Shell ==" in rendered
    assert "surface: Browser" in rendered
    assert "* tab 1 -> Switch to tab 1" in rendered
    assert "scope: tab" in rendered


def test_workspace_shell_state_includes_tabs_and_landmarks() -> None:
    state = build_shell_view_state(
        source="voice",
        overlay_state=OverlayState(
            source="voice",
            phase="idle",
            message="Voice loop is idle.",
        ),
        environment=EnvironmentSnapshot(active_app="Finder", active_bundle_id="com.apple.finder"),
        mode=SessionMode.COMMAND,
    )

    assert state.surface == "workspace"
    assert state.pointer_mode == "semantic"
    assert any(tab.active and tab.title == "Workspace" for tab in state.tabs)
    assert any(landmark.label == "Browser Lane" for landmark in state.landmarks)


def test_shell_smoke_returns_rendered_browser_shell(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "shell-smoke",
                "--runtime-dir",
                str(runtime_dir),
                "--surface",
                "browser",
                "--utterance",
                "tab 3",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["shell_state"]["surface"] == "browser"
    assert "== Interaction Shell ==" in payload["final_shell_rendered"]
    assert payload["voice_session_state"]["active_scope"] == "tab"


def test_shell_demo_controller_supports_precision_mode() -> None:
    controller = ShellDemoController(
        loop=VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True)),
        environment=EnvironmentSnapshot(
            active_app="Finder",
            active_bundle_id="com.apple.finder",
        ),
        mode=SessionMode.COMMAND,
        fallback_to_dictation=False,
        surface_override="workspace",
    )

    result = controller.run_utterance("precision mode")

    assert result.shell_state.pointer_mode == "precision"
    assert result.shell_state.selection_label in {"Precision Rail", "Cursor Anchor"}
    assert "pointer: precision" in result.shell_rendered


def test_shell_demo_controller_tracks_browser_navigation_state() -> None:
    controller = ShellDemoController(
        loop=VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True)),
        environment=EnvironmentSnapshot(
            active_app="Safari",
            active_bundle_id="com.apple.Safari",
            active_profile_id="safari_browser_family",
        ),
        mode=SessionMode.COMMAND,
        fallback_to_dictation=False,
        surface_override="browser",
    )

    result = controller.run_utterance("tab 3")

    assert result.shell_state.surface == "browser"
    assert result.shell_state.active_scope == "tab"
    assert "tab 1 -> Switch to tab 1" in result.shell_rendered


def test_shell_demo_controller_can_transition_to_notes_surface() -> None:
    controller = ShellDemoController(
        loop=VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True)),
        environment=EnvironmentSnapshot(
            active_app="Safari",
            active_bundle_id="com.apple.Safari",
            active_profile_id="safari_browser_family",
        ),
        mode=SessionMode.COMMAND,
        fallback_to_dictation=False,
        surface_override="browser",
    )

    result = controller.run_utterance("open Notes")

    assert result.shell_state.surface == "notes"
    assert result.shell_state.active_app == "Notes"


def test_shell_demo_controller_can_focus_workspace_landmark() -> None:
    controller = ShellDemoController(
        loop=VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True)),
        environment=EnvironmentSnapshot(
            active_app="Finder",
            active_bundle_id="com.apple.finder",
        ),
        mode=SessionMode.COMMAND,
        fallback_to_dictation=False,
        surface_override="workspace",
    )

    result = controller.run_utterance("focus command rail")

    assert result.shell_state.surface == "workspace"
    assert result.shell_state.selection_label == "Command Rail"


def test_shell_gui_demo_headless_returns_payload(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "shell-gui-demo",
                "--runtime-dir",
                str(runtime_dir),
                "--surface",
                "browser",
                "--utterance",
                "tab 3",
                "--headless",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["gui_demo"] is True
    assert payload["headless"] is True
    assert payload["shell_state"]["surface"] == "browser"


def test_shell_gui_demo_headless_defaults_to_workspace(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "shell-gui-demo",
                "--runtime-dir",
                str(runtime_dir),
                "--headless",
                "--utterance",
                "precision mode",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["shell_surface"] == "workspace"
    assert payload["shell_state"]["surface"] == "workspace"
    assert payload["shell_state"]["pointer_mode"] == "precision"


def test_shell_gui_demo_headless_tracks_surface_transition(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".interaction"
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "shell-gui-demo",
                "--runtime-dir",
                str(runtime_dir),
                "--surface",
                "browser",
                "--utterance",
                "open Notes",
                "--headless",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["environment"]["active_app"] == "Notes"
    assert payload["shell_state"]["surface"] == "notes"
    assert "surface: Notes" in payload["final_shell_rendered"]
