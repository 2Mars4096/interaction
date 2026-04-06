import importlib

from interaction.contracts import (
    ActionName,
    ActionProposal,
    DesktopTargetCandidate,
    EnvironmentSnapshot,
    RiskLevel,
    ScreenPoint,
    TargetCandidateSource,
)
from interaction.control import CommandBroker
from interaction.platform import MacOSPlatformAdapter, TargetCandidateQuery, capture_frontmost_environment


def test_macos_open_app_dry_run_returns_command_plan() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.OPEN_APP,
        arguments={"app_name": "Safari"},
        confidence=0.95,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Opening a named app is a bounded action.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Finder"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"] == [["open", "-a", "Safari"]]


def test_macos_scroll_down_dry_run_maps_to_page_down_key() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.SCROLL,
        arguments={"direction": "down"},
        confidence=0.9,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Scrolling is reversible navigation.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][0] == "osascript"


def test_macos_click_target_without_coordinates_uses_dry_run_placeholder() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.CLICK_TARGET,
        arguments={"target_ref": "target_1"},
        confidence=0.87,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Click requires confirmation in MVP.",
    )

    decision = broker.confirm(broker.decide(proposal))
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][0] == "osascript"


def test_macos_highlight_target_uses_notification_placeholder() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.HIGHLIGHT_TARGET,
        arguments={"target_label": "Compose button"},
        confidence=0.82,
        risk=RiskLevel.L0,
        requires_confirmation=False,
        rationale="Highlighting is a no-side-effect placeholder action.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][0] == "osascript"


def test_macos_type_text_can_delete_recent_characters_before_typing() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.TYPE_TEXT,
        arguments={"delete_characters": 5, "text": "hello"},
        confidence=0.9,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Replacing recent dictated text requires deleting then typing.",
    )

    decision = broker.confirm(broker.decide(proposal))
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Notes"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][0] == "osascript"
    assert "repeat 5 times" in result.details["commands"][0][2]
    assert 'keystroke "hello"' in result.details["commands"][0][2]


def test_macos_focus_target_with_point_uses_runtime_module() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.FOCUS_TARGET,
        arguments={"screen_point": {"x": 320, "y": 240}},
        confidence=0.8,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Focusing a screen point is bounded navigation.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][2] == "interaction.platform.macos_runtime"


def test_macos_focus_target_with_normalized_point_uses_normalized_runtime_command() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.FOCUS_TARGET,
        arguments={"normalized_point": {"x": 0.52, "y": 0.27}},
        confidence=0.8,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Focusing a normalized screen point is bounded navigation.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][2] == "interaction.platform.macos_runtime"
    assert result.details["commands"][0][3] == "move-normalized"


def test_macos_click_target_with_coordinates_is_plannable_after_confirmation() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.CLICK_TARGET,
        arguments={"target_ref": "target_1", "screen_point": {"x": 500, "y": 300}},
        confidence=0.87,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Click requires confirmation in MVP.",
    )

    decision = broker.confirm(broker.decide(proposal))
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][2] == "interaction.platform.macos_runtime"


def test_macos_right_click_target_with_normalized_point_is_plannable_after_confirmation() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.RIGHT_CLICK_TARGET,
        arguments={"target_ref": "target_1", "normalized_point": {"x": 0.41, "y": 0.63}},
        confidence=0.87,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Right click requires confirmation in MVP.",
    )

    decision = broker.confirm(broker.decide(proposal))
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][2] == "interaction.platform.macos_runtime"
    assert result.details["commands"][0][3] == "right-click-normalized"


def test_macos_drag_target_with_normalized_points_is_plannable_after_confirmation() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.DRAG_TARGET,
        arguments={
            "start_target_ref": "target_1",
            "target_ref": "target_2",
            "start_normalized_point": {"x": 0.22, "y": 0.31},
            "end_normalized_point": {"x": 0.74, "y": 0.69},
        },
        confidence=0.85,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Drag requires confirmation in MVP.",
    )

    decision = broker.confirm(broker.decide(proposal))
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["commands"][0][2] == "interaction.platform.macos_runtime"
    assert result.details["commands"][0][3] == "drag-normalized"


def test_macos_execute_uses_in_process_runtime_for_normalized_pointer_actions(monkeypatch) -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.CLICK_TARGET,
        arguments={"normalized_point": {"x": 0.4, "y": 0.6}},
        confidence=0.9,
        risk=RiskLevel.L2,
        requires_confirmation=True,
        rationale="Click requires confirmation in MVP.",
    )
    decision = broker.confirm(broker.decide(proposal))
    request = broker.build_execution_request(decision, EnvironmentSnapshot(active_app="Safari"))

    calls: list[tuple[float, float]] = []
    runtime = importlib.import_module("interaction.platform.macos_runtime")
    monkeypatch.setattr(runtime, "click_normalized", lambda x, y: calls.append((x, y)))

    result = MacOSPlatformAdapter(dry_run=False).execute(request)

    assert result.status.value == "success"
    assert result.details["dry_run"] is False
    assert calls == [(0.4, 0.6)]


def test_macos_describe_capabilities_reports_candidate_collection() -> None:
    capabilities = MacOSPlatformAdapter(dry_run=True).describe_capabilities()

    assert capabilities.platform_name == "macos"
    assert capabilities.candidate_collection is True


def test_macos_collect_target_candidates_uses_injected_collector() -> None:
    captured_queries: list[TargetCandidateQuery] = []

    def fake_collector(query: TargetCandidateQuery) -> tuple[DesktopTargetCandidate, ...]:
        captured_queries.append(query)
        return (
            DesktopTargetCandidate(
                candidate_id="ax-1-button",
                source=TargetCandidateSource.ACCESSIBILITY,
                display_label="Bold",
                label="Bold",
                role="AXButton",
                screen_point=ScreenPoint(x=query.screen_point.x, y=query.screen_point.y),
                enabled=True,
                invocable=True,
                available_actions=["AXPress"],
                app_name="TextEdit",
                pid=123,
            ),
        )

    adapter = MacOSPlatformAdapter(dry_run=True, target_candidate_collector=fake_collector)
    query = TargetCandidateQuery(
        screen_point=ScreenPoint(x=640.0, y=400.0),
        radius_px=36.0,
        limit=4,
        active_app="TextEdit",
    )

    candidates = adapter.collect_target_candidates(query)

    assert len(candidates) == 1
    assert captured_queries == [query]
    assert candidates[0].candidate_id == "ax-1-button"
    assert candidates[0].available_actions == ["AXPress"]


def test_macos_collect_target_candidates_returns_empty_tuple_on_collector_failure() -> None:
    def broken_collector(_query: TargetCandidateQuery) -> tuple[DesktopTargetCandidate, ...]:
        raise RuntimeError("boom")

    adapter = MacOSPlatformAdapter(dry_run=True, target_candidate_collector=broken_collector)

    candidates = adapter.collect_target_candidates(
        TargetCandidateQuery(screen_point=ScreenPoint(x=100.0, y=200.0))
    )

    assert candidates == ()


def test_macos_capabilities_advertise_structured_navigation_profiles() -> None:
    capabilities = MacOSPlatformAdapter(dry_run=True).describe_capabilities()

    assert ActionName.NAVIGATE_STRUCTURE.value in capabilities.supported_actions
    assert "tab" in capabilities.structured_navigation_scopes
    assert "chrome_chromium_family" in capabilities.app_profiles


def test_macos_structured_navigation_resolves_browser_tab_index_profile() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.NAVIGATE_STRUCTURE,
        arguments={"scope": "tab", "command": "index", "index": 3},
        confidence=0.92,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Tab index is bounded structured navigation.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(
        decision,
        EnvironmentSnapshot(
            active_app="Google Chrome",
            active_bundle_id="com.google.Chrome",
        ),
    )
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["profile_id"] == "chrome_chromium_family"
    assert result.details["navigation"]["scope"] == "tab"
    assert result.details["navigation"]["index"] == 3
    assert result.details["commands"][0][0] == "osascript"


def test_macos_structured_navigation_falls_back_to_global_profile() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.NAVIGATE_STRUCTURE,
        arguments={"scope": "space", "command": "left"},
        confidence=0.89,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Space left uses the global macOS navigation profile.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(
        decision,
        EnvironmentSnapshot(
            active_app="Mail",
            active_bundle_id="com.apple.mail",
        ),
    )
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "success"
    assert result.details["profile_id"] == "global_macos"
    assert result.details["fallback_profile_id"] == "mail_family"


def test_macos_structured_navigation_blocks_unsupported_profile_action() -> None:
    broker = CommandBroker()
    proposal = ActionProposal(
        action=ActionName.NAVIGATE_STRUCTURE,
        arguments={"scope": "pane", "command": "right"},
        confidence=0.88,
        risk=RiskLevel.L1,
        requires_confirmation=False,
        rationale="Mail does not support pane navigation in the first profile slice.",
    )

    decision = broker.decide(proposal)
    request = broker.build_execution_request(
        decision,
        EnvironmentSnapshot(
            active_app="Mail",
            active_bundle_id="com.apple.mail",
        ),
    )
    result = MacOSPlatformAdapter(dry_run=True).execute(request)

    assert result.status.value == "blocked"
    assert result.details["profile_id"] == "mail_family"
    assert "not supported" in result.message


def test_capture_frontmost_environment_parses_active_app_bundle_and_window() -> None:
    environment = capture_frontmost_environment(
        lambda _argv: "Finder\x1fcom.apple.finder\x1fDesktop"
    )

    assert environment.active_app == "Finder"
    assert environment.active_bundle_id == "com.apple.finder"
    assert environment.active_window_title == "Desktop"
