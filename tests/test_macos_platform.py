from interaction.contracts import ActionName, ActionProposal, EnvironmentSnapshot, RiskLevel
from interaction.control import CommandBroker
from interaction.platform import MacOSPlatformAdapter


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
