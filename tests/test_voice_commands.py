import pytest

from interaction.audio import ScriptedTranscriber
from interaction.contracts import ActionName, EnvironmentSnapshot, GazeObservation
from interaction.intent import VoiceDirectiveType, VoiceIntentInterpreter
from interaction.intent.voice import normalize_command
from interaction.runtime import MultimodalFusionLoop, VoiceCommandLoop
from interaction.vision import NormalizedScreenTarget


@pytest.mark.parametrize(
    ("utterance", "kind", "action", "arguments"),
    [
        ("Yes.", VoiceDirectiveType.CONFIRM, None, None),
        ("Please, stop!", VoiceDirectiveType.CANCEL, None, None),
        ("Could you please open Safari?", VoiceDirectiveType.PROPOSAL, ActionName.OPEN_APP, {"app_name": "Safari"}),
        ("scroll down.", VoiceDirectiveType.PROPOSAL, ActionName.SCROLL, {"direction": "down"}),
        ("open this.", VoiceDirectiveType.CLARIFY, None, None),
        ("Click!", VoiceDirectiveType.CLARIFY, None, None),
        ("double-click", VoiceDirectiveType.CLARIFY, None, None),
    ],
)
def test_recognized_command_formatting_and_deictic_precedence(utterance, kind, action, arguments):
    directive = VoiceIntentInterpreter().interpret(utterance)

    assert directive.kind == kind
    if action is not None:
        assert directive.proposal.action == action
        assert directive.proposal.arguments == arguments
    else:
        assert directive.proposal is None


def test_dictation_preserves_literal_content_through_transcriber_and_loop():
    content = "Hello,  WORLD!\nSecond line."
    utterance = f"Please type {content}"
    loop = VoiceCommandLoop()
    environment = EnvironmentSnapshot(active_app="Notes")

    assert ScriptedTranscriber().stream(utterance)[-1].text == utterance
    loop.run_text_turn(utterance, environment)

    assert loop.pending_decision.proposal.arguments["text"] == content
    confirmed = loop.run_text_turn("Yes.", environment)
    assert any(event.result and event.result.proposal.arguments["text"] == content for event in confirmed)


def test_dictation_that_looks_like_a_command_stays_literal():
    directive = VoiceIntentInterpreter().interpret("type Please stop listening!")

    assert directive.proposal.action == ActionName.TYPE_TEXT
    assert directive.proposal.arguments["text"] == "Please stop listening!"
    assert normalize_command("Could you please stop listening?") == "stop listening"


def _gaze_loop():
    loop = MultimodalFusionLoop()
    target = NormalizedScreenTarget(
        target_id="compose", label="Compose", role="button",
        x=0.4, y=0.4, width=0.2, height=0.2,
    ).to_grounded_target(confidence=0.95)
    loop.update_gaze_context(GazeObservation(confidence=0.95, x_norm=0.5, y_norm=0.5, fixation_ms=200), target)
    return loop


@pytest.mark.parametrize(
    ("utterance", "action"),
    [
        ("Click.", ActionName.CLICK_TARGET),
        ("Please right click!", ActionName.RIGHT_CLICK_TARGET),
        ("Double-click.", ActionName.DOUBLE_CLICK_TARGET),
        ("Can you open this?", ActionName.DOUBLE_CLICK_TARGET),
    ],
)
def test_short_commands_resolve_to_gaze_action_once(utterance, action):
    loop = _gaze_loop()
    environment = EnvironmentSnapshot(active_app="Mail")

    first = loop.run_voice_turn(utterance, environment)
    assert all(event.result is None for event in first)
    assert loop.pending_decision.proposal.action == action
    second = loop.run_voice_turn("yes.", environment)

    results = [event.result for event in second if event.result is not None]
    assert len(results) == 1
    assert results[0].proposal.action == action


@pytest.mark.parametrize("fused", [False, True])
def test_weak_recognition_does_not_execute_or_confirm_and_can_be_retried(fused):
    loop = _gaze_loop() if fused else VoiceCommandLoop()
    run = loop.run_voice_turn if fused else loop.run_text_turn
    environment = EnvironmentSnapshot(active_app="Notes")

    weak_command = run("open Safari", environment, final_confidence=0.01)
    assert all(event.result is None for event in weak_command)
    assert loop.pending_decision is None

    run("type hello", environment)
    pending = loop.pending_decision
    weak_confirmation = run("Yes.", environment, final_confidence=0.01)
    assert all(event.result is None for event in weak_confirmation)
    assert loop.pending_decision is pending

    clear_confirmation = run("Yes.", environment, final_confidence=0.95)
    assert len([event for event in clear_confirmation if event.result is not None]) == 1
    assert loop.pending_decision is None


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("replacement", ["open Safari", "unrecognized request"])
def test_new_command_supersedes_pending_typing(fused, replacement):
    loop = _gaze_loop() if fused else VoiceCommandLoop()
    run = loop.run_voice_turn if fused else loop.run_text_turn
    environment = EnvironmentSnapshot(active_app="Notes")

    run("type old request", environment)
    run(replacement, environment)
    confirmation = run("yes", environment)

    assert loop.pending_decision is None
    assert all(event.result is None for event in confirmation)


@pytest.mark.parametrize("fused", [False, True])
def test_cancellation_is_available_with_weak_recognition(fused):
    loop = _gaze_loop() if fused else VoiceCommandLoop()
    run = loop.run_voice_turn if fused else loop.run_text_turn
    environment = EnvironmentSnapshot(active_app="Notes")

    run("type hello", environment)
    run("Please stop listening.", environment, final_confidence=0.01)

    assert loop.pending_decision is None


def test_weak_gaze_command_is_not_rescued_by_high_gaze_confidence():
    loop = _gaze_loop()

    events = loop.run_voice_turn("click", EnvironmentSnapshot(active_app="Mail"), final_confidence=0.01)

    assert all(event.result is None for event in events)
    assert loop.pending_decision is None
