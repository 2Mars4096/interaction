from interaction.audio import PushToTalkTurnManager, ScriptedTranscriber, TurnPhase
from interaction.contracts import EnvironmentSnapshot
from interaction.feedback import VoiceLoopPhase
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime import VoiceCommandLoop


def test_scripted_transcriber_emits_partials_then_final() -> None:
    segments = ScriptedTranscriber().stream("open Safari")

    assert [segment.text for segment in segments] == ["open", "open Safari"]
    assert [segment.is_final for segment in segments] == [False, True]


def test_scripted_transcriber_accepts_custom_final_confidence() -> None:
    segments = ScriptedTranscriber().stream("open Safari", final_confidence=0.72)

    assert segments[-1].confidence == 0.72


def test_push_to_talk_turn_manager_cycles() -> None:
    manager = PushToTalkTurnManager()

    assert manager.phase == TurnPhase.IDLE
    assert manager.begin() == TurnPhase.LISTENING
    assert manager.end() == TurnPhase.COMPLETE
    assert manager.reset() == TurnPhase.IDLE


def test_voice_loop_open_app_executes_dry_run() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn("open Safari", EnvironmentSnapshot(active_app="Finder"))

    assert events[-1].phase == VoiceLoopPhase.IDLE
    assert any(event.result and event.result.details.get("commands") == [["open", "-a", "Safari"]] for event in events)


def test_voice_loop_target_dependent_command_clarifies_without_gaze() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn("click this", EnvironmentSnapshot(active_app="Safari"))

    assert any(event.phase == VoiceLoopPhase.RECOVERING and event.message for event in events)
    assert all(event.result is None for event in events)


def test_voice_loop_type_then_yes_executes_after_confirmation() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    first_events = loop.run_text_turn("type hello world", environment)
    second_events = loop.run_text_turn("yes", environment)

    assert any(event.phase == VoiceLoopPhase.CONFIRMING for event in first_events)
    assert any(event.result and event.result.details["commands"][0][0] == "osascript" for event in second_events)


def test_voice_loop_cancel_clears_pending_confirmation() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    first_events = loop.run_text_turn("type hello world", environment)
    second_events = loop.run_text_turn("cancel", environment)

    assert any(event.phase == VoiceLoopPhase.CONFIRMING for event in first_events)
    assert any(event.phase == VoiceLoopPhase.RECOVERING and event.message == "Pending action cancelled." for event in second_events)


def test_voice_loop_translate_reports_missing_provider() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn("translate hello to French", EnvironmentSnapshot(active_app="Notes"))

    assert any(
        event.phase == VoiceLoopPhase.RECOVERING
        and event.message == "Translation is recognized, but no local translation provider is configured yet."
        for event in events
    )
