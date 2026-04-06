from interaction.audio import PushToTalkTurnManager, ScriptedTranscriber, TurnPhase
from interaction.contracts import EnvironmentSnapshot, SessionMode
from interaction.feedback import VoiceLoopPhase
from interaction.intent import GemmaIntentInterpreter, VoiceInterpretationRouter
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime import VoiceCommandLoop


class FakeTranslationBackend:
    def interpret(self, request):
        return {
            "route": "translation",
            "confidence": 0.93,
            "translated_text": "bonjour",
            "target_language": "French",
            "rationale": f'Normalized translation for "{request.transcript}".',
        }


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
        and event.message == "Translation is unavailable because no Gemma language backend is configured."
        for event in events
    )


def test_voice_loop_structured_navigation_fast_path_executes() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Google Chrome", active_bundle_id="com.google.Chrome")

    events = loop.run_text_turn("tab 3", environment)

    assert any(event.result and event.result.details["navigation"]["scope"] == "tab" for event in events)
    assert any(event.result and event.result.details["profile_id"] == "chrome_chromium_family" for event in events)


def test_voice_loop_active_scope_follow_up_reuses_previous_navigation_scope() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Google Chrome", active_bundle_id="com.google.Chrome")

    first_events = loop.run_text_turn("tab 3", environment)
    second_events = loop.run_text_turn("next", environment)

    assert any(event.result and event.result.details["navigation"]["index"] == 3 for event in first_events)
    assert any(
        event.result
        and event.result.details["navigation"]["scope"] == "tab"
        and event.result.details["navigation"]["command"] == "next"
        for event in second_events
    )


def test_voice_loop_translation_backend_routes_to_type_text_after_confirmation() -> None:
    router = VoiceInterpretationRouter(
        gemma_interpreter=GemmaIntentInterpreter(FakeTranslationBackend()),
    )
    loop = VoiceCommandLoop(
        adapter=MacOSPlatformAdapter(dry_run=True),
        router=router,
    )
    environment = EnvironmentSnapshot(active_app="Notes")

    first_events = loop.run_text_turn("translate hello to French", environment)
    second_events = loop.run_text_turn("yes", environment)

    assert any(event.phase == VoiceLoopPhase.CONFIRMING for event in first_events)
    assert any(
        event.result
        and event.result.details["commands"][0][0] == "osascript"
        and "bonjour" in event.result.details["commands"][0][2]
        for event in second_events
    )


def test_voice_loop_dictation_mode_types_literal_text() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "hello there",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    assert any(
        event.phase == VoiceLoopPhase.EXECUTING
        and event.result
        and event.result.details["commands"][0][0] == "osascript"
        for event in events
    )


def test_voice_loop_hybrid_mode_falls_back_to_dictation_for_unmatched_text() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "hello there",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.COMMAND,
        fallback_to_dictation=True,
    )

    assert any(
        event.phase == VoiceLoopPhase.INTERPRETING
        and event.message == "Hybrid lane did not match a bounded command and fell back to dictation."
        for event in events
    )
    assert any(
        event.phase == VoiceLoopPhase.EXECUTING
        and event.result
        and event.result.details["commands"][0][0] == "osascript"
        for event in events
    )


def test_voice_loop_dictation_mode_removes_immediate_duplicate_words() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "hello hello there",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert "Hello there" in str(typed_commands[-1])


def test_voice_loop_dictation_mode_formats_spoken_punctuation() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "hello comma world period",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert "Hello, world." in str(typed_commands[-1])


def test_voice_loop_dictation_mode_formats_spoken_line_breaks_and_capitalization() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "hello world new line next sentence period",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert "Hello world\\nNext sentence." in str(typed_commands[-1])


def test_voice_loop_dictation_mode_strips_leading_fillers() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "um you know hello there",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert "Hello there" in str(typed_commands[-1])


def test_voice_loop_dictation_mode_collapses_short_false_start_phrase_repetition() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "i think i think we should go",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert "I think we should go" in str(typed_commands[-1])


def test_voice_loop_dictation_mode_formats_parentheses_and_new_sentence() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "open paren hello close paren new sentence goodbye period",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert "(Hello). Goodbye." in str(typed_commands[-1])


def test_voice_loop_dictation_mode_can_delete_last_clause() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there", environment, mode=SessionMode.DICTATION)
    events = loop.run_text_turn("scratch that", environment, mode=SessionMode.DICTATION)

    delete_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert delete_commands
    assert "repeat 11 times" in str(delete_commands[-1])


def test_voice_loop_dictation_mode_can_replace_last_clause() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there", environment, mode=SessionMode.DICTATION)
    events = loop.run_text_turn("replace that with goodbye", environment, mode=SessionMode.DICTATION)

    replace_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert replace_commands
    assert "repeat 11 times" in str(replace_commands[-1])
    assert 'keystroke "Goodbye"' in str(replace_commands[-1])


def test_voice_loop_dictation_mode_tracks_multiple_clauses_for_repeated_scratch_that() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there", environment, mode=SessionMode.DICTATION)
    loop.run_text_turn("goodbye now", environment, mode=SessionMode.DICTATION)

    first_delete_events = loop.run_text_turn("scratch that", environment, mode=SessionMode.DICTATION)
    second_delete_events = loop.run_text_turn("scratch that", environment, mode=SessionMode.DICTATION)

    first_delete_commands = [
        event.result.details["commands"]
        for event in first_delete_events
        if event.result is not None and "commands" in event.result.details
    ]
    second_delete_commands = [
        event.result.details["commands"]
        for event in second_delete_events
        if event.result is not None and "commands" in event.result.details
    ]
    assert first_delete_commands
    assert second_delete_commands
    assert "repeat 12 times" in str(first_delete_commands[-1])
    assert "repeat 11 times" in str(second_delete_commands[-1])
    assert loop.dictated_clause_history == []


def test_voice_loop_dictation_mode_replace_preserves_prior_clause_history() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("alpha", environment, mode=SessionMode.DICTATION)
    loop.run_text_turn("beta", environment, mode=SessionMode.DICTATION)

    replace_events = loop.run_text_turn("replace that with gamma", environment, mode=SessionMode.DICTATION)
    delete_events = loop.run_text_turn("scratch that", environment, mode=SessionMode.DICTATION)

    replace_commands = [
        event.result.details["commands"]
        for event in replace_events
        if event.result is not None and "commands" in event.result.details
    ]
    delete_commands = [
        event.result.details["commands"]
        for event in delete_events
        if event.result is not None and "commands" in event.result.details
    ]
    assert replace_commands
    assert delete_commands
    assert "repeat 5 times" in str(replace_commands[-1])
    assert 'keystroke " gamma"' in str(replace_commands[-1])
    assert "repeat 6 times" in str(delete_commands[-1])
    assert loop.dictated_clause_history == ["Alpha"]


def test_voice_loop_dictation_mode_continues_mid_sentence_across_clauses() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there", environment, mode=SessionMode.DICTATION)
    events = loop.run_text_turn("how are you", environment, mode=SessionMode.DICTATION)

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert 'keystroke " how are you"' in str(typed_commands[-1])
    assert loop.dictated_clause_history == ["Hello there", " how are you"]


def test_voice_loop_dictation_mode_starts_new_sentence_after_terminal_punctuation() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there period", environment, mode=SessionMode.DICTATION)
    events = loop.run_text_turn("how are you", environment, mode=SessionMode.DICTATION)

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert 'keystroke " How are you"' in str(typed_commands[-1])
    assert loop.dictated_clause_history == ["Hello there.", " How are you"]


def test_voice_loop_dictation_mode_self_correction_replaces_previous_clause() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there", environment, mode=SessionMode.DICTATION)
    events = loop.run_text_turn("i mean goodbye", environment, mode=SessionMode.DICTATION)

    replace_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert replace_commands
    assert "repeat 11 times" in str(replace_commands[-1])
    assert 'keystroke "Goodbye"' in str(replace_commands[-1])
    assert loop.dictated_clause_history == ["Goodbye"]


def test_voice_loop_dictation_mode_self_correction_phrase_is_literal_without_history() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))

    events = loop.run_text_turn(
        "i mean hello there",
        EnvironmentSnapshot(active_app="Notes"),
        mode=SessionMode.DICTATION,
    )

    typed_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert typed_commands
    assert 'keystroke "I mean hello there"' in str(typed_commands[-1])
    assert loop.dictated_clause_history == ["I mean hello there"]


def test_voice_loop_dictation_mode_ignores_duplicate_clause() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there", environment, mode=SessionMode.DICTATION)
    events = loop.run_text_turn("hello there", environment, mode=SessionMode.DICTATION)

    assert any(
        event.phase == VoiceLoopPhase.INTERPRETING
        and event.message == "Dictation lane ignored a duplicated clause."
        for event in events
    )
    assert all(event.result is None for event in events)
    assert loop.dictated_clause_history == ["Hello there"]


def test_voice_loop_dictation_mode_can_delete_last_sentence_span() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there period", environment, mode=SessionMode.DICTATION)
    loop.run_text_turn("how are you", environment, mode=SessionMode.DICTATION)
    loop.run_text_turn("today question mark", environment, mode=SessionMode.DICTATION)

    events = loop.run_text_turn("delete last sentence", environment, mode=SessionMode.DICTATION)

    delete_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert delete_commands
    assert "repeat 19 times" in str(delete_commands[-1])
    assert loop.dictated_clause_history == ["Hello there."]


def test_voice_loop_dictation_mode_can_replace_last_sentence_span() -> None:
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    environment = EnvironmentSnapshot(active_app="Notes")

    loop.run_text_turn("hello there period", environment, mode=SessionMode.DICTATION)
    loop.run_text_turn("how are you", environment, mode=SessionMode.DICTATION)
    loop.run_text_turn("today question mark", environment, mode=SessionMode.DICTATION)

    events = loop.run_text_turn("replace last sentence with goodbye now", environment, mode=SessionMode.DICTATION)

    replace_commands = [
        event.result.details["commands"]
        for event in events
        if event.result is not None and "commands" in event.result.details
    ]
    assert replace_commands
    assert "repeat 19 times" in str(replace_commands[-1])
    assert 'keystroke " Goodbye now"' in str(replace_commands[-1])
    assert loop.dictated_clause_history == ["Hello there.", " Goodbye now"]
