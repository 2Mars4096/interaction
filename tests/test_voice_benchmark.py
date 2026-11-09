import io
import json
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from interaction.app import main
from interaction.audio import LiveSpeechCapture
from interaction.runtime import (
    VoiceBenchmarkBudget,
    compare_voice_benchmark_summaries,
    evaluate_voice_benchmark_budget,
    generate_voice_fixture_corpus,
    render_voice_benchmark_comparison,
    render_voice_benchmark_budget_summary,
    render_voice_benchmark_delta_summary,
    render_voice_benchmark_summary,
    summarize_voice_benchmark_comparison,
    summarize_voice_benchmark_results,
)


def test_voice_benchmark_cli_aggregates_fixture_results(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, _paths, *, audio_path: str):
            self.audio_path = audio_path

        def capture_turn(
            self,
            *,
            duration_s: float = 0.0,
            locale: str = "en-US",
            recording_path: str | None = None,
        ) -> LiveSpeechCapture:
            del duration_s, recording_path
            audio_name = Path(self.audio_path).name
            if "open" in audio_name:
                transcript = "open Safari"
            else:
                transcript = "click this"
            return LiveSpeechCapture(
                transcript=transcript,
                confidence=0.93,
                locale=locale,
                duration_s=1.0,
                provider="fake_file",
                used_on_device=True,
                permission_state="granted",
                recording_path=self.audio_path,
            )

    monkeypatch.setattr("interaction.app.MacOSFileSpeechProvider", FakeProvider)

    manifest_dir = tmp_path / "fixtures" / "manifests"
    audio_dir = tmp_path / "fixtures" / "audio"
    manifest_dir.mkdir(parents=True)
    (audio_dir / "open").mkdir(parents=True)
    (audio_dir / "click").mkdir(parents=True)
    (audio_dir / "open" / "open.wav").write_bytes(b"RIFF")
    (audio_dir / "click" / "click.wav").write_bytes(b"RIFF")

    (manifest_dir / "canonical_open_safari.json").write_text(
        json.dumps(
            {
                "fixture_id": "canonical_open_safari",
                "category": "canonical_command",
                "prompt_text": "open Safari",
                "lane": "command",
                "reference_transcript": "open Safari",
                "expected_route": "command",
                "expected_action": "open_app",
                "expected_arguments": {"app_name": "Safari"},
                "expected_risk": "L1",
                "expected_broker_decision": "allow",
                "tags": ["canonical"],
                "variants": [
                    {
                        "variant_id": "say_alex_default",
                        "audio_path": "../audio/open/open.wav",
                        "source_engine": "macos_say",
                        "voice_id": "Alex",
                        "speaking_rate": 175,
                    }
                ],
            }
        )
    )
    (manifest_dir / "deictic_click.json").write_text(
        json.dumps(
            {
                "fixture_id": "deictic_click",
                "category": "deictic_command",
                "prompt_text": "click this",
                "lane": "command",
                "reference_transcript": "click this",
                "expected_route": "clarify",
                "expected_message_contains": "gaze grounding",
                "tags": ["deictic"],
                "variants": [
                    {
                        "variant_id": "say_alex_default",
                        "audio_path": "../audio/click/click.wav",
                        "source_engine": "macos_say",
                        "voice_id": "Alex",
                        "speaking_rate": 175,
                    }
                ],
            }
        )
    )

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "voice-benchmark",
                "--runtime-dir",
                str(tmp_path / ".interaction"),
                "--fixtures",
                str(manifest_dir),
                "--active-app",
                "Finder",
            ]
        )
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["signal_only"] is True
    assert payload["summary"]["fixtures_total"] == 2
    assert payload["summary"]["passed"] == 2
    assert payload["summary"]["failed"] == 0
    assert payload["summary_text"].startswith("Voice Benchmark Summary")
    assert "Fixtures   2/2 passed (100.0%) | failed 0" in payload["summary_text"]
    assert "Phases     speech " in payload["summary_text"]
    assert payload["summary"]["by_category"] == {
        "canonical_command": 1,
        "deictic_command": 1,
    }
    assert payload["summary"]["metrics"]["latency_breakdown_ms"]["transcription_ms"]["samples"] == 2
    assert payload["summary"]["metrics"]["latency_breakdown_ms"]["signal_ms"]["samples"] == 2
    assert {result["fixture_id"] for result in payload["results"]} == {
        "canonical_open_safari",
        "deictic_click",
    }
    assert all(result["audio_capture"]["status"] == "success" for result in payload["results"])
    assert all(result["evaluation"]["passed"] is True for result in payload["results"])
    assert all(isinstance(result["latency_breakdown_ms"], dict) for result in payload["results"])


def test_generate_voice_fixture_corpus_creates_audio_and_metadata(monkeypatch, tmp_path: Path) -> None:
    manifest_dir = tmp_path / "fixtures" / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "canonical_open_safari.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fixture_id": "canonical_open_safari",
                "category": "canonical_command",
                "prompt_text": "open Safari",
                "lane": "command",
                "reference_transcript": "open Safari",
                "variants": [
                    {
                        "variant_id": "say_alex_default",
                        "audio_path": "../audio/canonical_open_safari/say_alex_default.wav",
                        "source_engine": "macos_say",
                        "voice_id": "Alex",
                        "speaking_rate": 175,
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(
        "interaction.runtime.voice_fixture_generation.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        binary = Path(command[0]).name
        if binary == "say":
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_bytes(b"FORM" + (b"0" * 96))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if binary == "ffmpeg":
            output_path = Path(command[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF" + (b"1" * 128))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if binary == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="1.25\n", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    output_root = tmp_path / "generated"
    first = generate_voice_fixture_corpus(
        manifest_dir,
        output_root=output_root,
        command_runner=fake_runner,
    )
    second = generate_voice_fixture_corpus(
        manifest_dir,
        output_root=output_root,
        command_runner=fake_runner,
    )

    assert first[0]["status"] == "generated"
    assert second[0]["status"] == "skipped"
    assert second[0]["reason"] == "exists"

    audio_path = output_root / "audio" / "canonical_open_safari" / "say_alex_default.wav"
    metadata_path = output_root / "audio" / "canonical_open_safari" / "say_alex_default.wav.json"
    assert audio_path.is_file()
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["fixture_id"] == "canonical_open_safari"
    assert metadata["voice_id"] == "Alex"
    assert metadata["duration_s"] == 1.25


def test_generate_voice_fixture_corpus_supports_piper(monkeypatch, tmp_path: Path) -> None:
    manifest_dir = tmp_path / "fixtures" / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "canonical_open_safari.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fixture_id": "canonical_open_safari",
                "category": "canonical_command",
                "prompt_text": "open Safari",
                "lane": "command",
                "reference_transcript": "open Safari",
                "variants": [
                    {
                        "variant_id": "piper_lessac_low",
                        "audio_path": "../audio/canonical_open_safari/piper_lessac_low.wav",
                        "source_engine": "piper",
                        "voice_id": "en_US-lessac-low",
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(
        "interaction.runtime.voice_fixture_generation.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

    fake_piper_bin = tmp_path / "bin" / "piper"
    fake_piper_python = tmp_path / "bin" / "python"
    fake_piper_bin.parent.mkdir(parents=True)
    fake_piper_bin.write_text("")
    fake_piper_python.write_text("")
    monkeypatch.setenv("INTERACTION_PIPER_BIN", str(fake_piper_bin))
    monkeypatch.setenv("INTERACTION_PIPER_PYTHON", str(fake_piper_python))
    monkeypatch.setenv("INTERACTION_PIPER_VOICES_DIR", str(tmp_path / "voices"))

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == [str(fake_piper_python), "-m", "piper.download_voices"]:
            download_dir = Path(command[command.index("--download-dir") + 1])
            voice_id = command[3]
            download_dir.mkdir(parents=True, exist_ok=True)
            (download_dir / f"{voice_id}.onnx").write_bytes(b"onnx")
            (download_dir / f"{voice_id}.onnx.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == str(fake_piper_bin):
            output_path = Path(command[command.index("-f") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF" + (b"2" * 256))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if Path(command[0]).name == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="1.50\n", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    output_root = tmp_path / "generated"
    results = generate_voice_fixture_corpus(
        manifest_dir,
        output_root=output_root,
        command_runner=fake_runner,
    )

    assert results[0]["status"] == "generated"
    assert results[0]["source_engine"] == "piper"
    assert results[0]["voice_id"] == "en_US-lessac-low"

    audio_path = output_root / "audio" / "canonical_open_safari" / "piper_lessac_low.wav"
    metadata_path = output_root / "audio" / "canonical_open_safari" / "piper_lessac_low.wav.json"
    assert audio_path.is_file()
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["duration_s"] == 1.5
    assert metadata["source_engine"] == "piper"


def test_generate_voice_fixture_corpus_supports_noise_variants(monkeypatch, tmp_path: Path) -> None:
    manifest_dir = tmp_path / "fixtures" / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "canonical_open_safari_noise.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fixture_id": "canonical_open_safari_noise",
                "category": "canonical_command_noise",
                "prompt_text": "open Safari",
                "lane": "command",
                "reference_transcript": "open Safari",
                "variants": [
                    {
                        "variant_id": "piper_lessac_pink_noise",
                        "audio_path": "../audio/canonical_open_safari_noise/piper_lessac_pink_noise.wav",
                        "source_engine": "piper",
                        "voice_id": "en_US-lessac-low",
                        "background_noise": "pink",
                        "noise_level_db": -28.0,
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(
        "interaction.runtime.voice_fixture_generation.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

    fake_piper_bin = tmp_path / "bin" / "piper"
    fake_piper_python = tmp_path / "bin" / "python"
    fake_piper_bin.parent.mkdir(parents=True)
    fake_piper_bin.write_text("")
    fake_piper_python.write_text("")
    monkeypatch.setenv("INTERACTION_PIPER_BIN", str(fake_piper_bin))
    monkeypatch.setenv("INTERACTION_PIPER_PYTHON", str(fake_piper_python))
    monkeypatch.setenv("INTERACTION_PIPER_VOICES_DIR", str(tmp_path / "voices"))

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == [str(fake_piper_python), "-m", "piper.download_voices"]:
            download_dir = Path(command[command.index("--download-dir") + 1])
            voice_id = command[3]
            download_dir.mkdir(parents=True, exist_ok=True)
            (download_dir / f"{voice_id}.onnx").write_bytes(b"onnx")
            (download_dir / f"{voice_id}.onnx.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == str(fake_piper_bin):
            output_path = Path(command[command.index("-f") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF" + (b"2" * 256))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if Path(command[0]).name == "ffmpeg":
            output_path = Path(command[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF" + (b"3" * 256))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if Path(command[0]).name == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="1.50\n", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    output_root = tmp_path / "generated"
    results = generate_voice_fixture_corpus(
        manifest_dir,
        output_root=output_root,
        command_runner=fake_runner,
    )

    assert results[0]["status"] == "generated"
    assert results[0]["background_noise"] == "pink"
    assert results[0]["noise_level_db"] == -28.0

    audio_path = output_root / "audio" / "canonical_open_safari_noise" / "piper_lessac_pink_noise.wav"
    metadata_path = output_root / "audio" / "canonical_open_safari_noise" / "piper_lessac_pink_noise.wav.json"
    assert audio_path.is_file()
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["background_noise"] == "pink"
    assert metadata["noise_level_db"] == -28.0
    assert metadata["duration_s"] == 1.5


def test_summarize_voice_benchmark_results_reports_quality_metrics() -> None:
    results = [
        {
            "fixture_id": "canonical_open_safari",
            "category": "canonical_command",
            "tags": ["canonical", "command", "app-launch"],
            "lane": "command",
            "source_engine": "piper",
            "prompt_text": "open Safari",
            "reference_transcript": "open Safari",
            "expected_route": "command",
            "expected_action": "open_app",
            "audio_capture": {"status": "success", "duration_s": 1.0, "transcript": "Open Safari"},
            "signal_summary": {
                "transcript": "Open Safari",
                "route": "command",
                "proposal": {"action": "open_app"},
            },
            "evaluation": {
                "passed": True,
                "checks": [
                    {"name": "reference_transcript", "expected": "open Safari", "actual": "Open Safari", "passed": True},
                    {"name": "expected_route", "expected": "command", "actual": "command", "passed": True},
                    {"name": "expected_action", "expected": "open_app", "actual": "open_app", "passed": True},
                    {"name": "expected_arguments", "expected": {"app_name": "Safari"}, "actual": {"app_name": "Safari"}, "passed": True},
                    {"name": "expected_risk", "expected": "L1", "actual": "L1", "passed": True},
                ],
            },
            "duration_ms": 120.0,
        },
        {
            "fixture_id": "paraphrase_launch_safari",
            "category": "paraphrase_command",
            "tags": ["paraphrase", "command", "app-launch"],
            "lane": "command",
            "source_engine": "piper",
            "prompt_text": "launch Safari",
            "reference_transcript": "launch Safari",
            "expected_route": "command",
            "expected_action": "open_app",
            "audio_capture": {"status": "success", "duration_s": 1.2, "transcript": "Launch Safari"},
            "signal_summary": {
                "transcript": "Launch Safari",
                "route": "command",
                "proposal": {"action": "open_app"},
            },
            "evaluation": {
                "passed": True,
                "checks": [
                    {"name": "reference_transcript", "expected": "launch Safari", "actual": "Launch Safari", "passed": True},
                    {"name": "expected_route", "expected": "command", "actual": "command", "passed": True},
                    {"name": "expected_action", "expected": "open_app", "actual": "open_app", "passed": True},
                    {"name": "expected_arguments", "expected": {"app_name": "Safari"}, "actual": {"app_name": "Safari"}, "passed": True},
                    {"name": "expected_risk", "expected": "L1", "actual": "L1", "passed": True},
                ],
            },
            "duration_ms": 180.0,
        },
        {
            "fixture_id": "deictic_click_this_without_gaze",
            "category": "deictic_command",
            "tags": ["deictic", "clarify", "command"],
            "lane": "command",
            "source_engine": "piper",
            "prompt_text": "click this",
            "expected_route": "clarify",
            "audio_capture": {"status": "success", "duration_s": 0.8, "transcript": "click this"},
            "signal_summary": {
                "transcript": "click this",
                "route": "clarify",
                "proposal": None,
            },
            "evaluation": {
                "passed": True,
                "checks": [
                    {"name": "expected_route", "expected": "clarify", "actual": "clarify", "passed": True},
                    {"name": "expected_message_contains", "expected": "gaze grounding", "actual": "Need gaze grounding", "passed": True},
                ],
            },
            "duration_ms": 220.0,
        },
        {
            "fixture_id": "dictation_long_status_update",
            "category": "dictation_long_form",
            "tags": ["dictation", "long-form"],
            "lane": "dictation",
            "source_engine": "piper",
            "prompt_text": "hello team this should stay in dictation mode",
            "expected_route": "dictation",
            "expected_action": "type_text",
            "audio_capture": {
                "status": "success",
                "duration_s": 2.4,
                "transcript": "hello team this should stay in dictation mode",
            },
            "signal_summary": {
                "transcript": "hello team this should stay in dictation mode",
                "route": "dictation",
                "proposal": {"action": "type_text"},
            },
            "evaluation": {
                "passed": True,
                "checks": [
                    {"name": "expected_route", "expected": "dictation", "actual": "dictation", "passed": True},
                    {"name": "expected_action", "expected": "type_text", "actual": "type_text", "passed": True},
                    {"name": "expected_risk", "expected": "L2", "actual": "L2", "passed": True},
                    {"name": "expected_broker_decision", "expected": "allow", "actual": "allow", "passed": True},
                ],
            },
            "duration_ms": 260.0,
        },
        {
            "fixture_id": "negative_story_about_escape_key",
            "category": "negative_long_form",
            "tags": ["negative", "long-form", "command"],
            "lane": "command",
            "source_engine": "piper",
            "prompt_text": "the story mentioned the escape key but did not ask to press it",
            "expected_route": "reject",
            "audio_capture": {
                "status": "success",
                "duration_s": 3.0,
                "transcript": "the story mentioned the escape key but did not ask to press it",
            },
            "signal_summary": {
                "transcript": "the story mentioned the escape key but did not ask to press it",
                "route": "reject",
                "proposal": None,
            },
            "evaluation": {
                "passed": True,
                "checks": [
                    {"name": "expected_route", "expected": "reject", "actual": "reject", "passed": True},
                    {"name": "expected_message_contains", "expected": "could not map", "actual": "I could not map that request", "passed": True},
                ],
            },
            "duration_ms": 300.0,
        },
        {
            "fixture_id": "negative_false_positive",
            "category": "negative_long_form",
            "tags": ["negative", "long-form", "command"],
            "lane": "command",
            "source_engine": "piper",
            "prompt_text": "open Safari now",
            "expected_route": "reject",
            "audio_capture": {"status": "success", "duration_s": 1.8, "transcript": "open Safari now"},
            "signal_summary": {
                "transcript": "open Safari now",
                "route": "command",
                "proposal": {"action": "open_app"},
            },
            "evaluation": {
                "passed": False,
                "checks": [
                    {"name": "expected_route", "expected": "reject", "actual": "command", "passed": False},
                ],
            },
            "duration_ms": 400.0,
        },
    ]

    summary = summarize_voice_benchmark_results(results)

    assert summary["fixtures_total"] == 6
    assert summary["passed"] == 5
    assert summary["failed"] == 1
    assert summary["by_actual_route"] == {
        "clarify": 1,
        "command": 3,
        "dictation": 1,
        "reject": 1,
    }
    assert summary["by_actual_action"] == {
        "open_app": 3,
        "type_text": 1,
    }

    metrics = summary["metrics"]
    assert metrics["command_recognition_rate"] == {
        "eligible": 2,
        "passed": 2,
        "failed": 0,
        "pass_rate": 1.0,
    }
    assert metrics["natural_variation_score"] == {
        "eligible": 1,
        "passed": 1,
        "failed": 0,
        "pass_rate": 1.0,
    }
    assert metrics["deictic_clarification_rate"] == {
        "eligible": 1,
        "passed": 1,
        "failed": 0,
        "pass_rate": 1.0,
    }
    assert metrics["dictation_success_rate"] == {
        "eligible": 1,
        "passed": 1,
        "failed": 0,
        "pass_rate": 1.0,
    }
    assert metrics["false_positive_rate"] == {
        "eligible": 2,
        "false_positives": 1,
        "correct_rejections": 1,
        "rate": 0.5,
    }
    assert metrics["word_error_rate"] == {
        "samples": 6,
        "reference_words": 30,
        "edit_distance": 0,
        "wer": 0.0,
    }
    assert metrics["dictation_text_accuracy"] == {
        "samples": 0,
        "exact_matches": 0,
        "reference_characters": 0,
        "edit_distance": 0,
        "accuracy": 0.0,
        "exact_match_rate": 0.0,
    }
    assert metrics["translation_text_accuracy"] == {
        "samples": 0,
        "exact_matches": 0,
        "reference_characters": 0,
        "edit_distance": 0,
        "accuracy": 0.0,
        "exact_match_rate": 0.0,
    }
    assert metrics["translation_success_rate"] == {
        "eligible": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
    }
    assert metrics["translation_variation_score"] == {
        "eligible": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
    }
    assert metrics["latency_ms"] == {
        "samples": 6,
        "min": 120.0,
        "median": 240.0,
        "mean": 246.67,
        "p95": 400.0,
        "max": 400.0,
    }
    assert metrics["audio_duration_s"] == {
        "samples": 6,
        "min": 0.8,
        "median": 1.5,
        "mean": 1.7,
        "p95": 3.0,
        "max": 3.0,
    }
    assert metrics["check_pass_rates"]["expected_route"] == {
        "eligible": 6,
        "passed": 5,
        "failed": 1,
        "pass_rate": 0.8333,
    }
    diagnostics = summary["diagnostics"]
    assert diagnostics["failed_checks"] == {"expected_route": 1}
    assert diagnostics["failure_examples"][0]["case_id"] == "negative_false_positive"
    assert diagnostics["failure_examples"][0]["failed_checks"][0] == {
        "name": "expected_route",
        "expected": "reject",
        "actual": "command",
    }
    assert diagnostics["transcript_mismatches"] == []
    assert diagnostics["output_text_mismatches"] == []


def test_render_voice_benchmark_summary_formats_human_readable_report() -> None:
    summary = {
        "fixtures_total": 6,
        "passed": 5,
        "failed": 1,
        "pass_rate": 0.8333,
        "by_actual_route": {
            "clarify": 1,
            "command": 3,
            "dictation": 1,
            "reject": 1,
        },
        "by_source_engine": {"piper": 6},
        "metrics": {
            "word_error_rate": {"wer": 0.0125},
            "dictation_text_accuracy": {"samples": 0},
            "translation_text_accuracy": {"samples": 0},
            "translation_success_rate": {"eligible": 0, "pass_rate": 0.0},
            "translation_variation_score": {"eligible": 0, "pass_rate": 0.0},
            "command_recognition_rate": {"pass_rate": 1.0},
            "natural_variation_score": {"pass_rate": 1.0},
            "deictic_clarification_rate": {"pass_rate": 1.0},
            "dictation_success_rate": {"pass_rate": 1.0},
            "false_positive_rate": {"rate": 0.5},
            "latency_ms": {"median": 240.0, "p95": 400.0, "max": 401.0},
            "audio_duration_s": {"median": 1.5, "p95": 3.0, "max": 3.1},
            "check_pass_rates": {
                "reference_transcript": {"eligible": 4, "passed": 3, "pass_rate": 0.75},
            },
        },
        "failing_ids": ["negative_false_positive"],
    }

    rendered = render_voice_benchmark_summary(summary)

    assert rendered == "\n".join(
        [
            "Voice Benchmark Summary",
            "Fixtures   5/6 passed (83.3%) | failed 1",
            "Quality    WER 1.2% | CRR 100.0% | NVS 100.0% | Deictic 100.0% | Dictation 100.0% | FPR 50.0%",
            "Latency    median 240.00 ms | p95 400.00 ms | max 401.00 ms",
            "Audio      median 1.50 s | p95 3.00 s | max 3.10 s",
            "Transcript exact-match 75.0% (3/4)",
            "Routes     clarify=1, command=3, dictation=1, reject=1",
            "Sources    piper=6",
            "Failing    negative_false_positive",
        ]
    )


def test_render_voice_benchmark_summary_includes_diagnostics_when_present() -> None:
    rendered = render_voice_benchmark_summary(
        {
            "fixtures_total": 2,
            "passed": 1,
            "failed": 1,
            "pass_rate": 0.5,
            "by_actual_route": {"command": 1, "reject": 1},
            "by_source_engine": {"piper": 2},
            "metrics": {
                "word_error_rate": {"wer": 0.125},
                "dictation_text_accuracy": {"samples": 0},
                "translation_text_accuracy": {"samples": 0},
                "translation_success_rate": {"eligible": 0, "pass_rate": 0.0},
                "translation_variation_score": {"eligible": 0, "pass_rate": 0.0},
                "command_recognition_rate": {"pass_rate": 1.0},
                "natural_variation_score": {"pass_rate": 1.0},
                "deictic_clarification_rate": {"pass_rate": 1.0},
                "dictation_success_rate": {"pass_rate": 1.0},
                "false_positive_rate": {"rate": 0.0},
                "latency_ms": {"median": 200.0, "p95": 250.0, "max": 260.0},
                "audio_duration_s": {"median": 1.0, "p95": 1.2, "max": 1.3},
                "check_pass_rates": {"reference_transcript": {"eligible": 2, "passed": 1, "pass_rate": 0.5}},
            },
            "diagnostics": {
                "failed_checks": {"expected_route": 1, "reference_output_text": 1},
                "failure_examples": [
                    {
                        "case_id": "negative_false_positive",
                        "failed_checks": [
                            {"name": "expected_route", "expected": "reject", "actual": "command"}
                        ],
                    }
                ],
                "transcript_mismatches": [
                    {
                        "case_id": "canonical_scroll_down_faster:piper_lessac_low",
                        "reference": "scroll down faster",
                        "actual": "crawl down faster",
                        "edit_distance": 1,
                    }
                ],
                "output_text_mismatches": [
                    {
                        "case_id": "translation_hello_french",
                        "reference": "bonjour",
                        "actual": "bonjour!",
                        "edit_distance": 1,
                    }
                ],
            },
            "failing_ids": ["negative_false_positive"],
        }
    )

    assert "Checks     expected_route=1, reference_output_text=1" in rendered
    assert "Example    negative_false_positive expected_route reject -> command" in rendered
    assert (
        "Drift      transcript canonical_scroll_down_faster:piper_lessac_low scroll down faster -> crawl down faster | "
        "output translation_hello_french bonjour -> bonjour!"
    ) in rendered


def test_summarize_voice_benchmark_results_reports_output_text_accuracy_metrics() -> None:
    results = [
        {
            "fixture_id": "dictation_status",
            "category": "dictation_long_form",
            "tags": ["dictation"],
            "lane": "dictation",
            "reference_output_text": "Hello team",
            "expected_route": "dictation",
            "signal_summary": {
                "route": "dictation",
                "proposal": {
                    "action": "type_text",
                    "arguments": {"text": "Hello team", "voice_route": "dictation"},
                },
            },
            "evaluation": {
                "passed": True,
                "checks": [
                    {"name": "expected_route", "expected": "dictation", "actual": "dictation", "passed": True},
                    {"name": "reference_output_text", "expected": "Hello team", "actual": "Hello team", "passed": True},
                ],
            },
        },
        {
            "fixture_id": "translation_hello_french",
            "category": "translation_command",
            "tags": ["translation"],
            "lane": "command",
            "reference_output_text": "bonjour",
            "expected_route": "translation",
            "signal_summary": {
                "route": "translation",
                "proposal": {
                    "action": "type_text",
                    "arguments": {"text": "bonjour!", "voice_route": "translation"},
                },
            },
            "evaluation": {
                "passed": False,
                "checks": [
                    {"name": "expected_route", "expected": "translation", "actual": "translation", "passed": True},
                    {"name": "reference_output_text", "expected": "bonjour", "actual": "bonjour!", "passed": False},
                ],
            },
        },
    ]

    summary = summarize_voice_benchmark_results(results)

    assert summary["metrics"]["dictation_text_accuracy"] == {
        "samples": 1,
        "exact_matches": 1,
        "reference_characters": 10,
        "edit_distance": 0,
        "accuracy": 1.0,
        "exact_match_rate": 1.0,
    }
    assert summary["metrics"]["translation_text_accuracy"] == {
        "samples": 1,
        "exact_matches": 0,
        "reference_characters": 7,
        "edit_distance": 1,
        "accuracy": 0.8571,
        "exact_match_rate": 0.0,
    }
    assert summary["metrics"]["translation_success_rate"] == {
        "eligible": 1,
        "passed": 1,
        "failed": 0,
        "pass_rate": 1.0,
    }
    assert summary["metrics"]["translation_variation_score"] == {
        "eligible": 0,
        "passed": 0,
        "failed": 0,
        "pass_rate": 0.0,
    }
    diagnostics = summary["diagnostics"]
    assert diagnostics["failed_checks"] == {"reference_output_text": 1}
    assert diagnostics["output_text_mismatches"][0]["case_id"] == "translation_hello_french"
    assert diagnostics["output_text_mismatches"][0]["reference"] == "bonjour"
    assert diagnostics["output_text_mismatches"][0]["actual"] == "bonjour!"


def test_summarize_voice_benchmark_results_reports_transcript_drift_even_when_signal_passes() -> None:
    results = [
        {
            "fixture_id": "canonical_scroll_down_faster",
            "variant_id": "piper_lessac_low",
            "category": "canonical_command",
            "tags": ["canonical", "command", "scroll"],
            "lane": "command",
            "source_engine": "piper",
            "prompt_text": "scroll down faster",
            "reference_transcript": "scroll down faster",
            "expected_route": "command",
            "expected_action": "scroll",
            "audio_capture": {
                "status": "success",
                "duration_s": 1.1,
                "transcript": "crawl down faster",
            },
            "signal_summary": {
                "transcript": "crawl down faster",
                "route": "command",
                "proposal": {"action": "scroll", "arguments": {"direction": "down"}},
            },
            "evaluation": {
                "passed": True,
                "checks": [
                    {
                        "name": "reference_transcript",
                        "expected": "scroll down faster",
                        "actual": "crawl down faster",
                        "passed": False,
                    },
                    {"name": "expected_route", "expected": "command", "actual": "command", "passed": True},
                    {"name": "expected_action", "expected": "scroll", "actual": "scroll", "passed": True},
                ],
            },
            "duration_ms": 210.0,
        }
    ]

    summary = summarize_voice_benchmark_results(results)

    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["metrics"]["word_error_rate"] == {
        "samples": 1,
        "reference_words": 3,
        "edit_distance": 1,
        "wer": 0.3333,
    }
    assert summary["diagnostics"]["failed_checks"] == {"reference_transcript": 1}
    assert summary["diagnostics"]["transcript_mismatches"] == [
        {
            "case_id": "canonical_scroll_down_faster:piper_lessac_low",
            "fixture_id": "canonical_scroll_down_faster",
            "variant_id": "piper_lessac_low",
            "reference": "scroll down faster",
            "actual": "crawl down faster",
            "edit_distance": 1,
            "passed": True,
        }
    ]



def test_evaluate_voice_benchmark_budget_enforces_thresholds_and_skips_low_sample_metrics() -> None:
    summary = {
        "pass_rate": 1.0,
        "metrics": {
            "word_error_rate": {"samples": 36, "wer": 0.0367},
            "false_positive_rate": {"eligible": 10, "rate": 0.0},
            "latency_ms": {"samples": 36, "median": 297.43, "p95": 365.11, "max": 385.82},
            "dictation_text_accuracy": {"samples": 4, "accuracy": 0.9519},
            "translation_text_accuracy": {"samples": 4, "accuracy": 1.0},
            "translation_success_rate": {"eligible": 4, "pass_rate": 1.0},
            "translation_variation_score": {"eligible": 2, "pass_rate": 1.0},
        },
    }
    budget = VoiceBenchmarkBudget.model_validate(
        {
            "budget_id": "phase9-signal-only",
            "checks": [
                {"name": "pass_rate", "path": "pass_rate", "min": 1.0},
                {"name": "wer", "path": "metrics.word_error_rate.wer", "max": 0.05},
                {"name": "latency_p95", "path": "metrics.latency_ms.p95", "max": 400.0},
                {"name": "dictation_text", "path": "metrics.dictation_text_accuracy.accuracy", "min": 0.9, "min_samples": 1},
                {"name": "translation_text", "path": "metrics.translation_text_accuracy.accuracy", "min": 0.85, "min_samples": 1},
                {
                    "name": "translation_route",
                    "path": "metrics.translation_success_rate.pass_rate",
                    "min": 1.0,
                    "min_samples": 1,
                    "samples_path": "metrics.translation_success_rate.eligible",
                },
                {
                    "name": "translation_variation",
                    "path": "metrics.translation_variation_score.pass_rate",
                    "min": 1.0,
                    "min_samples": 1,
                    "samples_path": "metrics.translation_variation_score.eligible",
                },
            ],
        }
    )

    result = evaluate_voice_benchmark_budget(summary, budget)

    assert result["passed"] is True
    assert result["passed_checks"] == 7
    assert result["failed_checks"] == 0
    assert result["skipped_checks"] == 0
    by_name = {check["name"]: check for check in result["checks"]}
    assert by_name["pass_rate"]["status"] == "passed"
    assert by_name["wer"]["status"] == "passed"
    assert by_name["latency_p95"]["status"] == "passed"
    assert by_name["dictation_text"]["status"] == "passed"
    assert by_name["translation_text"]["status"] == "passed"
    assert by_name["translation_route"]["status"] == "passed"
    assert by_name["translation_variation"]["status"] == "passed"


def test_evaluate_voice_benchmark_budget_skips_irrelevant_subset_checks() -> None:
    summary = {
        "pass_rate": 1.0,
        "metrics": {
            "word_error_rate": {"samples": 2, "wer": 0.1667},
            "command_recognition_rate": {"eligible": 2, "pass_rate": 1.0},
            "natural_variation_score": {"eligible": 0, "pass_rate": 0.0},
            "deictic_clarification_rate": {"eligible": 0, "pass_rate": 0.0},
            "dictation_success_rate": {"eligible": 0, "pass_rate": 0.0},
            "false_positive_rate": {"eligible": 0, "rate": 0.0},
            "latency_ms": {"samples": 2, "median": 313.79, "p95": 343.91, "max": 343.91},
        },
    }
    budget = VoiceBenchmarkBudget.model_validate(
        {
            "budget_id": "phase9-signal-only",
            "checks": [
                {"name": "pass_rate", "path": "pass_rate", "min": 1.0},
                {
                    "name": "wer",
                    "path": "metrics.word_error_rate.wer",
                    "max": 0.05,
                    "min_samples": 20,
                    "samples_path": "metrics.word_error_rate.samples",
                },
                {
                    "name": "command_recognition_rate",
                    "path": "metrics.command_recognition_rate.pass_rate",
                    "min": 1.0,
                    "min_samples": 1,
                    "samples_path": "metrics.command_recognition_rate.eligible",
                },
                {
                    "name": "natural_variation_score",
                    "path": "metrics.natural_variation_score.pass_rate",
                    "min": 1.0,
                    "min_samples": 1,
                    "samples_path": "metrics.natural_variation_score.eligible",
                },
                {
                    "name": "deictic_clarification_rate",
                    "path": "metrics.deictic_clarification_rate.pass_rate",
                    "min": 1.0,
                    "min_samples": 1,
                    "samples_path": "metrics.deictic_clarification_rate.eligible",
                },
                {
                    "name": "dictation_success_rate",
                    "path": "metrics.dictation_success_rate.pass_rate",
                    "min": 1.0,
                    "min_samples": 1,
                    "samples_path": "metrics.dictation_success_rate.eligible",
                },
                {
                    "name": "false_positive_rate",
                    "path": "metrics.false_positive_rate.rate",
                    "max": 0.0,
                    "min_samples": 1,
                    "samples_path": "metrics.false_positive_rate.eligible",
                },
                {
                    "name": "latency_median_ms",
                    "path": "metrics.latency_ms.median",
                    "max": 400.0,
                    "min_samples": 10,
                    "samples_path": "metrics.latency_ms.samples",
                },
            ],
        }
    )

    result = evaluate_voice_benchmark_budget(summary, budget)

    assert result["passed"] is True
    assert result["passed_checks"] == 2
    assert result["failed_checks"] == 0
    assert result["skipped_checks"] == 6
    by_name = {check["name"]: check for check in result["checks"]}
    assert by_name["pass_rate"]["status"] == "passed"
    assert by_name["command_recognition_rate"]["status"] == "passed"
    assert by_name["wer"]["status"] == "skipped"
    assert by_name["natural_variation_score"]["status"] == "skipped"
    assert by_name["deictic_clarification_rate"]["status"] == "skipped"
    assert by_name["dictation_success_rate"]["status"] == "skipped"
    assert by_name["false_positive_rate"]["status"] == "skipped"
    assert by_name["latency_median_ms"]["status"] == "skipped"


def test_render_voice_benchmark_budget_summary_formats_human_readable_report() -> None:
    rendered = render_voice_benchmark_budget_summary(
        {
            "budget_id": "phase9-signal-only",
            "passed": False,
            "passed_checks": 3,
            "failed_checks": 1,
            "skipped_checks": 1,
            "checks": [
                {"name": "pass_rate", "status": "passed", "reason": None},
                {"name": "latency_p95", "status": "failed", "reason": "420.0000 > max 400.0000"},
                {"name": "translation_text", "status": "skipped", "reason": "samples 0 below minimum 1"},
            ],
        }
    )

    assert rendered == "\n".join(
        [
            "Voice Benchmark Budget",
            "Budget     phase9-signal-only | failed",
            "Checks     3 passed | 1 failed | 1 skipped",
            "Failed     latency_p95 (420.0000 > max 400.0000)",
            "Skipped    translation_text (samples 0 below minimum 1)",
        ]
    )


def test_render_voice_benchmark_summary_includes_translation_line_when_present() -> None:
    rendered = render_voice_benchmark_summary(
        {
            "fixtures_total": 4,
            "passed": 4,
            "failed": 0,
            "pass_rate": 1.0,
            "by_actual_route": {"translation": 4},
            "by_source_engine": {"piper": 4},
            "metrics": {
                "word_error_rate": {"wer": 0.02},
                "dictation_text_accuracy": {"samples": 0},
                "translation_text_accuracy": {"samples": 4, "accuracy": 1.0},
                "translation_success_rate": {"eligible": 4, "pass_rate": 1.0},
                "translation_variation_score": {"eligible": 2, "pass_rate": 1.0},
                "command_recognition_rate": {"pass_rate": 1.0},
                "natural_variation_score": {"pass_rate": 1.0},
                "deictic_clarification_rate": {"pass_rate": 1.0},
                "dictation_success_rate": {"pass_rate": 1.0},
                "false_positive_rate": {"rate": 0.0},
                "latency_ms": {"median": 200.0, "p95": 250.0, "max": 260.0},
                "audio_duration_s": {"median": 1.5, "p95": 1.9, "max": 2.0},
                "check_pass_rates": {"reference_transcript": {"eligible": 4, "passed": 4, "pass_rate": 1.0}},
            },
            "failing_ids": [],
        }
    )

    assert "Text Out   Trans 100.0%" in rendered
    assert "Translate route 100.0% | variation 100.0%" in rendered


def test_render_voice_benchmark_summary_includes_phase_breakdown_when_present() -> None:
    rendered = render_voice_benchmark_summary(
        {
            "fixtures_total": 2,
            "passed": 2,
            "failed": 0,
            "pass_rate": 1.0,
            "by_actual_route": {"command": 2},
            "by_source_engine": {"piper": 2},
            "metrics": {
                "word_error_rate": {"wer": 0.0},
                "dictation_text_accuracy": {"samples": 0},
                "translation_text_accuracy": {"samples": 0},
                "translation_success_rate": {"eligible": 0, "pass_rate": 0.0},
                "translation_variation_score": {"eligible": 0, "pass_rate": 0.0},
                "command_recognition_rate": {"pass_rate": 1.0},
                "natural_variation_score": {"pass_rate": 1.0},
                "deictic_clarification_rate": {"pass_rate": 1.0},
                "dictation_success_rate": {"pass_rate": 1.0},
                "false_positive_rate": {"rate": 0.0},
                "latency_ms": {"median": 240.0, "p95": 260.0, "max": 270.0},
                "latency_breakdown_ms": {
                    "transcription_ms": {"samples": 2, "median": 170.0},
                    "signal_ms": {"samples": 2, "median": 45.0},
                    "postprocess_ms": {"samples": 2, "median": 12.0},
                },
                "audio_duration_s": {"median": 1.5, "p95": 1.7, "max": 1.8},
                "check_pass_rates": {"reference_transcript": {"eligible": 2, "passed": 2, "pass_rate": 1.0}},
            },
            "failing_ids": [],
        }
    )

    assert "Phases     speech 170.00 ms | signal 45.00 ms | post 12.00 ms" in rendered


def test_compare_voice_benchmark_summaries_reports_metric_deltas() -> None:
    previous = {
        "pass_rate": 1.0,
        "metrics": {
            "word_error_rate": {"wer": 0.05},
            "command_recognition_rate": {"pass_rate": 0.9},
            "false_positive_rate": {"rate": 0.02},
            "dictation_text_accuracy": {"accuracy": 0.9},
            "translation_text_accuracy": {"accuracy": 1.0},
            "latency_ms": {"median": 320.0, "p95": 410.0, "max": 600.0},
        },
    }
    current = {
        "pass_rate": 1.0,
        "metrics": {
            "word_error_rate": {"wer": 0.0382},
            "command_recognition_rate": {"pass_rate": 1.0},
            "false_positive_rate": {"rate": 0.0},
            "dictation_text_accuracy": {"accuracy": 0.9519},
            "translation_text_accuracy": {"accuracy": 1.0},
            "latency_ms": {"median": 289.18, "p95": 388.49, "max": 416.56},
        },
    }

    delta = compare_voice_benchmark_summaries(previous, current)

    assert delta["has_baseline"] is True
    assert delta["status"] == "improved"
    assert delta["improved_metrics"] == 7
    assert delta["regressed_metrics"] == 0
    assert delta["unchanged_metrics"] == 2
    assert delta["metrics"]["word_error_rate"] == {
        "previous": 0.05,
        "current": 0.0382,
        "delta": -0.0118,
        "direction": "improved",
        "better": "lower",
        "unit": "ratio",
        "tolerance": 0.001,
    }
    assert delta["metrics"]["translation_text_accuracy"]["direction"] == "unchanged"
    assert delta["metrics"]["latency_max_ms"]["delta"] == -183.44
    assert delta["metrics"]["latency_max_ms"]["tolerance"] == 5.0


def test_compare_voice_benchmark_summaries_ignores_small_latency_jitter() -> None:
    previous = {
        "pass_rate": 1.0,
        "metrics": {
            "word_error_rate": {"wer": 0.0382},
            "command_recognition_rate": {"pass_rate": 1.0},
            "false_positive_rate": {"rate": 0.0},
            "dictation_text_accuracy": {"accuracy": 0.9519},
            "translation_text_accuracy": {"accuracy": 1.0},
            "latency_ms": {"median": 287.18, "p95": 360.79, "max": 398.88},
        },
    }
    current = {
        "pass_rate": 1.0,
        "metrics": {
            "word_error_rate": {"wer": 0.0382},
            "command_recognition_rate": {"pass_rate": 1.0},
            "false_positive_rate": {"rate": 0.0},
            "dictation_text_accuracy": {"accuracy": 0.9519},
            "translation_text_accuracy": {"accuracy": 1.0},
            "latency_ms": {"median": 282.4, "p95": 364.86, "max": 403.39},
        },
    }

    delta = compare_voice_benchmark_summaries(previous, current)

    assert delta["has_baseline"] is True
    assert delta["status"] == "unchanged"
    assert delta["improved_metrics"] == 0
    assert delta["regressed_metrics"] == 0
    assert delta["unchanged_metrics"] == 9
    assert delta["metrics"]["latency_median_ms"]["direction"] == "unchanged"
    assert delta["metrics"]["latency_p95_ms"]["direction"] == "unchanged"
    assert delta["metrics"]["latency_max_ms"]["direction"] == "unchanged"


def test_render_voice_benchmark_delta_summary_formats_human_readable_report() -> None:
    rendered = render_voice_benchmark_delta_summary(
        {
            "has_baseline": True,
            "status": "mixed",
            "improved_metrics": 4,
            "regressed_metrics": 2,
            "unchanged_metrics": 1,
            "metrics": {
                "word_error_rate": {"delta": -0.0118, "unit": "ratio", "direction": "improved"},
                "command_recognition_rate": {"delta": 0.1, "unit": "ratio", "direction": "improved"},
                "false_positive_rate": {"delta": -0.02, "unit": "ratio", "direction": "improved"},
                "dictation_text_accuracy": {"delta": 0.0519, "unit": "ratio", "direction": "improved"},
                "translation_text_accuracy": {"delta": 0.0, "unit": "ratio", "direction": "unchanged"},
                "latency_median_ms": {"delta": -30.82, "unit": "ms", "direction": "improved"},
                "latency_p95_ms": {"delta": 18.49, "unit": "ms", "direction": "regressed"},
                "latency_max_ms": {"delta": -183.44, "unit": "ms", "direction": "improved"},
            },
        }
    )

    assert rendered == "\n".join(
        [
            "Voice Benchmark Delta",
            "Delta      mixed | 4 improved | 2 regressed | 1 unchanged",
            "Quality    WER -1.2 pts | CRR +10.0 pts | FPR -2.0 pts",
            "Text Out   Dict +5.2 pts | Trans stable",
            "Latency    median -30.82 ms | p95 +18.49 ms | max -183.44 ms",
        ]
    )


def test_render_voice_benchmark_delta_summary_handles_missing_baseline() -> None:
    assert render_voice_benchmark_delta_summary({"has_baseline": False}) == "\n".join(
        [
            "Voice Benchmark Delta",
            "Delta      no previous baseline",
        ]
    )


def test_summarize_voice_benchmark_comparison_reports_side_by_side_metrics() -> None:
    comparison = summarize_voice_benchmark_comparison(
        {
            "regex": {
                "fixtures_total": 44,
                "passed": 40,
                "failed": 4,
                "pass_rate": 0.9091,
                "failing_ids": [
                    "translation_hello_french:piper_lessac_low",
                    "translation_hello_french:piper_amy_low",
                ],
                "metrics": {
                    "word_error_rate": {"samples": 44, "wer": 0.045},
                    "command_recognition_rate": {"eligible": 20, "pass_rate": 0.92},
                    "natural_variation_score": {"eligible": 8, "pass_rate": 0.75},
                    "false_positive_rate": {"eligible": 4, "rate": 0.04},
                    "latency_ms": {"samples": 44, "median": 320.0, "p95": 410.0},
                    "translation_success_rate": {"eligible": 0, "pass_rate": 0.0},
                },
            },
            "hybrid": {
                "fixtures_total": 44,
                "passed": 44,
                "failed": 0,
                "pass_rate": 1.0,
                "failing_ids": [],
                "metrics": {
                    "word_error_rate": {"samples": 44, "wer": 0.038},
                    "command_recognition_rate": {"eligible": 20, "pass_rate": 1.0},
                    "natural_variation_score": {"eligible": 8, "pass_rate": 1.0},
                    "false_positive_rate": {"eligible": 4, "rate": 0.0},
                    "latency_ms": {"samples": 44, "median": 285.0, "p95": 360.0},
                    "translation_success_rate": {"eligible": 0, "pass_rate": 0.0},
                },
            },
        },
        baseline_label="regex",
    )

    assert comparison["labels"] == ["regex", "hybrid"]
    assert comparison["baseline_label"] == "regex"
    assert comparison["runs"]["regex"] == {
        "fixtures_total": 44,
        "passed": 40,
        "failed": 4,
        "pass_rate": 0.9091,
        "failing_ids": [
            "translation_hello_french:piper_amy_low",
            "translation_hello_french:piper_lessac_low",
        ],
    }
    assert comparison["metrics"]["word_error_rate"]["best_labels"] == ["hybrid"]
    assert comparison["metrics"]["latency_median_ms"]["best_value"] == 285.0
    assert comparison["metrics"]["natural_variation_score"]["values"] == {
        "regex": 0.75,
        "hybrid": 1.0,
    }
    assert "translation_success_rate" not in comparison["metrics"]
    assert comparison["deltas"]["hybrid"]["status"] == "improved"
    assert comparison["failure_deltas"]["hybrid"] == {
        "fixed": [
            "translation_hello_french:piper_amy_low",
            "translation_hello_french:piper_lessac_low",
        ],
        "regressed": [],
    }


def test_render_voice_benchmark_comparison_formats_human_readable_report() -> None:
    rendered = render_voice_benchmark_comparison(
        {
            "labels": ["regex", "hybrid"],
            "baseline_label": "regex",
            "runs": {
                "regex": {"fixtures_total": 44, "passed": 40},
                "hybrid": {"fixtures_total": 44, "passed": 44},
            },
            "metrics": {
                "pass_rate": {
                    "label": "Pass",
                    "unit": "ratio",
                    "values": {"regex": 0.9091, "hybrid": 1.0},
                    "best_labels": ["hybrid"],
                },
                "word_error_rate": {
                    "label": "WER",
                    "unit": "ratio",
                    "values": {"regex": 0.045, "hybrid": 0.038},
                    "best_labels": ["hybrid"],
                },
                "command_recognition_rate": {
                    "label": "CRR",
                    "unit": "ratio",
                    "values": {"regex": 0.92, "hybrid": 1.0},
                    "best_labels": ["hybrid"],
                },
                "natural_variation_score": {
                    "label": "NVS",
                    "unit": "ratio",
                    "values": {"regex": 0.75, "hybrid": 1.0},
                    "best_labels": ["hybrid"],
                },
                "false_positive_rate": {
                    "label": "FPR",
                    "unit": "ratio",
                    "values": {"regex": 0.04, "hybrid": 0.0},
                    "best_labels": ["hybrid"],
                },
                "latency_median_ms": {
                    "label": "Median",
                    "unit": "ms",
                    "values": {"regex": 320.0, "hybrid": 285.0},
                    "best_labels": ["hybrid"],
                },
                "latency_p95_ms": {
                    "label": "p95",
                    "unit": "ms",
                    "values": {"regex": 410.0, "hybrid": 360.0},
                    "best_labels": ["hybrid"],
                },
            },
            "deltas": {
                "hybrid": {"status": "improved"},
            },
            "failure_deltas": {
                "hybrid": {
                    "fixed": [
                        "translation_hello_french:piper_amy_low",
                        "translation_hello_french:piper_lessac_low",
                        "translation_thank_you_german:piper_lessac_low",
                    ],
                    "regressed": [],
                }
            },
        }
    )

    assert rendered == "\n".join(
        [
            "Voice Benchmark Comparison",
            "Runs       regex, hybrid",
            "Fixtures   regex 40/44 | hybrid 44/44",
            "Pass       regex 90.9% | hybrid 100.0% | best hybrid",
            "WER        regex 4.5% | hybrid 3.8% | best hybrid",
            "CRR        regex 92.0% | hybrid 100.0% | best hybrid",
            "NVS        regex 75.0% | hybrid 100.0% | best hybrid",
            "FPR        regex 4.0% | hybrid 0.0% | best hybrid",
            "Median     regex 320.00 ms | hybrid 285.00 ms | best hybrid",
            "p95        regex 410.00 ms | hybrid 360.00 ms | best hybrid",
            "Delta      vs regex | hybrid improved",
            "Cases      hybrid fixed translation_hello_french x2, translation_thank_you_german",
        ]
    )


def test_phase9_voice_benchmark_supports_interpreter_comparison(monkeypatch, tmp_path: Path) -> None:
    from scripts import phase9_voice_benchmark as benchmark_script

    def fake_invoke(argv: list[str]) -> dict[str, object]:
        interpreter = argv[argv.index("--interpreter") + 1]
        pass_rate = 0.9091 if interpreter == "regex" else 1.0
        median_ms = 320.0 if interpreter == "regex" else 285.0
        return {
            "source": "voice-benchmark",
            "summary": {
                "fixtures_total": 44,
                "passed": 40 if interpreter == "regex" else 44,
                "failed": 4 if interpreter == "regex" else 0,
                "pass_rate": pass_rate,
                "metrics": {
                    "word_error_rate": {"samples": 44, "wer": 0.045 if interpreter == "regex" else 0.038},
                    "command_recognition_rate": {"eligible": 20, "pass_rate": 0.92 if interpreter == "regex" else 1.0},
                    "natural_variation_score": {"eligible": 8, "pass_rate": 0.75 if interpreter == "regex" else 1.0},
                    "false_positive_rate": {"eligible": 4, "rate": 0.04 if interpreter == "regex" else 0.0},
                    "latency_ms": {"samples": 44, "median": median_ms, "p95": median_ms + 90.0, "max": median_ms + 120.0},
                },
            },
            "summary_text": f"Summary {interpreter}",
        }

    written: dict[str, object] = {}
    monkeypatch.setattr(benchmark_script, "_invoke", fake_invoke)
    monkeypatch.setattr(benchmark_script, "_load_previous_summary", lambda _path: None)
    monkeypatch.setattr(
        benchmark_script,
        "_write_current_summary",
        lambda path, summary: written.update({"path": path, "summary": summary}),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "phase9_voice_benchmark.py",
            "--skip-generate",
            "--skip-budget",
            "--runtime-dir",
            str(tmp_path / ".interaction"),
            "--compare-interpreters",
            "regex",
            "hybrid",
        ],
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = benchmark_script.main()
    payload = json.loads(stdout.getvalue())

    assert exit_code == 0
    assert payload["benchmark_interpreter"] == "hybrid"
    assert payload["benchmark_run_order"] == ["regex", "hybrid"]
    assert set(payload["benchmark_runs"]) == {"regex", "hybrid"}
    assert payload["benchmark"]["summary"]["pass_rate"] == 1.0
    assert payload["benchmark_comparison"]["baseline_label"] == "hybrid"
    assert "Voice Benchmark Comparison" in payload["benchmark_comparison_text"]
    assert "Summary hybrid" in payload["benchmark_summary_text"]
    assert written["summary"] == payload["benchmark"]["summary"]
    assert "Voice Benchmark Comparison" in stderr.getvalue()
