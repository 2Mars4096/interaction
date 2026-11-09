"""Helpers for offline voice fixture evaluation."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
import json
import math
from pathlib import Path
import statistics
from typing import Any

from pydantic import Field

from interaction.contracts import ActionName, BrokerDecisionType, RiskLevel
from interaction.contracts.models import ContractModel


class VoiceEvalLane(StrEnum):
    COMMAND = "command"
    DICTATION = "dictation"
    HYBRID = "hybrid"


class VoiceFixtureVariant(ContractModel):
    variant_id: str
    audio_path: str
    source_engine: str | None = None
    voice_id: str | None = None
    speaking_rate: int | None = None
    background_noise: str | None = None
    noise_level_db: float | None = None
    generation_timestamp: str | None = None
    notes: str | None = None

    def resolved_audio_path(self, manifest_path: str | Path) -> Path:
        audio_path = Path(self.audio_path).expanduser()
        if audio_path.is_absolute():
            return audio_path.resolve()
        return (Path(manifest_path).expanduser().resolve().parent / audio_path).resolve()


class VoiceEvalManifest(ContractModel):
    fixture_id: str | None = None
    category: str | None = None
    prompt_text: str | None = None
    lane: VoiceEvalLane | None = None
    audio_path: str | None = None
    variants: list[VoiceFixtureVariant] = Field(default_factory=list)
    reference_transcript: str | None = None
    reference_output_text: str | None = None
    gemma_response: dict[str, Any] | None = None
    gemma_transcript_aliases: list[str] = Field(default_factory=list)
    expected_route: str | None = None
    expected_action: ActionName | None = None
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_risk: RiskLevel | None = None
    expected_broker_decision: BrokerDecisionType | None = None
    expected_message_contains: str | None = None
    locale: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_engine: str | None = None
    voice_id: str | None = None
    speaking_rate: int | None = None
    background_noise: str | None = None
    noise_level_db: float | None = None
    generation_timestamp: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "VoiceEvalManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def resolved_fixture_id(self, manifest_path: str | Path | None = None) -> str:
        if self.fixture_id:
            return self.fixture_id
        if manifest_path is not None:
            return Path(manifest_path).stem
        return "voice-fixture"

    def resolved_variants(self, manifest_path: str | Path) -> list[VoiceFixtureVariant]:
        if self.variants:
            return self.variants
        if self.audio_path:
            return [
                VoiceFixtureVariant(
                    variant_id="default",
                    audio_path=self.audio_path,
                    source_engine=self.source_engine,
                    voice_id=self.voice_id,
                    speaking_rate=self.speaking_rate,
                    background_noise=self.background_noise,
                    noise_level_db=self.noise_level_db,
                    generation_timestamp=self.generation_timestamp,
                )
            ]
        return []


class VoiceBenchmarkBudgetCheck(ContractModel):
    name: str
    path: str
    min: float | None = None
    max: float | None = None
    min_samples: int | None = None
    samples_path: str | None = None
    notes: str | None = None


class VoiceBenchmarkBudget(ContractModel):
    budget_id: str | None = None
    description: str | None = None
    checks: list[VoiceBenchmarkBudgetCheck] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "VoiceBenchmarkBudget":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def discover_voice_eval_manifests(
    path: str | Path,
    *,
    manifest_glob: str = "*.json",
) -> list[Path]:
    root = Path(path).expanduser()
    if root.is_file():
        return [root.resolve()]
    if not root.exists():
        raise FileNotFoundError(f'Voice fixture path not found: "{root}"')
    if not root.is_dir():
        raise ValueError(f'Voice fixture path must be a file or directory: "{root}"')
    return sorted(candidate.resolve() for candidate in root.rglob(manifest_glob) if candidate.is_file())


def summarize_voice_events(events: list[object]) -> dict[str, Any]:
    latest_transcript: str | None = None
    latest_message: str | None = None
    messages: list[str] = []
    decision_payload: dict[str, Any] | None = None
    proposal_payload: dict[str, Any] | None = None
    result_payload: dict[str, Any] | None = None
    signal_only = False
    for event in events:
        transcript = getattr(event, "transcript", None)
        if isinstance(transcript, str) and transcript.strip():
            latest_transcript = transcript
        message = getattr(event, "message", None)
        if isinstance(message, str) and message.strip() and message.strip() != "Voice loop is idle.":
            latest_message = message
            messages.append(message)
        decision = getattr(event, "decision", None)
        if decision is not None and hasattr(decision, "model_dump"):
            decision_payload = decision.model_dump(mode="json")
            proposal = getattr(decision, "proposal", None)
            if proposal is not None and hasattr(proposal, "model_dump"):
                proposal_payload = proposal.model_dump(mode="json")
        result = getattr(event, "result", None)
        if result is not None and hasattr(result, "model_dump"):
            result_payload = result.model_dump(mode="json")
            if proposal_payload is None:
                proposal = getattr(result, "proposal", None)
                if proposal is not None and hasattr(proposal, "model_dump"):
                    proposal_payload = proposal.model_dump(mode="json")
            details = getattr(result, "details", None)
            if isinstance(details, dict) and details.get("signal_only") is True:
                signal_only = True
    route = _infer_route(proposal_payload, latest_message, messages)
    return {
        "transcript": latest_transcript,
        "message": latest_message,
        "messages": messages,
        "route": route,
        "proposal": proposal_payload,
        "decision": decision_payload,
        "result": result_payload,
        "signal_only": signal_only,
    }


def evaluate_voice_manifest(
    summary: dict[str, Any],
    manifest: VoiceEvalManifest | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None

    checks: list[dict[str, Any]] = []
    proposal = summary.get("proposal") if isinstance(summary.get("proposal"), dict) else None
    decision = summary.get("decision") if isinstance(summary.get("decision"), dict) else None
    transcript = summary.get("transcript")
    message = summary.get("message")
    messages = [value for value in summary.get("messages", []) if isinstance(value, str)]

    if manifest.reference_transcript is not None:
        actual_transcript = str(transcript or "")
        checks.append(
            _check(
                "reference_transcript",
                manifest.reference_transcript,
                actual_transcript,
                _normalize_text(manifest.reference_transcript) == _normalize_text(actual_transcript),
            )
        )

    if manifest.reference_output_text is not None:
        actual_output_text = _summary_output_text(summary)
        checks.append(
            _check(
                "reference_output_text",
                manifest.reference_output_text,
                actual_output_text,
                manifest.reference_output_text == actual_output_text,
            )
        )

    if manifest.expected_route is not None:
        actual_route = summary.get("route")
        checks.append(
            _check(
                "expected_route",
                manifest.expected_route,
                actual_route,
                actual_route == manifest.expected_route,
            )
        )

    if manifest.expected_action is not None:
        actual_action = proposal.get("action") if proposal is not None else None
        checks.append(
            _check(
                "expected_action",
                manifest.expected_action.value,
                actual_action,
                actual_action == manifest.expected_action.value,
            )
        )

    if manifest.expected_arguments:
        actual_arguments = proposal.get("arguments") if proposal is not None and isinstance(proposal.get("arguments"), dict) else {}
        checks.append(
            _check(
                "expected_arguments",
                manifest.expected_arguments,
                actual_arguments,
                all(actual_arguments.get(key) == value for key, value in manifest.expected_arguments.items()),
            )
        )

    if manifest.expected_risk is not None:
        actual_risk = proposal.get("risk") if proposal is not None else None
        checks.append(
            _check(
                "expected_risk",
                manifest.expected_risk.value,
                actual_risk,
                actual_risk == manifest.expected_risk.value,
            )
        )

    if manifest.expected_broker_decision is not None:
        actual_decision = decision.get("decision") if decision is not None else None
        checks.append(
            _check(
                "expected_broker_decision",
                manifest.expected_broker_decision.value,
                actual_decision,
                actual_decision == manifest.expected_broker_decision.value,
            )
        )

    if manifest.expected_message_contains is not None:
        actual_message = _matching_message(messages, manifest.expected_message_contains) or str(message or "")
        checks.append(
            _check(
                "expected_message_contains",
                manifest.expected_message_contains,
                actual_message,
                actual_message != "" and manifest.expected_message_contains.lower() in actual_message.lower(),
            )
        )

    non_gating_checks = {"reference_transcript", "reference_output_text"}
    passed = all(check["passed"] for check in checks if check["name"] not in non_gating_checks) if checks else True
    return {
        "passed": passed,
        "checks": checks,
        "tags": list(manifest.tags),
    }


def summarize_voice_benchmark_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = 0
    categories: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    lanes: Counter[str] = Counter()
    source_engines: Counter[str] = Counter()
    actual_routes: Counter[str] = Counter()
    actual_actions: Counter[str] = Counter()
    failing_ids: list[str] = []
    check_stats: dict[str, dict[str, int]] = {}
    latency_ms: list[float] = []
    transcription_ms: list[float] = []
    signal_ms: list[float] = []
    postprocess_ms: list[float] = []
    audio_duration_s: list[float] = []
    transcript_samples = 0
    transcript_reference_words = 0
    transcript_edit_distance = 0
    dictation_output_samples = 0
    dictation_output_exact_matches = 0
    dictation_output_reference_characters = 0
    dictation_output_edit_distance = 0
    translation_output_samples = 0
    translation_output_exact_matches = 0
    translation_output_reference_characters = 0
    translation_output_edit_distance = 0
    command_cases = 0
    command_passed = 0
    paraphrase_cases = 0
    paraphrase_passed = 0
    deictic_cases = 0
    deictic_clarified = 0
    dictation_cases = 0
    dictation_passed = 0
    translation_cases = 0
    translation_passed = 0
    translation_paraphrase_cases = 0
    translation_paraphrase_passed = 0
    negative_cases = 0
    negative_false_positives = 0
    failed_checks: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    transcript_mismatches: list[dict[str, Any]] = []
    output_text_mismatches: list[dict[str, Any]] = []

    for result in results:
        evaluation = result.get("evaluation")
        fixture_id = str(result.get("fixture_id") or "voice-fixture")
        passed_now = bool(isinstance(evaluation, dict) and evaluation.get("passed") is True)
        if passed_now:
            passed += 1
        else:
            failing_ids.append(fixture_id if result.get("variant_id") in {None, "", "default"} else f"{fixture_id}:{result.get('variant_id')}")

        category = result.get("category")
        if isinstance(category, str) and category:
            categories[category] += 1

        lane = result.get("lane")
        if isinstance(lane, str) and lane:
            lanes[lane] += 1

        source_engine = result.get("source_engine")
        if isinstance(source_engine, str) and source_engine:
            source_engines[source_engine] += 1

        for tag in result.get("tags", []):
            if isinstance(tag, str) and tag:
                tags[tag] += 1

        signal_summary = result.get("signal_summary") if isinstance(result.get("signal_summary"), dict) else {}
        actual_route = signal_summary.get("route")
        if isinstance(actual_route, str) and actual_route:
            actual_routes[actual_route] += 1

        proposal = signal_summary.get("proposal") if isinstance(signal_summary.get("proposal"), dict) else None
        actual_action = proposal.get("action") if proposal is not None else None
        if isinstance(actual_action, str) and actual_action:
            actual_actions[actual_action] += 1

        duration_value = result.get("duration_ms")
        if isinstance(duration_value, (int, float)):
            latency_ms.append(float(duration_value))
        latency_breakdown = result.get("latency_breakdown_ms") if isinstance(result.get("latency_breakdown_ms"), dict) else {}
        transcription_value = latency_breakdown.get("transcription_ms")
        if isinstance(transcription_value, (int, float)):
            transcription_ms.append(float(transcription_value))
        signal_value = latency_breakdown.get("signal_ms")
        if isinstance(signal_value, (int, float)):
            signal_ms.append(float(signal_value))
        postprocess_value = latency_breakdown.get("postprocess_ms")
        if isinstance(postprocess_value, (int, float)):
            postprocess_ms.append(float(postprocess_value))

        audio_capture = result.get("audio_capture") if isinstance(result.get("audio_capture"), dict) else {}
        audio_duration = audio_capture.get("duration_s")
        if isinstance(audio_duration, (int, float)):
            audio_duration_s.append(float(audio_duration))

        checks_by_name = _checks_by_name(evaluation)
        for check_name, check in checks_by_name.items():
            bucket = check_stats.setdefault(check_name, {"eligible": 0, "passed": 0})
            bucket["eligible"] += 1
            if check.get("passed") is True:
                bucket["passed"] += 1
            else:
                failed_checks[check_name] += 1

        if not passed_now:
            failed_checks_now = [
                {
                    "name": str(check.get("name") or "check"),
                    "expected": check.get("expected"),
                    "actual": check.get("actual"),
                }
                for check in checks_by_name.values()
                if check.get("passed") is not True
            ]
            failure_examples.append(
                {
                    "case_id": _case_id(result),
                    "fixture_id": fixture_id,
                    "variant_id": result.get("variant_id"),
                    "category": result.get("category"),
                    "lane": lane,
                    "source_engine": source_engine,
                    "route": actual_route,
                    "action": actual_action,
                    "transcript": _metric_actual_transcript(result, checks_by_name),
                    "failed_checks": failed_checks_now,
                }
            )

        reference_text = _metric_reference_text(result, checks_by_name)
        actual_transcript = _metric_actual_transcript(result, checks_by_name)
        if reference_text and actual_transcript is not None:
            reference_tokens = _tokenize_words(reference_text)
            hypothesis_tokens = _tokenize_words(actual_transcript)
            transcript_distance = _levenshtein_distance(reference_tokens, hypothesis_tokens)
            transcript_samples += 1
            transcript_reference_words += len(reference_tokens)
            transcript_edit_distance += transcript_distance
            if transcript_distance > 0:
                transcript_mismatches.append(
                    {
                        "case_id": _case_id(result),
                        "fixture_id": fixture_id,
                        "variant_id": result.get("variant_id"),
                        "reference": reference_text,
                        "actual": actual_transcript,
                        "edit_distance": transcript_distance,
                        "passed": passed_now,
                    }
                )

        expected_route = _metric_expected_str(result, checks_by_name, "expected_route")
        expected_action = _metric_expected_str(result, checks_by_name, "expected_action")
        reference_output_text = _metric_reference_output_text(result, checks_by_name)
        actual_output_text = _metric_actual_output_text(result, checks_by_name)
        tag_set = {tag for tag in result.get("tags", []) if isinstance(tag, str) and tag}

        if expected_route == "command" and expected_action:
            command_cases += 1
            if _checks_passed(
                checks_by_name,
                ("expected_route", "expected_action", "expected_arguments", "expected_risk"),
            ):
                command_passed += 1

        if "paraphrase" in tag_set and expected_route == "command" and expected_action:
            paraphrase_cases += 1
            if _checks_passed(
                checks_by_name,
                ("expected_route", "expected_action", "expected_arguments", "expected_risk"),
            ):
                paraphrase_passed += 1

        if "deictic" in tag_set or expected_route == "clarify":
            deictic_cases += 1
            if actual_route == "clarify" and proposal is None:
                deictic_clarified += 1

        if lane == VoiceEvalLane.DICTATION.value or "dictation" in tag_set or expected_route == "dictation":
            dictation_cases += 1
            if _checks_passed(
                checks_by_name,
                ("expected_route", "expected_action", "expected_risk", "expected_broker_decision"),
            ):
                dictation_passed += 1
            if reference_output_text is not None and actual_output_text is not None:
                output_distance = _levenshtein_distance(list(reference_output_text), list(actual_output_text))
                dictation_output_samples += 1
                dictation_output_reference_characters += len(reference_output_text)
                dictation_output_edit_distance += output_distance
                if reference_output_text == actual_output_text:
                    dictation_output_exact_matches += 1
                elif output_distance > 0:
                    output_text_mismatches.append(
                        {
                            "case_id": _case_id(result),
                            "fixture_id": fixture_id,
                            "variant_id": result.get("variant_id"),
                            "reference": reference_output_text,
                            "actual": actual_output_text,
                            "edit_distance": output_distance,
                            "passed": passed_now,
                        }
                    )

        if "translation" in tag_set or expected_route == "translation":
            translation_cases += 1
            if _checks_passed(
                checks_by_name,
                ("expected_route", "expected_action", "expected_risk", "expected_broker_decision"),
            ):
                translation_passed += 1
            if "paraphrase" in tag_set:
                translation_paraphrase_cases += 1
                if _checks_passed(
                    checks_by_name,
                    ("expected_route", "expected_action", "expected_risk", "expected_broker_decision"),
                ):
                    translation_paraphrase_passed += 1
            if reference_output_text is not None and actual_output_text is not None:
                output_distance = _levenshtein_distance(list(reference_output_text), list(actual_output_text))
                translation_output_samples += 1
                translation_output_reference_characters += len(reference_output_text)
                translation_output_edit_distance += output_distance
                if reference_output_text == actual_output_text:
                    translation_output_exact_matches += 1
                elif output_distance > 0:
                    output_text_mismatches.append(
                        {
                            "case_id": _case_id(result),
                            "fixture_id": fixture_id,
                            "variant_id": result.get("variant_id"),
                            "reference": reference_output_text,
                            "actual": actual_output_text,
                            "edit_distance": output_distance,
                            "passed": passed_now,
                        }
                    )

        if "negative" in tag_set or expected_route == "reject":
            negative_cases += 1
            if proposal is not None:
                negative_false_positives += 1

    failed = total - passed
    return {
        "fixtures_total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total), 4) if total else 0.0,
        "by_category": dict(sorted(categories.items())),
        "by_lane": dict(sorted(lanes.items())),
        "by_source_engine": dict(sorted(source_engines.items())),
        "by_tag": dict(sorted(tags.items())),
        "by_actual_route": dict(sorted(actual_routes.items())),
        "by_actual_action": dict(sorted(actual_actions.items())),
        "metrics": {
            "check_pass_rates": {
                name: {
                    "eligible": stats["eligible"],
                    "passed": stats["passed"],
                    "failed": stats["eligible"] - stats["passed"],
                    "pass_rate": round((stats["passed"] / stats["eligible"]), 4) if stats["eligible"] else 0.0,
                }
                for name, stats in sorted(check_stats.items())
            },
            "word_error_rate": {
                "samples": transcript_samples,
                "reference_words": transcript_reference_words,
                "edit_distance": transcript_edit_distance,
                "wer": round((transcript_edit_distance / transcript_reference_words), 4)
                if transcript_reference_words
                else 0.0,
            },
            "dictation_text_accuracy": _text_accuracy_metric(
                samples=dictation_output_samples,
                exact_matches=dictation_output_exact_matches,
                reference_units=dictation_output_reference_characters,
                edit_distance=dictation_output_edit_distance,
            ),
            "translation_text_accuracy": _text_accuracy_metric(
                samples=translation_output_samples,
                exact_matches=translation_output_exact_matches,
                reference_units=translation_output_reference_characters,
                edit_distance=translation_output_edit_distance,
            ),
            "translation_success_rate": _rate_metric(translation_passed, translation_cases),
            "translation_variation_score": _rate_metric(translation_paraphrase_passed, translation_paraphrase_cases),
            "command_recognition_rate": _rate_metric(command_passed, command_cases),
            "natural_variation_score": _rate_metric(paraphrase_passed, paraphrase_cases),
            "deictic_clarification_rate": _rate_metric(deictic_clarified, deictic_cases),
            "dictation_success_rate": _rate_metric(dictation_passed, dictation_cases),
            "false_positive_rate": {
                "eligible": negative_cases,
                "false_positives": negative_false_positives,
                "correct_rejections": max(negative_cases - negative_false_positives, 0),
                "rate": round((negative_false_positives / negative_cases), 4) if negative_cases else 0.0,
            },
            "latency_ms": _distribution(latency_ms),
            "latency_breakdown_ms": {
                "transcription_ms": _distribution(transcription_ms),
                "signal_ms": _distribution(signal_ms),
                "postprocess_ms": _distribution(postprocess_ms),
            },
            "audio_duration_s": _distribution(audio_duration_s),
        },
        "diagnostics": {
            "failed_checks": dict(sorted(failed_checks.items())),
            "failure_examples": failure_examples[:5],
            "transcript_mismatches": _sort_diagnostics_by_distance(transcript_mismatches)[:5],
            "output_text_mismatches": _sort_diagnostics_by_distance(output_text_mismatches)[:5],
        },
        "failing_ids": failing_ids,
    }


def render_voice_benchmark_summary(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    check_pass_rates = metrics.get("check_pass_rates") if isinstance(metrics.get("check_pass_rates"), dict) else {}
    diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), dict) else {}

    lines = [
        "Voice Benchmark Summary",
        (
            f"Fixtures   {int(summary.get('passed') or 0)}/{int(summary.get('fixtures_total') or 0)} passed "
            f"({(float(summary.get('pass_rate') or 0.0) * 100.0):.1f}%) | failed {int(summary.get('failed') or 0)}"
        ),
        (
            f"Quality    WER {_format_percent(_metric_fraction(metrics.get('word_error_rate'), 'wer'))} | "
            f"CRR {_format_percent(_metric_fraction(metrics.get('command_recognition_rate'), 'pass_rate'))} | "
            f"NVS {_format_percent(_metric_fraction(metrics.get('natural_variation_score'), 'pass_rate'))} | "
            f"Deictic {_format_percent(_metric_fraction(metrics.get('deictic_clarification_rate'), 'pass_rate'))} | "
            f"Dictation {_format_percent(_metric_fraction(metrics.get('dictation_success_rate'), 'pass_rate'))} | "
            f"FPR {_format_percent(_metric_fraction(metrics.get('false_positive_rate'), 'rate'))}"
        ),
        (
            f"Latency    median {_format_float(_metric_value(metrics.get('latency_ms'), 'median'), suffix=' ms')} | "
            f"p95 {_format_float(_metric_value(metrics.get('latency_ms'), 'p95'), suffix=' ms')} | "
            f"max {_format_float(_metric_value(metrics.get('latency_ms'), 'max'), suffix=' ms')}"
        ),
        (
            f"Audio      median {_format_float(_metric_value(metrics.get('audio_duration_s'), 'median'), suffix=' s')} | "
            f"p95 {_format_float(_metric_value(metrics.get('audio_duration_s'), 'p95'), suffix=' s')} | "
            f"max {_format_float(_metric_value(metrics.get('audio_duration_s'), 'max'), suffix=' s')}"
        ),
    ]
    latency_breakdown = metrics.get("latency_breakdown_ms") if isinstance(metrics.get("latency_breakdown_ms"), dict) else {}
    phase_parts: list[str] = []
    transcription_distribution = latency_breakdown.get("transcription_ms")
    if _metric_value(transcription_distribution, "samples"):
        phase_parts.append(f"speech {_format_float(_metric_value(transcription_distribution, 'median'), suffix=' ms')}")
    signal_distribution = latency_breakdown.get("signal_ms")
    if _metric_value(signal_distribution, "samples"):
        phase_parts.append(f"signal {_format_float(_metric_value(signal_distribution, 'median'), suffix=' ms')}")
    postprocess_distribution = latency_breakdown.get("postprocess_ms")
    if _metric_value(postprocess_distribution, "samples"):
        phase_parts.append(f"post {_format_float(_metric_value(postprocess_distribution, 'median'), suffix=' ms')}")
    if phase_parts:
        lines.append(f"Phases     {' | '.join(phase_parts)}")

    dictation_text_accuracy = metrics.get("dictation_text_accuracy")
    translation_text_accuracy = metrics.get("translation_text_accuracy")
    if _metric_value(dictation_text_accuracy, "samples") or _metric_value(translation_text_accuracy, "samples"):
        parts: list[str] = []
        if _metric_value(dictation_text_accuracy, "samples"):
            parts.append(f"Dict {_format_percent(_metric_fraction(dictation_text_accuracy, 'accuracy'))}")
        if _metric_value(translation_text_accuracy, "samples"):
            parts.append(f"Trans {_format_percent(_metric_fraction(translation_text_accuracy, 'accuracy'))}")
        lines.append(f"Text Out   {' | '.join(parts)}")

    if _metric_value(metrics.get("translation_success_rate"), "eligible"):
        lines.append(
            "Translate "
            f"route {_format_percent(_metric_fraction(metrics.get('translation_success_rate'), 'pass_rate'))}"
            + (
                f" | variation {_format_percent(_metric_fraction(metrics.get('translation_variation_score'), 'pass_rate'))}"
                if _metric_value(metrics.get("translation_variation_score"), "eligible")
                else ""
            )
        )

    reference_transcript = check_pass_rates.get("reference_transcript")
    if isinstance(reference_transcript, dict):
        lines.append(
            "Transcript "
            f"exact-match {_format_percent(_metric_fraction(reference_transcript, 'pass_rate'))} "
            f"({int(reference_transcript.get('passed') or 0)}/{int(reference_transcript.get('eligible') or 0)})"
        )

    route_counts = _format_count_mapping(summary.get("by_actual_route"))
    if route_counts:
        lines.append(f"Routes     {route_counts}")

    source_counts = _format_count_mapping(summary.get("by_source_engine"))
    if source_counts:
        lines.append(f"Sources    {source_counts}")

    failed_checks = _format_count_mapping(diagnostics.get("failed_checks"))
    if failed_checks:
        lines.append(f"Checks     {failed_checks}")

    failure_examples = diagnostics.get("failure_examples")
    if isinstance(failure_examples, list) and failure_examples:
        lines.append(
            "Example    "
            + " | ".join(
                _format_failure_example(example)
                for example in failure_examples[:2]
                if isinstance(example, dict)
            )
        )

    drift_summary = _format_drift_summary(diagnostics)
    if drift_summary:
        lines.append(f"Drift      {drift_summary}")

    failing_ids = summary.get("failing_ids")
    if isinstance(failing_ids, list) and failing_ids:
        preview = ", ".join(str(item) for item in failing_ids[:5])
        if len(failing_ids) > 5:
            preview += f", +{len(failing_ids) - 5} more"
        lines.append(f"Failing    {preview}")

    return "\n".join(lines)


def summarize_voice_benchmark_comparison(
    summaries: dict[str, dict[str, Any]],
    *,
    baseline_label: str | None = None,
) -> dict[str, Any]:
    labels = [
        label
        for label, summary in summaries.items()
        if isinstance(label, str) and label and isinstance(summary, dict) and summary
    ]
    if not labels:
        return {
            "labels": [],
            "baseline_label": None,
            "runs": {},
            "metrics": {},
            "deltas": {},
        }

    resolved_baseline = baseline_label if baseline_label in labels else labels[0]
    metrics: dict[str, dict[str, Any]] = {}
    for name, path, better, unit, label, samples_path in _comparison_metric_specs():
        values: dict[str, float] = {}
        samples: dict[str, int] = {}
        for run_label in labels:
            summary = summaries.get(run_label)
            value = _resolve_path(summary, path)
            if not isinstance(value, (int, float)):
                continue
            values[run_label] = round(float(value), 4)
            if samples_path is None:
                continue
            sample_value = _resolve_path(summary, samples_path)
            if isinstance(sample_value, (int, float)):
                samples[run_label] = int(sample_value)
        if not values:
            continue
        if samples_path is not None and max(samples.values(), default=0) <= 0:
            continue

        best_value = min(values.values()) if better == "lower" else max(values.values())
        best_labels = [
            run_label
            for run_label, value in values.items()
            if math.isclose(
                value,
                best_value,
                abs_tol=(0.01 if unit == "ms" else 0.0001),
            )
        ]
        metrics[name] = {
            "label": label,
            "path": path,
            "better": better,
            "unit": unit,
            "values": values,
            "samples": samples,
            "best_value": round(best_value, 4),
            "best_labels": best_labels,
            "spread": round(max(values.values()) - min(values.values()), 4),
        }

    deltas = {
        run_label: compare_voice_benchmark_summaries(
            summaries.get(resolved_baseline),
            summaries.get(run_label) or {},
        )
        for run_label in labels
        if run_label != resolved_baseline
    }
    baseline_failures = _summary_failing_ids(summaries.get(resolved_baseline))
    failure_deltas = {
        run_label: {
            "fixed": sorted(baseline_failures - _summary_failing_ids(summaries.get(run_label))),
            "regressed": sorted(_summary_failing_ids(summaries.get(run_label)) - baseline_failures),
        }
        for run_label in labels
        if run_label != resolved_baseline
    }

    return {
        "labels": labels,
        "baseline_label": resolved_baseline,
        "runs": {
            run_label: {
                "fixtures_total": int((summaries.get(run_label) or {}).get("fixtures_total") or 0),
                "passed": int((summaries.get(run_label) or {}).get("passed") or 0),
                "failed": int((summaries.get(run_label) or {}).get("failed") or 0),
                "pass_rate": round(float((summaries.get(run_label) or {}).get("pass_rate") or 0.0), 4),
                "failing_ids": sorted(_summary_failing_ids(summaries.get(run_label))),
            }
            for run_label in labels
        },
        "metrics": metrics,
        "deltas": deltas,
        "failure_deltas": failure_deltas,
    }


def render_voice_benchmark_comparison(comparison: dict[str, Any]) -> str:
    labels = comparison.get("labels") if isinstance(comparison.get("labels"), list) else []
    if not labels:
        return "\n".join(
            [
                "Voice Benchmark Comparison",
                "Runs       none",
            ]
        )

    runs = comparison.get("runs") if isinstance(comparison.get("runs"), dict) else {}
    metrics = comparison.get("metrics") if isinstance(comparison.get("metrics"), dict) else {}
    baseline_label = comparison.get("baseline_label")
    lines = [
        "Voice Benchmark Comparison",
        f"Runs       {', '.join(str(label) for label in labels)}",
    ]

    fixture_parts: list[str] = []
    for label in labels:
        run = runs.get(label)
        if not isinstance(run, dict):
            continue
        fixture_parts.append(
            f"{label} {int(run.get('passed') or 0)}/{int(run.get('fixtures_total') or 0)}"
        )
    if fixture_parts:
        lines.append(f"Fixtures   {' | '.join(fixture_parts)}")

    for metric_name in (
        "pass_rate",
        "word_error_rate",
        "command_recognition_rate",
        "natural_variation_score",
        "deictic_clarification_rate",
        "dictation_success_rate",
        "false_positive_rate",
        "dictation_text_accuracy",
        "translation_success_rate",
        "translation_variation_score",
        "translation_text_accuracy",
        "latency_median_ms",
        "latency_p95_ms",
    ):
        metric = metrics.get(metric_name)
        if not isinstance(metric, dict):
            continue
        values = metric.get("values")
        if not isinstance(values, dict) or not values:
            continue
        parts = [
            f"{label} {_format_comparison_metric_value(values.get(label), unit=str(metric.get('unit') or 'ratio'))}"
            for label in labels
            if isinstance(values.get(label), (int, float))
        ]
        if not parts:
            continue
        best_labels = metric.get("best_labels")
        if isinstance(best_labels, list) and best_labels:
            parts.append(f"best {', '.join(str(label) for label in best_labels)}")
        lines.append(f"{str(metric.get('label') or metric_name):<11}{' | '.join(parts)}")

    deltas = comparison.get("deltas") if isinstance(comparison.get("deltas"), dict) else {}
    if isinstance(baseline_label, str) and baseline_label and deltas:
        delta_parts = []
        for label in labels:
            if label == baseline_label:
                continue
            delta = deltas.get(label)
            if not isinstance(delta, dict):
                continue
            delta_parts.append(f"{label} {str(delta.get('status') or 'unchanged')}")
        if delta_parts:
            lines.append(f"Delta      vs {baseline_label} | " + " | ".join(delta_parts))

    failure_deltas = comparison.get("failure_deltas") if isinstance(comparison.get("failure_deltas"), dict) else {}
    if isinstance(baseline_label, str) and baseline_label and failure_deltas:
        failure_parts = []
        for label in labels:
            if label == baseline_label:
                continue
            payload = failure_deltas.get(label)
            if not isinstance(payload, dict):
                continue
            fixed = payload.get("fixed") if isinstance(payload.get("fixed"), list) else []
            regressed = payload.get("regressed") if isinstance(payload.get("regressed"), list) else []
            parts: list[str] = []
            if fixed:
                parts.append(f"fixed {_summarize_case_ids(fixed)}")
            if regressed:
                parts.append(f"regressed {_summarize_case_ids(regressed)}")
            if parts:
                failure_parts.append(f"{label} {' | '.join(parts)}")
        if failure_parts:
            lines.append("Cases      " + " | ".join(failure_parts))

    return "\n".join(lines)


def compare_voice_benchmark_summaries(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(previous, dict) or not previous:
        return {
            "has_baseline": False,
            "status": "no_baseline",
            "metrics": {},
        }

    metric_specs = (
        ("pass_rate", "pass_rate", "higher", "ratio", 0.0001),
        ("word_error_rate", "metrics.word_error_rate.wer", "lower", "ratio", 0.001),
        ("command_recognition_rate", "metrics.command_recognition_rate.pass_rate", "higher", "ratio", 0.001),
        ("natural_variation_score", "metrics.natural_variation_score.pass_rate", "higher", "ratio", 0.001),
        ("deictic_clarification_rate", "metrics.deictic_clarification_rate.pass_rate", "higher", "ratio", 0.001),
        ("dictation_success_rate", "metrics.dictation_success_rate.pass_rate", "higher", "ratio", 0.001),
        ("translation_success_rate", "metrics.translation_success_rate.pass_rate", "higher", "ratio", 0.001),
        ("translation_variation_score", "metrics.translation_variation_score.pass_rate", "higher", "ratio", 0.001),
        ("false_positive_rate", "metrics.false_positive_rate.rate", "lower", "ratio", 0.001),
        ("dictation_text_accuracy", "metrics.dictation_text_accuracy.accuracy", "higher", "ratio", 0.001),
        ("translation_text_accuracy", "metrics.translation_text_accuracy.accuracy", "higher", "ratio", 0.001),
        ("latency_median_ms", "metrics.latency_ms.median", "lower", "ms", 5.0),
        ("latency_p95_ms", "metrics.latency_ms.p95", "lower", "ms", 5.0),
        ("latency_max_ms", "metrics.latency_ms.max", "lower", "ms", 5.0),
    )

    metrics: dict[str, dict[str, Any]] = {}
    improved = 0
    regressed = 0
    unchanged = 0
    for name, path, better, unit, tolerance in metric_specs:
        previous_value = _resolve_path(previous, path)
        current_value = _resolve_path(current, path)
        if not isinstance(previous_value, (int, float)) or not isinstance(current_value, (int, float)):
            continue
        delta = float(current_value) - float(previous_value)
        if math.isclose(delta, 0.0, abs_tol=tolerance):
            direction = "unchanged"
            unchanged += 1
        elif (better == "higher" and delta > 0.0) or (better == "lower" and delta < 0.0):
            direction = "improved"
            improved += 1
        else:
            direction = "regressed"
            regressed += 1
        metrics[name] = {
            "previous": round(float(previous_value), 4),
            "current": round(float(current_value), 4),
            "delta": round(delta, 4),
            "direction": direction,
            "better": better,
            "unit": unit,
            "tolerance": tolerance,
        }

    status = "unchanged"
    if regressed and improved:
        status = "mixed"
    elif regressed:
        status = "regressed"
    elif improved:
        status = "improved"

    return {
        "has_baseline": True,
        "status": status,
        "metrics": metrics,
        "improved_metrics": improved,
        "regressed_metrics": regressed,
        "unchanged_metrics": unchanged,
    }


def render_voice_benchmark_delta_summary(delta: dict[str, Any]) -> str:
    if not isinstance(delta, dict) or delta.get("has_baseline") is not True:
        return "\n".join(
            [
                "Voice Benchmark Delta",
                "Delta      no previous baseline",
            ]
        )

    metrics = delta.get("metrics") if isinstance(delta.get("metrics"), dict) else {}
    lines = [
        "Voice Benchmark Delta",
        (
            f"Delta      {str(delta.get('status') or 'unchanged')} | "
            f"{int(delta.get('improved_metrics') or 0)} improved | "
            f"{int(delta.get('regressed_metrics') or 0)} regressed | "
            f"{int(delta.get('unchanged_metrics') or 0)} unchanged"
        ),
    ]

    quality_items = _format_delta_group(
        metrics,
        ("word_error_rate", "command_recognition_rate", "false_positive_rate"),
        labels={"word_error_rate": "WER", "command_recognition_rate": "CRR", "false_positive_rate": "FPR"},
    )
    if quality_items:
        lines.append(f"Quality    {' | '.join(quality_items)}")

    text_items = _format_delta_group(
        metrics,
        ("dictation_text_accuracy", "translation_text_accuracy"),
        labels={"dictation_text_accuracy": "Dict", "translation_text_accuracy": "Trans"},
    )
    if text_items:
        lines.append(f"Text Out   {' | '.join(text_items)}")

    latency_items = _format_delta_group(
        metrics,
        ("latency_median_ms", "latency_p95_ms", "latency_max_ms"),
        labels={"latency_median_ms": "median", "latency_p95_ms": "p95", "latency_max_ms": "max"},
    )
    if latency_items:
        lines.append(f"Latency    {' | '.join(latency_items)}")

    return "\n".join(lines)


def evaluate_voice_benchmark_budget(
    summary: dict[str, Any],
    budget: VoiceBenchmarkBudget,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    skipped = 0

    for check in budget.checks:
        samples: int | None = None
        samples_path = check.samples_path or _default_samples_path(check.path)
        if check.min_samples is not None and samples_path is not None:
            samples_value = _resolve_path(summary, samples_path)
            samples = int(samples_value) if isinstance(samples_value, (int, float)) else None
            if samples is None or samples < check.min_samples:
                skipped += 1
                checks.append(
                    {
                        "name": check.name,
                        "path": check.path,
                        "status": "skipped",
                        "actual": _resolve_path(summary, check.path),
                        "min": check.min,
                        "max": check.max,
                        "samples": samples,
                        "min_samples": check.min_samples,
                        "samples_path": samples_path,
                        "notes": check.notes,
                        "reason": (
                            f"samples {samples if samples is not None else 'missing'} "
                            f"below minimum {check.min_samples}"
                        ),
                    }
                )
                continue

        actual = _resolve_path(summary, check.path)
        if not isinstance(actual, (int, float)):
            failed += 1
            checks.append(
                {
                    "name": check.name,
                    "path": check.path,
                    "status": "failed",
                    "actual": actual,
                    "min": check.min,
                    "max": check.max,
                    "samples": samples,
                    "min_samples": check.min_samples,
                    "samples_path": samples_path,
                    "notes": check.notes,
                    "reason": "metric missing or non-numeric",
                }
            )
            continue

        reasons: list[str] = []
        if check.min is not None and float(actual) < float(check.min):
            reasons.append(f"{float(actual):.4f} < min {float(check.min):.4f}")
        if check.max is not None and float(actual) > float(check.max):
            reasons.append(f"{float(actual):.4f} > max {float(check.max):.4f}")

        status = "passed" if not reasons else "failed"
        if status == "passed":
            passed += 1
        else:
            failed += 1

        checks.append(
            {
                "name": check.name,
                "path": check.path,
                "status": status,
                "actual": round(float(actual), 4),
                "min": check.min,
                "max": check.max,
                "samples": samples,
                "min_samples": check.min_samples,
                "samples_path": samples_path,
                "notes": check.notes,
                "reason": "; ".join(reasons) if reasons else None,
            }
        )

    return {
        "budget_id": budget.budget_id or "voice-benchmark-budget",
        "description": budget.description,
        "passed": failed == 0,
        "passed_checks": passed,
        "failed_checks": failed,
        "skipped_checks": skipped,
        "checks": checks,
    }


def render_voice_benchmark_budget_summary(result: dict[str, Any]) -> str:
    checks = result.get("checks") if isinstance(result.get("checks"), list) else []
    lines = [
        "Voice Benchmark Budget",
        (
            f"Budget     {result.get('budget_id') or 'voice-benchmark-budget'} | "
            f"{'passed' if result.get('passed') is True else 'failed'}"
        ),
        (
            f"Checks     {int(result.get('passed_checks') or 0)} passed | "
            f"{int(result.get('failed_checks') or 0)} failed | "
            f"{int(result.get('skipped_checks') or 0)} skipped"
        ),
    ]

    failed = [check for check in checks if isinstance(check, dict) and check.get("status") == "failed"]
    if failed:
        lines.append(
            "Failed     "
            + ", ".join(
                f"{check.get('name')} ({check.get('reason') or 'failed'})"
                for check in failed[:5]
            )
        )

    skipped = [check for check in checks if isinstance(check, dict) and check.get("status") == "skipped"]
    if skipped:
        lines.append(
            "Skipped    "
            + ", ".join(
                f"{check.get('name')} ({check.get('reason') or 'skipped'})"
                for check in skipped[:5]
            )
        )

    return "\n".join(lines)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _check(name: str, expected: Any, actual: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": passed,
    }


def _matching_message(messages: list[str], expected_fragment: str) -> str | None:
    lowered_fragment = expected_fragment.lower()
    for message in messages:
        if lowered_fragment in message.lower():
            return message
    return None


def _checks_by_name(evaluation: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(evaluation, dict):
        return {}
    checks = evaluation.get("checks")
    if not isinstance(checks, list):
        return {}
    payload: dict[str, dict[str, Any]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = check.get("name")
        if isinstance(name, str) and name:
            payload[name] = check
    return payload


def _metric_reference_text(
    result: dict[str, Any],
    checks_by_name: dict[str, dict[str, Any]],
) -> str | None:
    value = result.get("reference_transcript")
    if isinstance(value, str) and value.strip():
        return value
    check = checks_by_name.get("reference_transcript")
    if isinstance(check, dict):
        expected = check.get("expected")
        if isinstance(expected, str) and expected.strip():
            return expected
    prompt_text = result.get("prompt_text")
    if isinstance(prompt_text, str) and prompt_text.strip():
        return prompt_text
    return None


def _metric_reference_output_text(
    result: dict[str, Any],
    checks_by_name: dict[str, dict[str, Any]],
) -> str | None:
    value = result.get("reference_output_text")
    if isinstance(value, str):
        return value
    check = checks_by_name.get("reference_output_text")
    if isinstance(check, dict):
        expected = check.get("expected")
        if isinstance(expected, str):
            return expected
    return None


def _metric_actual_transcript(
    result: dict[str, Any],
    checks_by_name: dict[str, dict[str, Any]],
) -> str | None:
    signal_summary = result.get("signal_summary")
    if isinstance(signal_summary, dict):
        transcript = signal_summary.get("transcript")
        if isinstance(transcript, str):
            return transcript
    check = checks_by_name.get("reference_transcript")
    if isinstance(check, dict):
        actual = check.get("actual")
        if isinstance(actual, str):
            return actual
    audio_capture = result.get("audio_capture")
    if isinstance(audio_capture, dict):
        transcript = audio_capture.get("transcript")
        if isinstance(transcript, str):
            return transcript
    return None


def _metric_actual_output_text(
    result: dict[str, Any],
    checks_by_name: dict[str, dict[str, Any]],
) -> str | None:
    signal_summary = result.get("signal_summary")
    if isinstance(signal_summary, dict):
        output_text = _summary_output_text(signal_summary)
        if output_text is not None:
            return output_text
    check = checks_by_name.get("reference_output_text")
    if isinstance(check, dict):
        actual = check.get("actual")
        if isinstance(actual, str):
            return actual
    return None


def _metric_expected_str(
    result: dict[str, Any],
    checks_by_name: dict[str, dict[str, Any]],
    name: str,
) -> str | None:
    value = result.get(name)
    if isinstance(value, str) and value.strip():
        return value
    check = checks_by_name.get(name)
    if not isinstance(check, dict):
        return None
    expected = check.get("expected")
    if isinstance(expected, str) and expected.strip():
        return expected
    return None


def _checks_passed(
    checks_by_name: dict[str, dict[str, Any]],
    names: tuple[str, ...],
) -> bool:
    matched = [checks_by_name[name] for name in names if name in checks_by_name]
    return bool(matched) and all(check.get("passed") is True for check in matched)


def _rate_metric(passed: int, eligible: int) -> dict[str, Any]:
    return {
        "eligible": eligible,
        "passed": passed,
        "failed": max(eligible - passed, 0),
        "pass_rate": round((passed / eligible), 4) if eligible else 0.0,
    }


def _text_accuracy_metric(
    *,
    samples: int,
    exact_matches: int,
    reference_units: int,
    edit_distance: int,
) -> dict[str, Any]:
    accuracy = max(0.0, 1.0 - (edit_distance / reference_units)) if reference_units else 0.0
    exact_match_rate = (exact_matches / samples) if samples else 0.0
    return {
        "samples": samples,
        "exact_matches": exact_matches,
        "reference_characters": reference_units,
        "edit_distance": edit_distance,
        "accuracy": round(accuracy, 4),
        "exact_match_rate": round(exact_match_rate, 4),
    }


def _metric_fraction(payload: Any, key: str) -> float:
    if not isinstance(payload, dict):
        return 0.0
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _metric_value(payload: Any, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _format_float(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.2f}{suffix}"


def _format_count_mapping(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    items: list[str] = []
    for key, value in sorted(payload.items()):
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            continue
        items.append(f"{key}={int(value)}")
    return ", ".join(items)


def _format_failure_example(payload: dict[str, Any]) -> str:
    case_id = str(payload.get("case_id") or payload.get("fixture_id") or "fixture")
    checks = payload.get("failed_checks")
    if not isinstance(checks, list) or not checks:
        return case_id
    first = checks[0]
    if not isinstance(first, dict):
        return case_id
    name = str(first.get("name") or "check")
    expected = _compact_value(first.get("expected"))
    actual = _compact_value(first.get("actual"))
    rendered = f"{case_id} {name}"
    if expected != "-" or actual != "-":
        rendered += f" {expected} -> {actual}"
    return rendered


def _format_delta_group(
    metrics: dict[str, Any],
    names: tuple[str, ...],
    *,
    labels: dict[str, str],
) -> list[str]:
    items: list[str] = []
    for name in names:
        payload = metrics.get(name)
        if not isinstance(payload, dict):
            continue
        label = labels.get(name, name)
        items.append(f"{label} {_format_delta_value(payload)}")
    return items


def _format_delta_value(payload: dict[str, Any]) -> str:
    direction = payload.get("direction")
    if direction == "unchanged":
        return "stable"
    delta = payload.get("delta")
    unit = payload.get("unit")
    if not isinstance(delta, (int, float)):
        return "-"
    if unit == "ms":
        return f"{float(delta):+.2f} ms"
    return f"{float(delta) * 100.0:+.1f} pts"


def _comparison_metric_specs() -> tuple[tuple[str, str, str, str, str, str | None], ...]:
    return (
        ("pass_rate", "pass_rate", "higher", "ratio", "Pass", "fixtures_total"),
        ("word_error_rate", "metrics.word_error_rate.wer", "lower", "ratio", "WER", "metrics.word_error_rate.samples"),
        (
            "command_recognition_rate",
            "metrics.command_recognition_rate.pass_rate",
            "higher",
            "ratio",
            "CRR",
            "metrics.command_recognition_rate.eligible",
        ),
        (
            "natural_variation_score",
            "metrics.natural_variation_score.pass_rate",
            "higher",
            "ratio",
            "NVS",
            "metrics.natural_variation_score.eligible",
        ),
        (
            "deictic_clarification_rate",
            "metrics.deictic_clarification_rate.pass_rate",
            "higher",
            "ratio",
            "Deictic",
            "metrics.deictic_clarification_rate.eligible",
        ),
        (
            "dictation_success_rate",
            "metrics.dictation_success_rate.pass_rate",
            "higher",
            "ratio",
            "Dictation",
            "metrics.dictation_success_rate.eligible",
        ),
        (
            "false_positive_rate",
            "metrics.false_positive_rate.rate",
            "lower",
            "ratio",
            "FPR",
            "metrics.false_positive_rate.eligible",
        ),
        (
            "dictation_text_accuracy",
            "metrics.dictation_text_accuracy.accuracy",
            "higher",
            "ratio",
            "Dict Text",
            "metrics.dictation_text_accuracy.samples",
        ),
        (
            "translation_success_rate",
            "metrics.translation_success_rate.pass_rate",
            "higher",
            "ratio",
            "Translate",
            "metrics.translation_success_rate.eligible",
        ),
        (
            "translation_variation_score",
            "metrics.translation_variation_score.pass_rate",
            "higher",
            "ratio",
            "Trans Var",
            "metrics.translation_variation_score.eligible",
        ),
        (
            "translation_text_accuracy",
            "metrics.translation_text_accuracy.accuracy",
            "higher",
            "ratio",
            "Trans Text",
            "metrics.translation_text_accuracy.samples",
        ),
        ("latency_median_ms", "metrics.latency_ms.median", "lower", "ms", "Median", "metrics.latency_ms.samples"),
        ("latency_p95_ms", "metrics.latency_ms.p95", "lower", "ms", "p95", "metrics.latency_ms.samples"),
    )


def _format_comparison_metric_value(value: Any, *, unit: str) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if unit == "ms":
        return _format_float(float(value), suffix=" ms")
    return _format_percent(float(value))


def _format_drift_summary(diagnostics: dict[str, Any]) -> str:
    parts: list[str] = []
    transcript_mismatches = diagnostics.get("transcript_mismatches")
    if isinstance(transcript_mismatches, list) and transcript_mismatches:
        parts.append(
            "transcript "
            + ", ".join(
                _format_drift_case(item)
                for item in transcript_mismatches[:2]
                if isinstance(item, dict)
            )
        )
    output_text_mismatches = diagnostics.get("output_text_mismatches")
    if isinstance(output_text_mismatches, list) and output_text_mismatches:
        parts.append(
            "output "
            + ", ".join(
                _format_drift_case(item)
                for item in output_text_mismatches[:2]
                if isinstance(item, dict)
            )
        )
    return " | ".join(parts)


def _format_drift_case(payload: dict[str, Any]) -> str:
    case_id = str(payload.get("case_id") or payload.get("fixture_id") or "fixture")
    reference = _compact_value(payload.get("reference"))
    actual = _compact_value(payload.get("actual"))
    return f"{case_id} {reference} -> {actual}"


def _summary_failing_ids(summary: Any) -> set[str]:
    if not isinstance(summary, dict):
        return set()
    failing_ids = summary.get("failing_ids")
    if not isinstance(failing_ids, list):
        return set()
    return {str(item) for item in failing_ids if isinstance(item, str) and item}


def _summarize_case_ids(case_ids: list[str], *, limit: int = 3) -> str:
    counts = Counter(_case_root(case_id) for case_id in case_ids if isinstance(case_id, str) and case_id)
    if not counts:
        return "-"
    rendered = [
        f"{case_id} x{count}" if count > 1 else case_id
        for case_id, count in sorted(counts.items())
    ]
    preview = ", ".join(rendered[:limit])
    if len(rendered) > limit:
        preview += f", +{len(rendered) - limit} more"
    return preview


def _case_root(case_id: str) -> str:
    return case_id.split(":", 1)[0]


def _compact_value(value: Any, *, limit: int = 36) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        normalized = " ".join(value.split())
        if len(normalized) > limit:
            return normalized[: limit - 1] + "…"
        return normalized or "-"
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(rendered) > limit:
        return rendered[: limit - 1] + "…"
    return rendered


def _summary_output_text(summary: dict[str, Any]) -> str | None:
    proposal = summary.get("proposal")
    if not isinstance(proposal, dict):
        return None
    arguments = proposal.get("arguments")
    if not isinstance(arguments, dict):
        return None
    text = arguments.get("text")
    if isinstance(text, str):
        return text
    return None


def _default_samples_path(path: str) -> str | None:
    parts = [segment for segment in path.split(".") if segment]
    if len(parts) < 2:
        return None
    return ".".join([*parts[:-1], "samples"])


def _resolve_path(payload: Any, path: str) -> Any:
    current = payload
    for segment in path.split("."):
        if not segment:
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "min": round(ordered[0], 2),
        "median": round(float(statistics.median(ordered)), 2),
        "mean": round(float(statistics.mean(ordered)), 2),
        "p95": round(float(_percentile_nearest_rank(ordered, 0.95)), 2),
        "max": round(ordered[-1], 2),
    }


def _case_id(result: dict[str, Any]) -> str:
    fixture_id = str(result.get("fixture_id") or "voice-fixture")
    variant_id = result.get("variant_id")
    if not isinstance(variant_id, str) or not variant_id or variant_id == "default":
        return fixture_id
    return f"{fixture_id}:{variant_id}"


def _sort_diagnostics_by_distance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            -int(item.get("edit_distance") or 0),
            str(item.get("case_id") or item.get("fixture_id") or ""),
        ),
    )


def _percentile_nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    index = max(0, math.ceil(len(values) * quantile) - 1)
    return values[index]


def _tokenize_words(value: str) -> list[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []
    return normalized.split(" ")


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _infer_route(
    proposal: dict[str, Any] | None,
    message: str | None,
    messages: list[str] | None = None,
) -> str | None:
    all_messages = " ".join((messages or [])).lower()
    if proposal is not None:
        action = proposal.get("action")
        if action == ActionName.TYPE_TEXT.value:
            arguments = proposal.get("arguments")
            if isinstance(arguments, dict):
                voice_route = arguments.get("voice_route")
                if isinstance(voice_route, str) and voice_route:
                    return voice_route
            lowered_message = (message or "").lower()
            if "translation" in lowered_message or "translation" in all_messages:
                return "translation"
            if "dictation" in lowered_message or "dictation" in all_messages:
                return "dictation"
            return "command"
        return "command"

    lowered_message = (message or "").lower()
    if "dictation" in lowered_message:
        return "dictation"
    if "translation" in lowered_message:
        return "translation"
    if "need" in lowered_message and "before i can" in lowered_message:
        return "clarify"
    if "could not map" in lowered_message or "outside the current bounded command set" in lowered_message:
        return "reject"
    return None
