"""Synthetic voice fixture generation helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any

from .voice_eval import VoiceEvalManifest, VoiceFixtureVariant, discover_voice_eval_manifests


class VoiceFixtureGenerationError(RuntimeError):
    """Raised when a synthetic voice fixture cannot be generated."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def generate_voice_fixture_corpus(
    fixtures_path: str | Path,
    *,
    manifest_glob: str = "*.json",
    output_root: str | Path | None = None,
    force: bool = False,
    command_runner=None,
) -> list[dict[str, Any]]:
    manifests = discover_voice_eval_manifests(fixtures_path, manifest_glob=manifest_glob)
    results: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = VoiceEvalManifest.load(manifest_path)
        variants = manifest.resolved_variants(manifest_path)
        if not variants:
            results.append(
                {
                    "status": "skipped",
                    "fixture_id": manifest.resolved_fixture_id(manifest_path),
                    "manifest_path": str(manifest_path),
                    "reason": "no_variants",
                }
            )
            continue
        for variant in variants:
            try:
                results.append(
                    generate_voice_fixture_variant(
                        manifest_path=manifest_path,
                        manifest=manifest,
                        variant=variant,
                        output_root=output_root,
                        force=force,
                        command_runner=command_runner,
                    )
                )
            except VoiceFixtureGenerationError as error:
                results.append(
                    {
                        "status": "error",
                        "fixture_id": manifest.resolved_fixture_id(manifest_path),
                        "variant_id": variant.variant_id,
                        "manifest_path": str(Path(manifest_path).resolve()),
                        "audio_path": str(_resolved_output_path(manifest_path, variant, output_root)),
                        "source_engine": variant.source_engine or manifest.source_engine or "macos_say",
                        "voice_id": variant.voice_id or manifest.voice_id,
                        "speaking_rate": variant.speaking_rate or manifest.speaking_rate,
                        "message": str(error),
                    }
                )
    return results


def generate_voice_fixture_variant(
    *,
    manifest_path: str | Path,
    manifest: VoiceEvalManifest,
    variant: VoiceFixtureVariant,
    output_root: str | Path | None = None,
    force: bool = False,
    command_runner=None,
) -> dict[str, Any]:
    runner = command_runner or _run_command
    prompt_text = (manifest.prompt_text or manifest.reference_transcript or "").strip()
    if not prompt_text:
        raise VoiceFixtureGenerationError(
            f'Fixture "{manifest.resolved_fixture_id(manifest_path)}" is missing prompt_text.'
        )

    source_engine = (variant.source_engine or manifest.source_engine or "macos_say").strip().lower()
    if source_engine not in {"macos_say", "piper", "piper_tts"}:
        return {
            "status": "skipped",
            "fixture_id": manifest.resolved_fixture_id(manifest_path),
            "variant_id": variant.variant_id,
            "manifest_path": str(Path(manifest_path).resolve()),
            "reason": "unsupported_source_engine",
            "source_engine": source_engine,
        }

    probe = _resolve_first_binary(("ffprobe",))
    output_path = _resolved_output_path(manifest_path, variant, output_root)

    metadata = {
        "fixture_id": manifest.resolved_fixture_id(manifest_path),
        "category": manifest.category,
        "lane": manifest.lane.value if manifest.lane is not None else None,
        "manifest_path": str(Path(manifest_path).resolve()),
        "prompt_text": prompt_text,
        "reference_transcript": manifest.reference_transcript,
        "source_engine": source_engine,
        "variant_id": variant.variant_id,
        "voice_id": variant.voice_id or manifest.voice_id,
        "speaking_rate": variant.speaking_rate or manifest.speaking_rate,
        "background_noise": variant.background_noise or manifest.background_noise,
        "noise_level_db": variant.noise_level_db if variant.noise_level_db is not None else manifest.noise_level_db,
        "tags": list(manifest.tags),
        "audio_path": str(output_path),
    }
    background_noise = (variant.background_noise or manifest.background_noise or "").strip().lower() or None
    noise_level_db = variant.noise_level_db if variant.noise_level_db is not None else manifest.noise_level_db

    if output_path.exists() and not force:
        duration_s = _probe_duration(output_path, probe, runner)
        if duration_s > 0.0:
            metadata["duration_s"] = duration_s
            metadata["status"] = "skipped"
            metadata["reason"] = "exists"
            metadata_path = _metadata_path(output_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return metadata
    intermediate_output_path = output_path
    if background_noise is not None:
        if background_noise not in {"white", "pink", "brown", "violet"}:
            raise VoiceFixtureGenerationError(
                f'Unsupported background noise profile "{background_noise}". '
                "Use one of: white, pink, brown, violet."
            )
        temp_handle = NamedTemporaryFile(suffix=".wav", delete=False)
        temp_handle.close()
        intermediate_output_path = Path(temp_handle.name)

    try:
        if source_engine == "macos_say":
            _require_binary("say")
            converter = _resolve_first_binary(("ffmpeg",))
            _generate_with_macos_say(
                prompt_text=prompt_text,
                output_path=intermediate_output_path,
                voice_id=variant.voice_id or manifest.voice_id,
                speaking_rate=variant.speaking_rate or manifest.speaking_rate,
                converter=converter,
                probe=probe,
                runner=runner,
            )
        else:
            _generate_with_piper(
                prompt_text=prompt_text,
                output_path=intermediate_output_path,
                voice_id=variant.voice_id or manifest.voice_id,
                probe=probe,
                runner=runner,
            )

        if background_noise is not None:
            ffmpeg_bin = _resolve_first_binary(("ffmpeg",))
            _mix_background_noise(
                base_audio_path=intermediate_output_path,
                output_path=output_path,
                noise_profile=background_noise,
                noise_level_db=float(noise_level_db if noise_level_db is not None else -30.0),
                ffmpeg_binary=ffmpeg_bin,
                probe=probe,
                runner=runner,
            )
    finally:
        if intermediate_output_path != output_path:
            intermediate_output_path.unlink(missing_ok=True)

    duration_s = _probe_duration(output_path, probe, runner)
    if duration_s <= 0.0:
        raise VoiceFixtureGenerationError(f'Generated WAV has no duration: "{output_path}"')

    metadata["duration_s"] = duration_s
    metadata["status"] = "generated"
    metadata_path = _metadata_path(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _generate_with_macos_say(
    *,
    prompt_text: str,
    output_path: Path,
    voice_id: str | None,
    speaking_rate: int | None,
    converter: str,
    probe: str,
    runner,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".aiff", delete=False) as handle:
        intermediate_path = Path(handle.name)

    try:
        say_command = ["say"]
        if voice_id:
            say_command.extend(["-v", voice_id])
        if speaking_rate:
            say_command.extend(["-r", str(int(speaking_rate))])
        say_command.extend(["-o", str(intermediate_path), prompt_text])
        runner(say_command)
        _verify_nonempty_file(intermediate_path, label="macOS say output")

        runner(
            [
                converter,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(intermediate_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )
        _verify_nonempty_file(output_path, label="generated wav output")
        duration_s = _probe_duration(output_path, probe, runner)
        if duration_s <= 0.0:
            raise VoiceFixtureGenerationError(
                "macOS say created an audio file shell with zero audio bytes. "
                "Use a Piper fixture variant or another external TTS source instead."
            )
    finally:
        intermediate_path.unlink(missing_ok=True)


def _generate_with_piper(
    *,
    prompt_text: str,
    output_path: Path,
    voice_id: str | None,
    probe: str,
    runner,
) -> None:
    del probe
    if not voice_id:
        raise VoiceFixtureGenerationError("Piper generation requires a voice_id like 'en_US-lessac-low'.")

    piper_bin = _resolve_piper_binary()
    piper_python = _resolve_piper_python()
    voices_dir = _resolve_piper_voices_dir()
    voices_dir.mkdir(parents=True, exist_ok=True)
    model_path = voices_dir / f"{voice_id}.onnx"
    config_path = voices_dir / f"{voice_id}.onnx.json"

    if (not model_path.is_file()) or (not config_path.is_file()):
        runner(
            [
                piper_python,
                "-m",
                "piper.download_voices",
                voice_id,
                "--download-dir",
                str(voices_dir),
            ]
        )

    if not model_path.is_file() or not config_path.is_file():
        raise VoiceFixtureGenerationError(
            f'Piper voice files were not downloaded correctly for "{voice_id}".'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as handle:
        input_path = Path(handle.name)
        handle.write(prompt_text.strip() + "\n")

    try:
        runner(
            [
                piper_bin,
                "-m",
                str(model_path),
                "-c",
                str(config_path),
                "-i",
                str(input_path),
                "-f",
                str(output_path),
            ]
        )
        _verify_nonempty_file(output_path, label="generated wav output")
    finally:
        input_path.unlink(missing_ok=True)


def _mix_background_noise(
    *,
    base_audio_path: Path,
    output_path: Path,
    noise_profile: str,
    noise_level_db: float,
    ffmpeg_binary: str,
    probe: str,
    runner,
) -> None:
    duration_s = _probe_duration(base_audio_path, probe, runner)
    if duration_s <= 0.0:
        raise VoiceFixtureGenerationError(f'Cannot add background noise to empty audio: "{base_audio_path}"')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner(
        [
            ffmpeg_binary,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(base_audio_path),
            "-f",
            "lavfi",
            "-t",
            f"{duration_s:.6f}",
            "-i",
            f"anoisesrc=color={noise_profile}:sample_rate=16000",
            "-filter_complex",
            (
                f"[1:a]volume={noise_level_db}dB[noise];"
                "[0:a][noise]amix=inputs=2:duration=first:normalize=0[aout]"
            ),
            "-map",
            "[aout]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    _verify_nonempty_file(output_path, label="noisy wav output")


def _resolved_output_path(
    manifest_path: str | Path,
    variant: VoiceFixtureVariant,
    output_root: str | Path | None,
) -> Path:
    audio_path = Path(variant.audio_path).expanduser()
    if output_root is None:
        return variant.resolved_audio_path(manifest_path)
    if audio_path.is_absolute():
        return audio_path.resolve()
    cleaned_parts = [part for part in audio_path.parts if part not in {"..", "."}]
    cleaned_relative = Path(*cleaned_parts) if cleaned_parts else Path(audio_path.name)
    return (Path(output_root).expanduser().resolve() / cleaned_relative).resolve()


def _metadata_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(f"{audio_path.suffix}.json")


def _verify_nonempty_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise VoiceFixtureGenerationError(f"{label} was not created: {path}")
    if path.stat().st_size <= 44:
        raise VoiceFixtureGenerationError(f"{label} is unexpectedly small: {path}")


def _require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise VoiceFixtureGenerationError(f'Required binary "{name}" is not available.')
    return resolved


def _resolve_first_binary(names: tuple[str, ...]) -> str:
    for name in names:
        resolved = shutil.which(name)
        if resolved is not None:
            return resolved
    joined = ", ".join(names)
    raise VoiceFixtureGenerationError(f"None of the required binaries are available: {joined}")


def _resolve_piper_binary() -> str:
    env_value = os.environ.get("INTERACTION_PIPER_BIN")
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
    repo_candidate = _project_root() / ".tmp" / "piper-venv" / "bin" / "piper"
    if repo_candidate.is_file():
        return str(repo_candidate)
    resolved = shutil.which("piper")
    if resolved is not None:
        return resolved
    raise VoiceFixtureGenerationError(
        "Piper is not installed. Set INTERACTION_PIPER_BIN or install it into .tmp/piper-venv."
    )


def _resolve_piper_python() -> str:
    env_value = os.environ.get("INTERACTION_PIPER_PYTHON")
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
    repo_candidate = _project_root() / ".tmp" / "piper-venv" / "bin" / "python"
    if repo_candidate.is_file():
        return str(repo_candidate)
    raise VoiceFixtureGenerationError(
        "Piper's Python environment is not installed. Set INTERACTION_PIPER_PYTHON or install Piper into .tmp/piper-venv."
    )


def _resolve_piper_voices_dir() -> Path:
    env_value = os.environ.get("INTERACTION_PIPER_VOICES_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (_project_root() / ".tmp" / "piper-voices").resolve()


def _probe_duration(path: Path, probe_binary: str, runner) -> float:
    completed = runner(
        [
            probe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    output = str(completed.stdout or "").strip()
    try:
        return float(output)
    except ValueError as error:
        raise VoiceFixtureGenerationError(f'Could not parse audio duration for "{path}": {output!r}') from error


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "unknown command failure"
        raise VoiceFixtureGenerationError(f'Command failed: {" ".join(command)} ({detail})')
    return completed
