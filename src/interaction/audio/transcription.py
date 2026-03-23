"""Transcript stream helpers for the Phase 2 voice loop."""

from __future__ import annotations

from interaction.contracts import TranscriptSegment


class ScriptedTranscriber:
    """Generate partial and final transcript segments from typed text."""

    def stream(self, utterance: str, *, final_confidence: float | None = 1.0) -> list[TranscriptSegment]:
        text = utterance.strip()
        if not text:
            return []
        words = text.split()
        segments: list[TranscriptSegment] = []
        for index in range(1, len(words) + 1):
            partial_text = " ".join(words[:index])
            segments.append(
                TranscriptSegment(
                    text=partial_text,
                    is_final=index == len(words),
                    confidence=final_confidence if index == len(words) else None,
                )
            )
        return segments
