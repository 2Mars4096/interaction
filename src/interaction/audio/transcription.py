"""Transcript stream helpers for the Phase 2 voice loop."""

from __future__ import annotations

import re

from interaction.contracts import TranscriptSegment


class ScriptedTranscriber:
    """Generate partial and final transcript segments from typed text."""

    def stream(self, utterance: str, *, final_confidence: float | None = 1.0) -> list[TranscriptSegment]:
        if not utterance.strip():
            return []
        words = list(re.finditer(r"\S+", utterance))
        segments: list[TranscriptSegment] = []
        for index, word in enumerate(words, start=1):
            is_final = index == len(words)
            partial_text = utterance if is_final else utterance[:word.end()]
            segments.append(
                TranscriptSegment(
                    text=partial_text,
                    is_final=is_final,
                    confidence=final_confidence if is_final else None,
                )
            )
        return segments
