"""Audio and turn-handling primitives for the voice loop."""

from .macos_speech import MacOSLiveSpeechProvider
from .provider_types import (
    LiveSpeechCapture,
    LiveSpeechClause,
    LiveSpeechStreamCapture,
    SpeechCaptureError,
    SpeechProvider,
)
from .transcription import ScriptedTranscriber
from .turns import PushToTalkTurnManager, TurnPhase

__all__ = [
    "LiveSpeechCapture",
    "LiveSpeechClause",
    "LiveSpeechStreamCapture",
    "MacOSLiveSpeechProvider",
    "PushToTalkTurnManager",
    "ScriptedTranscriber",
    "SpeechCaptureError",
    "SpeechProvider",
    "TurnPhase",
]
