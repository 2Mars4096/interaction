"""Audio and turn-handling primitives for the voice loop."""

from .file_speech import MacOSFileSpeechProvider
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
    "MacOSFileSpeechProvider",
    "MacOSLiveSpeechProvider",
    "PushToTalkTurnManager",
    "ScriptedTranscriber",
    "SpeechCaptureError",
    "SpeechProvider",
    "TurnPhase",
]
