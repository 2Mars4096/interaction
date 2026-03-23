"""Audio and turn-handling primitives for the voice loop."""

from .macos_speech import LiveSpeechCapture, MacOSLiveSpeechProvider, SpeechCaptureError
from .transcription import ScriptedTranscriber
from .turns import PushToTalkTurnManager, TurnPhase

__all__ = [
    "LiveSpeechCapture",
    "MacOSLiveSpeechProvider",
    "PushToTalkTurnManager",
    "ScriptedTranscriber",
    "SpeechCaptureError",
    "TurnPhase",
]
