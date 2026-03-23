"""Vision and gaze-tracking primitives."""

from .calibration import CalibrationProfile, CalibrationSample, NormalizedPoint
from .gaze import DwellTrigger, GazeSample, GazeSmoother, GazeTargetInferencer, NormalizedScreenTarget
from .mediapipe_provider import MediaPipeWebcamGazeProvider
from .opencv_provider import OpenCVWebcamGazeProvider, WebcamGazeReading, WebcamProvider, WebcamProviderError, WebcamSampleAggregate

__all__ = [
    "CalibrationProfile",
    "CalibrationSample",
    "DwellTrigger",
    "MediaPipeWebcamGazeProvider",
    "GazeSample",
    "GazeSmoother",
    "GazeTargetInferencer",
    "NormalizedPoint",
    "NormalizedScreenTarget",
    "OpenCVWebcamGazeProvider",
    "WebcamProvider",
    "WebcamGazeReading",
    "WebcamProviderError",
    "WebcamSampleAggregate",
]
