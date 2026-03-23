"""Vision and gaze-tracking primitives."""

from .calibration import CalibrationProfile, CalibrationSample, NormalizedPoint
from .gaze import DwellTrigger, GazeSample, GazeSmoother, GazeTargetInferencer, NormalizedScreenTarget
from .opencv_provider import OpenCVWebcamGazeProvider, WebcamGazeReading, WebcamProviderError, WebcamSampleAggregate

__all__ = [
    "CalibrationProfile",
    "CalibrationSample",
    "DwellTrigger",
    "GazeSample",
    "GazeSmoother",
    "GazeTargetInferencer",
    "NormalizedPoint",
    "NormalizedScreenTarget",
    "OpenCVWebcamGazeProvider",
    "WebcamGazeReading",
    "WebcamProviderError",
    "WebcamSampleAggregate",
]
