"""Small OpenCV-powered webcam debug preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class WebcamPreviewOverlay:
    title: str = "Interaction Webcam Preview"
    max_width: int = 420


class LiveWebcamDebugPreview:
    """Render a compact camera debug window with tracking overlays."""

    def __init__(self, *, title: str = "Interaction Webcam Preview", max_width: int = 420) -> None:
        self.title = title
        self.max_width = max(240, int(max_width))
        self._opened = False

    def open(self) -> None:
        if self._opened:
            return
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.max_width, int(self.max_width * 0.75))
        self._opened = True

    def show_reading(
        self,
        reading: Any,
        *,
        calibrated_point: tuple[float, float] | None = None,
        cursor_point: tuple[float, float] | None = None,
        target_label: str | None = None,
    ) -> None:
        frame = getattr(reading, "frame", None)
        if frame is None:
            return
        if not self._opened:
            self.open()

        canvas = frame.copy()
        height, width = canvas.shape[:2]

        face_bbox = getattr(reading, "face_bbox", None)
        if isinstance(face_bbox, tuple) and len(face_bbox) == 4:
            fx, fy, fw, fh = face_bbox
            cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (48, 214, 124), 2)

            eye_boxes = getattr(reading, "eye_boxes", ())
            for ex, ey, ew, eh in eye_boxes:
                cv2.rectangle(canvas, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (72, 173, 255), 2)

        pupil_points = getattr(reading, "pupil_points", ())
        for px, py in pupil_points:
            cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)

        sample = getattr(reading, "sample", None)
        confidence = getattr(sample, "confidence", None)
        point = getattr(sample, "point", None)
        if point is not None:
            raw_x = int(max(0, min(width - 1, round(point.x * max(1, width - 1)))))
            raw_y = int(max(0, min(height - 1, round(point.y * max(1, height - 1)))))
            cv2.circle(canvas, (raw_x, raw_y), 7, (0, 90, 255), 2)
            cv2.putText(
                canvas,
                "raw gaze",
                (raw_x + 10, max(18, raw_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 90, 255),
                1,
                cv2.LINE_AA,
            )

        if calibrated_point is not None:
            cal_x = int(max(0, min(width - 1, round(calibrated_point[0] * max(1, width - 1)))))
            cal_y = int(max(0, min(height - 1, round(calibrated_point[1] * max(1, height - 1)))))
            cv2.circle(canvas, (cal_x, cal_y), 9, (255, 255, 255), 1)
            cv2.circle(canvas, (cal_x, cal_y), 4, (255, 255, 255), -1)

        lines = [
            f"confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else "confidence: -",
            f"target: {target_label or '-'}",
        ]
        if cursor_point is not None:
            lines.append(f"cursor: {cursor_point[0]:.2f}, {cursor_point[1]:.2f}")
        elif calibrated_point is not None:
            lines.append(f"calibrated: {calibrated_point[0]:.2f}, {calibrated_point[1]:.2f}")

        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (12, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (245, 247, 250),
                1,
                cv2.LINE_AA,
            )

        preview = _resize_for_preview(canvas, max_width=self.max_width)
        cv2.imshow(self.title, preview)
        cv2.waitKey(1)

    def close(self) -> None:
        if not self._opened:
            return
        cv2.destroyWindow(self.title)
        self._opened = False


def _resize_for_preview(frame: np.ndarray, *, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
