#!/usr/bin/env python3
"""
Multi-Camera Tracker Engine (YOLO11-Seg + Per-Camera Tracker Instances)
========================================================================
Maintains a single shared YOLO11-Seg model for batched GPU forward passes,
while maintaining STRICTLY INDEPENDENT tracker instances per camera stream.
This guarantees that track IDs and Kalman filter states never cross-contaminate
between cameras.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from ..config import TRACKER_TYPE


@dataclass
class TrackedDetection:
    track_id: int
    box: np.ndarray          # [x1, y1, x2, y2]
    mask_bool: np.ndarray    # Binary mask boolean array (H_frame, W_frame)
    conf: float
    cam_id: str


def get_default_tracker_config() -> SimpleNamespace:
    """Loads default ByteTrack configuration from ultralytics."""
    import ultralytics
    yaml_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": True,
        }
    return SimpleNamespace(**data)


class CameraTracker:
    """Individual tracker instance dedicated to a single camera stream."""

    def __init__(self, cam_id: str, tracker_type: str = TRACKER_TYPE):
        self.cam_id = cam_id
        if tracker_type != "bytetrack":
            print(
                f"[Tracker] TRACKER_TYPE={tracker_type!r} is not implemented; using bytetrack.",
                flush=True,
            )
            tracker_type = "bytetrack"
        self.tracker_type = tracker_type
        self.cfg = get_default_tracker_config()
        self.byte_tracker = BYTETracker(self.cfg)

    def reset(self):
        """Resets tracker state."""
        self.byte_tracker = BYTETracker(self.cfg)

    def update(
        self,
        yolo_result,
        frame_shape: Tuple[int, int],
    ) -> List[TrackedDetection]:
        """
        Updates camera tracker with YOLO detections and matches masks.
        """
        if yolo_result is None or yolo_result.boxes is None:
            return []

        h_frame, w_frame = frame_shape[:2]
        boxes_cpu = yolo_result.boxes.cpu().numpy()
        try:
            # Empty frames still advance ByteTrack so lost IDs age out.
            tracked_out = self.byte_tracker.update(boxes_cpu)
        except Exception as e:
            print(f"[Tracker] Error running BYTETracker.update for {self.cam_id}: {e}", flush=True)
            return []

        if tracked_out is None or len(tracked_out) == 0:
            return []

        # YOLO raw detections
        raw_boxes = boxes_cpu.xyxy
        raw_confs = boxes_cpu.conf
        raw_masks = yolo_result.masks.data.cpu().numpy() if yolo_result.masks is not None else None

        detections = []
        for t in tracked_out:
            # t: [x1, y1, x2, y2, track_id, score, cls, raw_det_idx]
            x1, y1, x2, y2 = t[:4]
            tid = int(t[4])
            score = float(t[5])
            raw_idx = int(t[7]) if len(t) > 7 else -1

            # Match mask
            if raw_masks is not None and 0 <= raw_idx < len(raw_masks):
                m_single = cv2.resize(raw_masks[raw_idx], (w_frame, h_frame), interpolation=cv2.INTER_LINEAR) > 0.5
            else:
                # Fallback if mask missing
                m_single = np.zeros((h_frame, w_frame), dtype=bool)
                bx1, by1 = max(0, int(round(x1))), max(0, int(round(y1)))
                bx2, by2 = min(w_frame, int(round(x2))), min(h_frame, int(round(y2)))
                m_single[by1:by2, bx1:bx2] = True

            detections.append(
                TrackedDetection(
                    track_id=tid,
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    mask_bool=m_single,
                    conf=score,
                    cam_id=self.cam_id,
                )
            )

        return detections


class MultiCameraTrackerEngine:
    """
    Central inference engine managing the YOLO11-Seg detector and
    per-camera tracking instances.
    """

    def __init__(self, model_path: str = "yolo11s-seg.pt", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"[TrackerEngine] Loading YOLO11-Seg from {model_path} onto {self.device}...")
        self.model = YOLO(model_path)
        self.trackers: Dict[str, CameraTracker] = {}

    def get_or_create_tracker(self, cam_id: str) -> CameraTracker:
        if cam_id not in self.trackers:
            self.trackers[cam_id] = CameraTracker(cam_id=cam_id)
        return self.trackers[cam_id]

    def reset_camera_tracker(self, cam_id: str):
        if cam_id in self.trackers:
            self.trackers[cam_id].reset()

    def reset_all(self):
        for tracker in self.trackers.values():
            tracker.reset()
        self.trackers.clear()

    def detect_batch(self, frames: List[np.ndarray], conf: float = 0.35) -> List[Any]:
        """
        Runs batched GPU detection and instance segmentation across multiple camera frames.
        """
        if not frames:
            return []
        results = self.model.predict(
            source=frames,
            classes=[0],  # Person class only
            conf=conf,
            verbose=False,
            device=self.device,
        )
        return results

    def track_camera_frame(
        self,
        cam_id: str,
        frame: np.ndarray,
        yolo_result: Any,
    ) -> List[TrackedDetection]:
        """
        Updates the camera's dedicated tracker instance.
        """
        tracker = self.get_or_create_tracker(cam_id)
        return tracker.update(yolo_result, frame.shape[:2])
