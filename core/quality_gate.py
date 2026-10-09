#!/usr/bin/env python3
"""
Clear-Frame Quality Gate
========================
Filters out noisy detections before appearance embeddings can enter
a tracklet or affect identity decisions:
  1. Minimum crop dimensions (height >= 90px, width >= 45px)
  2. Border margin (>= 15px from image frame boundaries)
  3. Instance segmentation mask solidity ratio in [0.20, 0.85]
  4. Detection confidence >= 0.45
  5. Laplacian variance blur check
  6. Multi-person occlusion check (IoU overlap < 0.35)
  7. Temporal spacing (crops from same track must be separated by >= 0.40s)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class GateEvaluationResult:
    passed: bool
    rejection_reasons: List[str]
    height: int
    width: int
    solidity: float
    confidence: float
    blur_score: float
    max_overlap_iou: float
    time_delta: float


def compute_box_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = float(areaA + areaB - inter)
    return (inter / union) if union > 0 else 0.0


class ClearFrameQualityGate:
    """
    Evaluates individual person detections against strict quality standards.
    Maintains per-track temporal spacing state to guarantee that consecutive
    frames aren't near-identical duplicates.
    """

    def __init__(
        self,
        min_height: int = 90,
        min_width: int = 45,
        border_margin: int = 15,
        solidity_range: Tuple[float, float] = (0.20, 0.85),
        min_confidence: float = 0.45,
        min_blur_score: float = 35.0,
        max_occlusion_iou: float = 0.35,
        temporal_spacing_sec: float = 0.40,
        min_gate_passes: int = 3,
    ):
        self.min_height = min_height
        self.min_width = min_width
        self.border_margin = border_margin
        self.solidity_min, self.solidity_max = solidity_range
        self.min_confidence = min_confidence
        self.min_blur_score = min_blur_score
        self.max_occlusion_iou = max_occlusion_iou
        self.temporal_spacing_sec = temporal_spacing_sec
        self.min_gate_passes = min_gate_passes

        # Track temporal history: (cam_id, track_id) -> last_accepted_capture_ts
        self._last_accepted_ts: Dict[Tuple[str, int], float] = {}

    def reset_track(self, cam_id: str, track_id: int):
        """Clears state when a track terminates."""
        self._last_accepted_ts.pop((cam_id, track_id), None)

    def evaluate(
        self,
        cam_id: str,
        track_id: int,
        box: np.ndarray,
        mask_bool: np.ndarray,
        confidence: float,
        frame_shape: Tuple[int, int],
        frame_bgr: np.ndarray,
        all_other_boxes: List[np.ndarray],
        capture_ts: float,
    ) -> GateEvaluationResult:
        """
        Runs the full quality gate battery on a detected person instance.
        """
        reasons = []
        h_frame, w_frame = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        # 1. Dimension Check
        if h < self.min_height:
            reasons.append(f"Height too small ({h}px < {self.min_height}px)")
        if w < self.min_width:
            reasons.append(f"Width too small ({w}px < {self.min_width}px)")

        # 2. Border Proximity Margin (reject cut-off bodies entering/exiting frame)
        if (
            x1 < self.border_margin
            or y1 < self.border_margin
            or x2 > (w_frame - self.border_margin)
            or y2 > (h_frame - self.border_margin)
        ):
            reasons.append("Cut off at image boundary")

        # 3. Mask Solidity (foreground mask area / bounding box area)
        box_area = float(w * h)
        if box_area > 0 and mask_bool is not None:
            # Crop mask to bounding box
            my1, my2 = max(0, y1), min(h_frame, y2)
            mx1, mx2 = max(0, x1), min(w_frame, x2)
            crop_mask = mask_bool[my1:my2, mx1:mx2]
            mask_pixels = float(np.count_nonzero(crop_mask))
            solidity = mask_pixels / box_area
            if solidity < self.solidity_min:
                reasons.append(f"Solidity too sparse ({solidity:.2f} < {self.solidity_min})")
            elif solidity > self.solidity_max:
                reasons.append(f"Solidity too dense / box fill ({solidity:.2f} > {self.solidity_max})")
        else:
            solidity = 0.0
            reasons.append("Invalid mask or zero box area")

        # 4. Confidence Check
        if confidence < self.min_confidence:
            reasons.append(f"Low detection confidence ({confidence:.2f} < {self.min_confidence})")

        # 5. Multi-person Occlusion Check
        max_iou = 0.0
        for ob in all_other_boxes:
            iou = compute_box_iou(box, ob)
            if iou > max_iou:
                max_iou = iou
        if max_iou > self.max_occlusion_iou:
            reasons.append(f"Heavy occlusion with another person (IoU {max_iou:.2f} > {self.max_occlusion_iou})")

        # 6. Laplacian Blur Check
        blur_score = 0.0
        if w >= 20 and h >= 20 and frame_bgr is not None:
            crop_raw = frame_bgr[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
            if crop_raw.size > 0:
                gray = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if blur_score < self.min_blur_score:
                    reasons.append(f"Motion blur detected (Var {blur_score:.1f} < {self.min_blur_score})")

        # 7. Temporal Spacing Check
        last_ts = self._last_accepted_ts.get((cam_id, track_id), 0.0)
        time_delta = capture_ts - last_ts
        if time_delta < self.temporal_spacing_sec and last_ts > 0.0:
            reasons.append(f"Temporal spacing too small ({time_delta:.2f}s < {self.temporal_spacing_sec}s)")

        passed = len(reasons) == 0

        # If passed all checks, update accepted timestamp
        if passed:
            self._last_accepted_ts[(cam_id, track_id)] = capture_ts

        return GateEvaluationResult(
            passed=passed,
            rejection_reasons=reasons,
            height=h,
            width=w,
            solidity=round(solidity, 3),
            confidence=round(confidence, 3),
            blur_score=round(blur_score, 1),
            max_overlap_iou=round(max_iou, 3),
            time_delta=round(time_delta, 3),
        )
