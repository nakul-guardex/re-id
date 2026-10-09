#!/usr/bin/env python3
"""
Tracklet Builder & Quality Accumulator
======================================
Accumulates gate-passing, temporally-spaced appearance embeddings for each track.
Performs intra-tracklet outlier rejection to filter out boundary bleed,
computes representative prototype vectors, and caches high-clarity avatar thumbnails.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class TrackletSample:
    embedding: np.ndarray
    capture_ts: float
    box: np.ndarray
    crop_bgr: np.ndarray
    quality_score: float


class Tracklet:
    """Represents a continuous track trajectory in a single camera."""

    def __init__(self, cam_id: str, track_id: int, max_buffer: int = 10):
        self.cam_id = cam_id
        self.track_id = track_id
        self.max_buffer = max_buffer
        self.samples: Deque[TrackletSample] = deque(maxlen=max_buffer)

        # Global identity state
        self.assigned_gid: Optional[str] = None
        self.candidate_gid: Optional[str] = None
        self.similarity_score: float = 0.0
        self.status: str = "UNKNOWN"  # "UNKNOWN", "TENTATIVE", "CONFIRMED"
        self.last_seen_ts: float = 0.0
        self.last_reid_ts: float = 0.0
        self.consecutive_passes: int = 0
        self.consecutive_weak: int = 0
        self.best_avatar: Optional[np.ndarray] = None
        self.best_avatar_quality: float = 0.0

    def mark_seen(self, capture_ts: float):
        self.last_seen_ts = capture_ts

    def reset_identity(self):
        """Forget GID assignment but keep the local tracker ID."""
        self.samples.clear()
        self.assigned_gid = None
        self.candidate_gid = None
        self.similarity_score = 0.0
        self.status = "UNKNOWN"
        self.consecutive_passes = 0
        self.consecutive_weak = 0
        self.best_avatar = None
        self.best_avatar_quality = 0.0
        self.last_reid_ts = 0.0

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def add_sample(
        self,
        embedding: np.ndarray,
        capture_ts: float,
        box: np.ndarray,
        crop_bgr: np.ndarray,
        quality_score: float,
        outlier_rejection_cos: float = 0.30,
    ) -> bool:
        """
        Adds a sample with intra-track outlier rejection.
        Returns True if accepted, False if rejected as outlier.
        """
        self.last_seen_ts = capture_ts

        # Intra-track outlier rejection if we already have samples
        if len(self.samples) >= 3:
            median_emb = np.median(np.stack([s.embedding for s in self.samples]), axis=0)
            median_emb = l2norm(median_emb)
            sim_to_median = float(np.dot(embedding, median_emb))
            if sim_to_median < 0.45:
                # ID Switch detected! The tracker reused an ID for a completely different person.
                self.reset_identity()

        sample = TrackletSample(
            embedding=embedding,
            capture_ts=capture_ts,
            box=box,
            crop_bgr=crop_bgr,
            quality_score=quality_score,
        )
        self.samples.append(sample)

        # Update best avatar thumbnail for UI card
        if quality_score > self.best_avatar_quality and crop_bgr is not None and crop_bgr.size > 0:
            self.best_avatar = crop_bgr.copy()
            self.best_avatar_quality = quality_score

        return True

    def note_match(self, gid: str, score: float, threshold: float, lock_frames: int) -> bool:
        """
        Accumulate consecutive hits against the same GID.
        Returns True only on the frame the track locks CONFIRMED.
        """
        if score < threshold:
            self.consecutive_passes = 0
            self.candidate_gid = None
            self.status = "UNKNOWN"
            return False
        if self.candidate_gid == gid:
            self.consecutive_passes += 1
        else:
            self.candidate_gid = gid
            self.consecutive_passes = 1
        self.similarity_score = score
        self.status = "TENTATIVE"
        if self.consecutive_passes >= lock_frames:
            self.assigned_gid = gid
            self.status = "CONFIRMED"
            return True
        return False

    def observe_confirmed_similarity(
        self,
        sim: float,
        ema_alpha: float,
        threshold: float,
        release_n: int,
    ) -> bool:
        """Smooth the live cosine. Returns True when the smoothed score has stayed weak long enough to release."""
        prev = self.similarity_score
        if prev <= 0.0:
            self.similarity_score = float(sim)
        else:
            self.similarity_score = (1.0 - ema_alpha) * prev + ema_alpha * float(sim)
        if self.similarity_score < threshold:
            self.consecutive_weak += 1
        else:
            self.consecutive_weak = 0
        return self.consecutive_weak >= release_n

    def get_prototype_embedding(self) -> Optional[np.ndarray]:
        """Returns the mean L2-normalized embedding across accumulated samples."""
        if not self.samples:
            return None
        stacked = np.stack([s.embedding for s in self.samples])
        return l2norm(np.mean(stacked, axis=0))

    def get_latest_box(self) -> Optional[np.ndarray]:
        if self.samples:
            return self.samples[-1].box
        return None


class TrackletManager:
    """Manages all active tracklets across all 6 camera streams."""

    def __init__(self, max_buffer_len: int = 10, lost_timeout_sec: float = 5.0):
        self.max_buffer_len = max_buffer_len
        self.lost_timeout_sec = lost_timeout_sec
        # (cam_id, track_id) -> Tracklet
        self.tracklets: Dict[Tuple[str, int], Tracklet] = {}

    def get_or_create(self, cam_id: str, track_id: int) -> Tracklet:
        key = (cam_id, track_id)
        if key not in self.tracklets:
            self.tracklets[key] = Tracklet(cam_id, track_id, max_buffer=self.max_buffer_len)
        return self.tracklets[key]

    def prune_dead_tracks(self, current_ts: float) -> List[Tuple[str, int]]:
        """Removes tracklets that haven't been updated within lost_timeout_sec."""
        dead_keys = [
            k for k, trk in self.tracklets.items()
            if trk.last_seen_ts > 0.0 and (current_ts - trk.last_seen_ts) > self.lost_timeout_sec
        ]
        for k in dead_keys:
            self.tracklets.pop(k, None)
        return dead_keys

    def reset(self):
        self.tracklets.clear()

    def get_active_tracks_for_camera(self, cam_id: str) -> List[Tracklet]:
        return [trk for (cid, _), trk in self.tracklets.items() if cid == cam_id]
