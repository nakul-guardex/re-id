#!/usr/bin/env python3
"""
Multi-Camera Pipeline Coordinator
=================================
Orchestrates parallel RTSP streams, batched GPU detection (YOLO11-Seg),
independent ByteTrack instances, clear-frame quality filtering, batched
segmented Re-ID feature extraction, Hungarian in-camera matching, and
rich visual frame rendering with alpha masks and contour outlines.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from ..config import (
    COLOR_TENTATIVE,
    COLOR_UNKNOWN,
    IMAGENET_MEAN_BGR,
    MATCH_SIM_THRESHOLD,
    PALETTE,
    RECONCILIATION_INTERVAL_SEC,
    STICKY_HYSTERESIS_BOOST,
    TARGET_DETECTION_FPS,
)
from ..core.event_logger import SessionEventLogger
from ..core.foreground_extractor import extract_segmented_crop
from ..core.identity_manager import GlobalIdentityManager
from ..core.quality_gate import ClearFrameQualityGate
from ..core.session_manager import SessionManager, SessionState
from ..core.stream_worker import StreamWorker
from ..core.tracker_engine import MultiCameraTrackerEngine, TrackedDetection
from ..core.tracklet_builder import TrackletManager
from ..models.reid_bnneck import ResNet50IBN_WithBNNeck

FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_styled_label(img: np.ndarray, box: np.ndarray, label: str, color: Tuple[int, int, int]):
    """Draws a modern solid badge label with contrasting text above the bounding box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h_img, w_img = img.shape[:2]
    c_int = tuple(int(c) for c in color)
    cv2.rectangle(img, (x1, y1), (x2, y2), c_int, 2)

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.44, 1)
    top = max(0, y1 - th - 8)
    right = min(w_img, x1 + tw + 10)
    cv2.rectangle(img, (x1, top), (right, top + th + 8), c_int, -1)
    # White or dark text depending on color luminance
    text_color = (0, 0, 0) if (c_int[0] + c_int[1] + c_int[2]) > 380 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 5, top + th + 3), FONT, 0.44, text_color, 1, cv2.LINE_AA)


class MultiCameraCoordinator:
    """Master coordinator driving inference, tracking, Re-ID, and rendering."""

    def __init__(
        self,
        session_manager: SessionManager,
        tracker_engine: MultiCameraTrackerEngine,
        reid_extractor: ResNet50IBN_WithBNNeck,
        identity_manager: GlobalIdentityManager,
        quality_gate: ClearFrameQualityGate,
        tracklet_manager: TrackletManager,
        event_logger: SessionEventLogger,
    ):
        self.session_mgr = session_manager
        self.tracker_engine = tracker_engine
        self.reid_extractor = reid_extractor
        self.identity_mgr = identity_manager
        self.quality_gate = quality_gate
        self.tracklet_mgr = tracklet_manager
        self.event_logger = event_logger

        # Hook session start/stop callbacks
        self.session_mgr.on_session_start = self._on_session_started
        self.session_mgr.on_session_stop = self._on_session_stopped

        self.running = False
        self.inference_thread: Optional[threading.Thread] = None
        self.reconcile_thread: Optional[threading.Thread] = None

    def _on_session_started(self, session_id: str, active_camera_ids: List[str]):
        """Triggered when user clicks Start Analysis."""
        self.event_logger.start_session(session_id)
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, name="Coord-Inference", daemon=True)
        self.reconcile_thread = threading.Thread(target=self._reconciliation_loop, name="Coord-Reconcile", daemon=True)
        self.inference_thread.start()
        self.reconcile_thread.start()
        print(f"[Coordinator] Multi-camera pipeline active for session: {session_id}", flush=True)

    def _on_session_stopped(self, session_id: str):
        """Triggered when user clicks Stop Analysis."""
        self.running = False
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        if self.reconcile_thread and self.reconcile_thread.is_alive():
            self.reconcile_thread.join(timeout=2.0)
        self.event_logger.close()
        print(f"[Coordinator] Multi-camera pipeline stopped for session: {session_id}", flush=True)

    def _reconciliation_loop(self):
        """Runs the background online reconciliation and auto-merge engine."""
        while self.running:
            try:
                merges = self.identity_mgr.run_reconciliation_cycle()
                if merges:
                    for m in merges:
                        self.event_logger.log("RECONCILE_MERGE", m)
            except Exception as ex:
                print(f"[Coordinator] Error in reconciliation loop: {ex}", flush=True)
            time.sleep(RECONCILIATION_INTERVAL_SEC)

    def _inference_loop(self):
        """Main inference loop driving detection, tracking, Re-ID, and annotation."""
        target_cycle_dt = 1.0 / TARGET_DETECTION_FPS
        last_heartbeat = time.monotonic()
        cycle_count = 0

        while self.running:
            t_cycle_start = time.monotonic()
            try:
                last_heartbeat, cycle_count = self._inference_step(last_heartbeat, cycle_count)
            except Exception as ex:
                import traceback
                print(f"[Coordinator] Error in inference cycle: {ex}", flush=True)
                traceback.print_exc()

            cycle_elapsed = time.monotonic() - t_cycle_start
            sleep_needed = target_cycle_dt - cycle_elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    def _inference_step(self, last_heartbeat: float, cycle_count: int) -> Tuple[float, int]:
        if not hasattr(self, 'last_cap_ts'):
            self.last_cap_ts = {}
            self.last_annotated = {}

        # 1. Collect freshest raw frames from all active camera workers
        active_workers: List[Tuple[str, StreamWorker, np.ndarray, float]] = []
        for cid, worker in self.session_mgr.workers.items():
            if not worker.running:
                continue
            frame, cap_ts, fps, status = worker.get_latest_frame()
            if frame is not None:
                if cap_ts > self.last_cap_ts.get(cid, 0.0):
                    active_workers.append((cid, worker, frame, cap_ts))
                    self.last_cap_ts[cid] = cap_ts
                else:
                    # Pad video: no new frame from camera, duplicate last annotated frame to maintain 5 FPS
                    last_ann = self.last_annotated.get(cid)
                    if last_ann is not None:
                        worker.set_annotated_frame(last_ann)

        if not active_workers:
            time.sleep(0.02)
            return last_heartbeat, cycle_count

        # 2. Batched YOLO11-Seg Detection on all frames in parallel on GPU
        frames_batch = [f for _, _, f, _ in active_workers]
        results_batch = self.tracker_engine.detect_batch(frames_batch)

        # Crops queue for batched Re-ID extraction across all cameras
        pending_reid_queries = []  # list of (cam_id, track_id, crop, box, quality_res, cap_ts)

        # Per-camera tracked detections
        cam_detections: Dict[str, List[TrackedDetection]] = {}

        # 3. Per-Camera Independent Tracking & Quality Gate Evaluation
        for i, (cid, worker, frame, cap_ts) in enumerate(active_workers):
            res = results_batch[i] if i < len(results_batch) else None
            dets = self.tracker_engine.track_camera_frame(cid, frame, res)
            cam_detections[cid] = dets

            all_boxes = [d.box for d in dets]

            for det in dets:
                tracklet = self.tracklet_mgr.get_or_create(cid, det.track_id)
                if tracklet.status == "CONFIRMED":
                    continue  # Fast-path: already confirmed, skip expensive Re-ID extraction

                # Run Quality Gate
                q_res = self.quality_gate.evaluate(
                    cam_id=cid,
                    track_id=det.track_id,
                    box=det.box,
                    mask_bool=det.mask_bool,
                    confidence=det.conf,
                    frame_shape=frame.shape[:2],
                    frame_bgr=frame,
                    all_other_boxes=[b for b in all_boxes if not np.array_equal(b, det.box)],
                    capture_ts=cap_ts,
                )

                if q_res.passed:
                    # Extract Segmented Foreground Crop with ImageNet neutral mean fill
                    seg_crop = extract_segmented_crop(frame, det.box, det.mask_bool, bg_fill=IMAGENET_MEAN_BGR)
                    if seg_crop.size > 0:
                        pending_reid_queries.append((cid, det.track_id, seg_crop, det.box, q_res, cap_ts))

        # 4. Batched Appearance Feature Extraction on GPU (ResNet50-IBN BNNeck)
        if pending_reid_queries:
            crops_to_embed = [item[2] for item in pending_reid_queries]
            embeddings = self.reid_extractor.extract_batch(crops_to_embed)

            # Feed into Tracklets
            for j, (cid, tid, crop, box, q_res, cap_ts) in enumerate(pending_reid_queries):
                emb = embeddings[j]
                if emb is not None:
                    tracklet = self.tracklet_mgr.get_or_create(cid, tid)
                    tracklet.add_sample(
                        embedding=emb,
                        capture_ts=cap_ts,
                        box=box,
                        crop_bgr=crop,
                        quality_score=q_res.confidence,
                    )

        # 5. In-Camera 1-to-1 Hungarian Matching for tracks with >= 3 clear samples
        for cid, worker, frame, cap_ts in active_workers:
            dets = cam_detections.get(cid, [])
            tracks_for_matching = []
            frame_assigned_gids = []

            for det in dets:
                tracklet = self.tracklet_mgr.get_or_create(cid, det.track_id)
                if tracklet.status == "CONFIRMED":
                    if tracklet.assigned_gid:
                        gid = self.identity_mgr.resolve_gid(tracklet.assigned_gid)
                        frame_assigned_gids.append(gid)
                        if gid in self.identity_mgr.identities:
                            self.identity_mgr.identities[gid].active_presence[cid] = (det.track_id, cap_ts)
                    continue

                # Requires at least 3 clear samples
                if tracklet.sample_count >= 3:
                    proto = tracklet.get_prototype_embedding()
                    if proto is not None:
                        tracks_for_matching.append((det.track_id, proto, det.box, tracklet.assigned_gid))

            if tracks_for_matching:
                assignments = self.identity_mgr.match_camera_tracks(cid, tracks_for_matching, cap_ts)
                for tid, (gid, score) in assignments.items():
                    tracklet = self.tracklet_mgr.get_or_create(cid, tid)
                    resolved_gid = self.identity_mgr.resolve_gid(gid)
                    tracklet.assigned_gid = resolved_gid
                    tracklet.similarity_score = score
                    tracklet.status = "CONFIRMED"
                    frame_assigned_gids.append(resolved_gid)

            # Record co-occurrences of all assigned GIDs in this frame (both old and new)
            if len(frame_assigned_gids) >= 2:
                self.identity_mgr.record_co_occurrence(cid, frame_assigned_gids)

            # 6. Render Visual Frame: Alpha-blended masks, contour outlines & styled labels
            annotated = frame.copy()
            mask_overlay = annotated.copy()

            for det in dets:
                tracklet = self.tracklet_mgr.get_or_create(cid, det.track_id)
                gid = self.identity_mgr.resolve_gid(tracklet.assigned_gid) if tracklet.assigned_gid else None

                if gid and gid in self.identity_mgr.identities:
                    ident = self.identity_mgr.identities[gid]
                    color = ident.color
                elif tracklet.sample_count < 3:
                    color = COLOR_TENTATIVE
                else:
                    color = COLOR_UNKNOWN

                # Fill mask overlay
                mask_overlay[det.mask_bool] = (
                    mask_overlay[det.mask_bool] * 0.65 + np.array(color) * 0.35
                ).astype(np.uint8)

                # Draw crisp contour boundary
                cnts, _ = cv2.findContours(det.mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask_overlay, cnts, -1, tuple(int(c) for c in color), 1, cv2.LINE_AA)

            # Blend masks
            cv2.addWeighted(mask_overlay, 1.0, annotated, 0.0, 0, annotated)

            # Draw top badges and tags
            for det in dets:
                tracklet = self.tracklet_mgr.get_or_create(cid, det.track_id)
                gid = self.identity_mgr.resolve_gid(tracklet.assigned_gid) if tracklet.assigned_gid else None

                if gid and gid in self.identity_mgr.identities:
                    ident = self.identity_mgr.identities[gid]
                    label = f"[{ident.name}] Sim: {tracklet.similarity_score:.2f} | #{det.track_id}"
                    color = ident.color
                elif tracklet.sample_count < 3:
                    label = f"Buffering ({tracklet.sample_count}/3) | #{det.track_id}"
                    color = COLOR_TENTATIVE
                else:
                    label = f"UNKNOWN | #{det.track_id}"
                    color = COLOR_UNKNOWN

                draw_styled_label(annotated, det.box, label, color)

            # Top HUD Banner
            w_frame = annotated.shape[1]
            cv2.rectangle(annotated, (0, 0), (w_frame, 32), (18, 22, 26), -1)
            hud_text = f"{worker.name} | {worker.fps:.1f} FPS | People: {len(dets)}"
            cv2.putText(annotated, hud_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 140), 1, cv2.LINE_AA)

            # Send annotated frame to worker's MJPEG web feed and save it for padding
            worker.set_annotated_frame(annotated)
            self.last_annotated[cid] = annotated

        # Periodic terminal progress logging (every 2.5s)
        cycle_count += 1
        now_mono = time.monotonic()
        if (now_mono - last_heartbeat) >= 2.5:
            stats = [
                f"{cid}: {w.fps:.1f}fps ({len(cam_detections.get(cid, []))} people)"
                for cid, w, _, _ in active_workers
            ]
            total_enrolled = len(self.identity_mgr.identities)
            print(
                f"[Analysis] Cycle #{cycle_count:05d} | {' | '.join(stats)} | Total Enrolled: {total_enrolled} IDs",
                flush=True,
            )
            last_heartbeat = now_mono

        return last_heartbeat, cycle_count
