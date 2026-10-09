#!/usr/bin/env python3
"""
Multi-Camera Pipeline Coordinator
=================================
Orchestrates parallel RTSP streams, batched GPU detection (YOLO11-Seg),
independent ByteTrack instances, clear-frame quality filtering, batched
segmented Re-ID feature extraction, Hungarian in-camera matching, and
rich visual frame rendering with alpha masks and contour outlines.
"""

import subprocess
import threading
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from ..config import (
    COLOR_TENTATIVE,
    COLOR_UNKNOWN,
    CONSECUTIVE_LOCK_FRAMES,
    ENROLLMENT_FRAMES,
    IMAGENET_MEAN_BGR,
    MATCH_SIM_THRESHOLD,
    RECORDINGS_DIR,
    REID_CHECK_INTERVAL_SEC,
    TARGET_DETECTION_FPS,
    is_enrollment_camera,
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


class FfmpegMp4Writer:
    """Writes BGR frames with a standalone ffmpeg process (libx264)."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(fps),
                "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def isOpened(self) -> bool:
        return self.proc.poll() is None and self.proc.stdin is not None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            pass

    def release(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


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
        self.video_writers: Dict[str, cv2.VideoWriter] = {}
        self.last_cap_ts: Dict[str, float] = {}
        self.last_annotated: Dict[str, np.ndarray] = {}
        self.last_seen_ts: Dict[str, float] = {}

    def _reset_pipeline_state(self):
        """Drop identities, tracklets, and tracker state so a new session starts clean."""
        self.identity_mgr.reset()
        self.tracklet_mgr.reset()
        self.quality_gate.reset()
        if hasattr(self.tracker_engine, "reset_all"):
            self.tracker_engine.reset_all()
        self.last_cap_ts.clear()
        self.last_annotated.clear()
        self.last_seen_ts.clear()

    def _on_session_started(self, session_id: str, active_camera_ids: List[str]):
        """Triggered when user clicks Start Analysis."""
        self._reset_pipeline_state()
        self.event_logger.start_session(session_id)

        recordings_dir = getattr(self.session_mgr, "recordings_dir", RECORDINGS_DIR)
        run_folder = Path(recordings_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_folder.mkdir(parents=True, exist_ok=True)
        self.video_writers = {}
        for cid in active_camera_ids:
            # Note: We assume 1080p, we can dynamically get shape in inference_step if needed, 
            # but usually it's best to set when first frame arrives. We'll init as None here.
            self.video_writers[cid] = None
        self.run_folder = str(run_folder)

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, name="Coord-Inference", daemon=True)
        self.inference_thread.start()
        print(f"[Coordinator] Multi-camera pipeline active for session: {session_id}. Saving to {self.run_folder}", flush=True)

    def _on_session_stopped(self, session_id: str):
        """Triggered when user clicks Stop Analysis."""
        self.running = False
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        self.event_logger.close()

        # Close video writers
        for cid, writer in self.video_writers.items():
            if writer is not None:
                writer.release()
        self.video_writers.clear()

        print(f"[Coordinator] Multi-camera pipeline stopped for session: {session_id}", flush=True)

        # The Colab notebook copies the run folder itself. Skip rclone there.
        if os.environ.get("OMNIREID_SKIP_RCLONE") == "1":
            return

        # Automatically upload to Google Drive using rclone in the background
        if hasattr(self, 'run_folder') and self.run_folder:
            # The remote is assumed to be named 'drive'. It uploads to a folder named after the session.
            remote_path = f"drive:OmniReID_Recordings/{os.path.basename(self.run_folder)}"
            print(f"[Coordinator] Starting background rclone sync to Google Drive: {remote_path}", flush=True)
            try:
                subprocess.Popen(
                    ["rclone", "copy", self.run_folder, remote_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except FileNotFoundError:
                print("[Coordinator] Warning: 'rclone' command not found. Skipping Google Drive upload.")

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
        DROP_TIMEOUT_SEC = 5.0

        # 1. Collect only new frames. Stutters keep the last annotated JPEG;
        #    they are not fed back into YOLO or ByteTrack.
        active_workers: List[Tuple[str, StreamWorker, np.ndarray, float]] = []
        now = time.monotonic()

        for cid, worker in self.session_mgr.workers.items():
            if not worker.running:
                continue

            frame, cap_ts, fps, status = worker.get_latest_frame()

            if frame is not None and cap_ts > self.last_cap_ts.get(cid, 0.0):
                active_workers.append((cid, worker, frame, cap_ts))
                self.last_cap_ts[cid] = cap_ts
                self.last_seen_ts[cid] = now
                worker.release_file_frame()
            else:
                last_seen = self.last_seen_ts.get(cid, now)
                if (now - last_seen) > DROP_TIMEOUT_SEC:
                    continue
                last_ann = self.last_annotated.get(cid)
                if last_ann is not None:
                    worker.set_annotated_frame(last_ann)

        if not active_workers:
            file_workers = [w for w in self.session_mgr.workers.values() if w.file_mode]
            if file_workers and all(w.finished for w in file_workers):
                print("[Coordinator] Recorded videos finished.", flush=True)
                self.running = False
            else:
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
                tracklet.mark_seen(cap_ts)
                enrollment = is_enrollment_camera(cid)

                # Confirmed tracks stay labeled, but are re-embedded on an interval
                # so a recycled ByteTrack ID cannot keep the previous person forever.
                if tracklet.status == "CONFIRMED":
                    if (cap_ts - tracklet.last_reid_ts) < REID_CHECK_INTERVAL_SEC:
                        continue
                elif (not enrollment) and tracklet.status == "UNKNOWN" and tracklet.last_reid_ts > 0.0:
                    if (cap_ts - tracklet.last_reid_ts) < REID_CHECK_INTERVAL_SEC:
                        continue

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

                if not q_res.passed:
                    continue
                tracklet.last_reid_ts = cap_ts
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
                if emb is None:
                    continue
                tracklet = self.tracklet_mgr.get_or_create(cid, tid)
                was_confirmed = tracklet.status == "CONFIRMED" and bool(tracklet.assigned_gid)
                held_gid = tracklet.assigned_gid
                tracklet.add_sample(
                    embedding=emb,
                    capture_ts=cap_ts,
                    box=box,
                    crop_bgr=crop,
                    quality_score=q_res.confidence,
                )
                if was_confirmed and held_gid and tracklet.status == "CONFIRMED":
                    sim = self.identity_mgr.similarity_to(held_gid, emb)
                    if sim < MATCH_SIM_THRESHOLD:
                        tracklet.consecutive_weak += 1
                        if tracklet.consecutive_weak >= CONSECUTIVE_LOCK_FRAMES:
                            self.quality_gate.reset_track(cid, tid)
                            tracklet.reset_identity()
                    else:
                        tracklet.consecutive_weak = 0
                        tracklet.similarity_score = sim

        # 5. In-camera matching. GIDs already held on this camera are occupied,
        #    so a second person cannot inherit the first person's ID.
        for cid, worker, frame, cap_ts in active_workers:
            dets = cam_detections.get(cid, [])
            tracks_for_matching = []
            occupied_gids = set()
            enrollment = is_enrollment_camera(cid)
            req_samples = ENROLLMENT_FRAMES if enrollment else 1

            for det in dets:
                tracklet = self.tracklet_mgr.get_or_create(cid, det.track_id)
                tracklet.mark_seen(cap_ts)
                if tracklet.status == "CONFIRMED" and tracklet.assigned_gid:
                    gid = self.identity_mgr.resolve_gid(tracklet.assigned_gid)
                    tracklet.assigned_gid = gid
                    occupied_gids.add(gid)
                    self.identity_mgr.update_presence(gid, cid, det.track_id, cap_ts)
                    continue

                if tracklet.sample_count >= req_samples:
                    proto = tracklet.get_prototype_embedding()
                    if proto is not None:
                        sticky = tracklet.candidate_gid or tracklet.assigned_gid
                        tracks_for_matching.append((det.track_id, proto, det.box, sticky))

            if tracks_for_matching:
                assignments = self.identity_mgr.match_camera_tracks(
                    cid,
                    tracks_for_matching,
                    cap_ts,
                    occupied_gids=occupied_gids,
                    is_enrollment_cam=enrollment,
                    commit_exemplars=enrollment,
                )
                protos = {tid: proto for tid, proto, _, _ in tracks_for_matching}
                matched_tids = set(assignments.keys())
                for tid, _, _, _ in tracks_for_matching:
                    if tid not in matched_tids and not enrollment:
                        self.tracklet_mgr.get_or_create(cid, tid).note_match(
                            "", 0.0, MATCH_SIM_THRESHOLD, CONSECUTIVE_LOCK_FRAMES
                        )

                for tid, (gid, score) in assignments.items():
                    tracklet = self.tracklet_mgr.get_or_create(cid, tid)
                    resolved_gid = self.identity_mgr.resolve_gid(gid)
                    if enrollment:
                        tracklet.assigned_gid = resolved_gid
                        tracklet.candidate_gid = resolved_gid
                        tracklet.similarity_score = score
                        tracklet.status = "CONFIRMED"
                    else:
                        locked = tracklet.note_match(
                            resolved_gid, score, MATCH_SIM_THRESHOLD, CONSECUTIVE_LOCK_FRAMES
                        )
                        if locked:
                            proto = protos.get(tid)
                            if proto is not None:
                                self.identity_mgr.apply_match(
                                    resolved_gid, cid, tid, proto, cap_ts, score, log=True
                                )

            # 6. Render Visual Frame: Alpha-blended masks, contour outlines & styled labels
            annotated = frame.copy()
            mask_overlay = annotated.copy()

            for det in dets:
                tracklet = self.tracklet_mgr.get_or_create(cid, det.track_id)
                view = self.identity_mgr.identity_view(tracklet.assigned_gid) if tracklet.assigned_gid else None

                if view and tracklet.status == "CONFIRMED":
                    _, _, color = view
                elif tracklet.status == "TENTATIVE" or (enrollment and tracklet.sample_count < req_samples):
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
                view = self.identity_mgr.identity_view(tracklet.assigned_gid) if tracklet.assigned_gid else None

                if view and tracklet.status == "CONFIRMED":
                    _, name, color = view
                    label = f"[{name}] Sim: {tracklet.similarity_score:.2f} | #{det.track_id}"
                elif enrollment and tracklet.sample_count < req_samples:
                    label = f"Buffering ({tracklet.sample_count}/{req_samples}) | #{det.track_id}"
                    color = COLOR_TENTATIVE
                elif tracklet.status == "TENTATIVE":
                    label = (
                        f"TENTATIVE ({tracklet.consecutive_passes}/{CONSECUTIVE_LOCK_FRAMES})"
                        f" | #{det.track_id}"
                    )
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

            # Write to disk via a separate ffmpeg process. OpenCV's VideoWriter
            # shares FFmpeg with the RTSP readers and drops these frames.
            if cid in self.video_writers:
                if self.video_writers[cid] is None:
                    h, w = annotated.shape[:2]
                    out_path = os.path.join(self.run_folder, f"{cid}.mp4")
                    out_fps = worker.playback_fps if worker.playback_fps > 0 else TARGET_DETECTION_FPS
                    self.video_writers[cid] = FfmpegMp4Writer(out_path, out_fps, w, h)
                writer = self.video_writers[cid]
                if writer is not None and writer.isOpened():
                    writer.write(annotated)

        for dead_cam, dead_tid in self.tracklet_mgr.prune_dead_tracks(now):
            self.quality_gate.reset_track(dead_cam, dead_tid)

        # Periodic terminal progress logging (every 2.5s)
        cycle_count += 1
        now_mono = time.monotonic()
        if (now_mono - last_heartbeat) >= 2.5:
            stats = [
                f"{cid}: {w.fps:.1f}fps ({len(cam_detections.get(cid, []))} people)"
                for cid, w, _, _ in active_workers
            ]
            with self.identity_mgr.lock:
                total_enrolled = len(self.identity_mgr.identities)
            print(
                f"[Analysis] Cycle #{cycle_count:05d} | {' | '.join(stats)} | Total Enrolled: {total_enrolled} IDs",
                flush=True,
            )
            last_heartbeat = now_mono

        return last_heartbeat, cycle_count
