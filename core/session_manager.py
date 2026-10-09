#!/usr/bin/env python3
"""
Session Manager State Machine
=============================
Governs the user-controlled lifecycle:
  IDLE -> PROBING -> READY -> RUNNING -> STOPPING -> IDLE
Guarantees thread-safe transitions, generates session IDs, probes streams
in parallel, and manages hot-joining / graceful shutdown.
"""

from datetime import datetime
import enum
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Any

from ..config import INGESTION_DOWNSCALE_WIDTH, RECORDINGS_DIR
from .stream_worker import StreamWorker, StreamStatus


class SessionState(str, enum.Enum):
    IDLE = "IDLE"
    PROBING = "PROBING"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"


class SessionManager:
    """
    Manages session lifecycle and camera workers.
    Analysis only starts when the user issues an explicit Start command.
    """

    def __init__(
        self,
        cameras_config: Dict[str, Dict[str, Any]],
        on_session_start: Optional[Callable[[str, List[str]], None]] = None,
        on_session_stop: Optional[Callable[[str], None]] = None,
    ):
        self.lock = threading.Lock()
        self.state = SessionState.IDLE
        self.session_id: Optional[str] = None
        self.start_mono_time: float = 0.0
        self.cameras_config = cameras_config

        # Callbacks to notify pipeline coordinator
        self.on_session_start = on_session_start
        self.on_session_stop = on_session_stop

        # Workers registry
        self.workers: Dict[str, StreamWorker] = {}
        self.probe_results: Dict[str, Dict[str, Any]] = {}
        self.recordings_dir = RECORDINGS_DIR

        # Instantiate stream workers (in stopped/offline state initially)
        for cam_id, cfg in self.cameras_config.items():
            self.workers[cam_id] = StreamWorker(
                cam_id=cam_id,
                name=cfg["name"],
                rtsp_url=cfg.get("rtsp", ""),
                fallback_video=cfg.get("fallback_video"),
                downscale_width=INGESTION_DOWNSCALE_WIDTH,
            )

    def probe_cameras(self, timeout_sec: float = 6.0, min_frames: int = 15, camera_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Transitions to PROBING, tests all configured streams in parallel,
        and transitions to READY upon completion.
        """
        with self.lock:
            if self.state not in (SessionState.IDLE, SessionState.READY):
                raise RuntimeError(f"Cannot probe cameras while in state {self.state}")
            self.state = SessionState.PROBING
            self.probe_results.clear()

        threads = []
        results: Dict[str, Dict[str, Any]] = {}
        res_lock = threading.Lock()

        def _probe_worker(cid: str, worker: StreamWorker):
            ok, msg, fps, (w, h) = worker.probe(timeout_sec=timeout_sec, min_frames=min_frames)
            with res_lock:
                results[cid] = {
                    "ok": ok,
                    "message": msg,
                    "fps": round(fps, 1),
                    "resolution": f"{w}x{h}" if w > 0 else "N/A",
                }

        selected = set(camera_ids) if camera_ids else None
        for cid, worker in self.workers.items():
            cfg = self.cameras_config.get(cid, {})
            if selected is not None and cid not in selected:
                with res_lock:
                    results[cid] = {
                        "ok": True,
                        "message": "Not selected",
                        "fps": 0.0,
                        "resolution": "N/A",
                        "standby": True,
                    }
                continue
            if not cfg.get("enabled", True) or not (cfg.get("rtsp") or cfg.get("fallback_video")):
                with res_lock:
                    results[cid] = {
                        "ok": True,
                        "message": "Standby / Empty",
                        "fps": 0.0,
                        "resolution": "N/A",
                        "standby": True,
                    }
                continue
            t = threading.Thread(target=_probe_worker, args=(cid, worker), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=timeout_sec + 2.0)

        with self.lock:
            self.probe_results = results
            self.state = SessionState.READY
            active_configured = [
                cid for cid, cfg in self.cameras_config.items()
                if (selected is None or cid in selected)
                and cfg.get("enabled", True)
                and (cfg.get("rtsp") or cfg.get("fallback_video"))
            ]
            passing_count = sum(1 for cid in active_configured if results.get(cid, {}).get("ok"))
            total_active = len(active_configured) if active_configured else len(self.workers)
            return {
                "state": self.state.value,
                "passing_cameras": f"{passing_count}/{total_active}",
                "details": self.probe_results,
            }

    def start_session(self, camera_ids: Optional[List[str]] = None) -> str:
        """
        Transitions from READY to RUNNING. Starts worker capture threads
        from a shared monotonic clock t0.
        """
        with self.lock:
            if self.state != SessionState.READY and self.state != SessionState.IDLE:
                raise RuntimeError(f"Cannot start session from state {self.state}. Please probe or stop first.")

            # Create unique session ID
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"sess_{now_str}_{uuid.uuid4().hex[:6]}"
            self.start_mono_time = time.monotonic()
            self.state = SessionState.RUNNING
            sess_id = self.session_id

        active_cams = camera_ids or list(self.workers.keys())

        print(f"\n[SessionManager] >>> STARTING SESSION: {sess_id} <<<", flush=True)
        print(f"[SessionManager] Launching camera worker(s)...", flush=True)

        for cid in active_cams:
            if cid in self.workers:
                cfg = self.cameras_config.get(cid, {})
                if not cfg.get("enabled", True) or not (cfg.get("rtsp") or cfg.get("fallback_video")):
                    continue
                self.workers[cid].start(is_hot_join=False)

        # Notify coordinator callback outside lock
        if self.on_session_start:
            self.on_session_start(sess_id, active_cams)

        return sess_id

    def stop_session(self) -> Dict[str, Any]:
        """
        Transitions from RUNNING to STOPPING -> IDLE.
        Stops workers with timeout joins to prevent RTSP hangs.
        """
        with self.lock:
            if self.state != SessionState.RUNNING:
                return {"state": self.state.value, "message": "Session was not running"}
            self.state = SessionState.STOPPING
            sess_id = self.session_id
            uptime = time.monotonic() - self.start_mono_time if self.start_mono_time > 0 else 0.0

        print(f"\n[SessionManager] >>> STOPPING SESSION: {sess_id} <<<", flush=True)

        # Signal all workers to stop concurrently
        for cid, worker in self.workers.items():
            with worker.lock:
                worker.running = False
                worker.status = StreamStatus.STOPPED

        # Wait for all workers to finish up to a shared maximum timeout
        deadline = time.monotonic() + 2.0
        for cid, worker in self.workers.items():
            if worker.thread and worker.thread.is_alive():
                remaining = max(0.0, deadline - time.monotonic())
                worker.thread.join(timeout=remaining)
            if worker.thread and not worker.thread.is_alive():
                worker.thread = None
            if worker.video_writer is not None:
                worker.video_writer.release()
                worker.video_writer = None

        # Notify coordinator callback outside lock
        if self.on_session_stop and sess_id:
            self.on_session_stop(sess_id)

        with self.lock:
            self.state = SessionState.IDLE
            self.session_id = None
            self.start_mono_time = 0.0

        return {
            "state": "IDLE",
            "stopped_session_id": sess_id,
            "uptime_sec": round(uptime, 1),
        }

    def update_camera_config(self, cam_id: str, new_rtsp: str) -> bool:
        """Dynamically updates camera RTSP URL without restarting server."""
        with self.lock:
            if cam_id in self.workers:
                worker = self.workers[cam_id]
                worker.rtsp_url = worker.sanitize_rtsp_url(new_rtsp)
                if cam_id in self.cameras_config:
                    self.cameras_config[cam_id]["rtsp"] = worker.rtsp_url
                return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status dictionary for dashboard."""
        with self.lock:
            uptime = (time.monotonic() - self.start_mono_time) if self.state == SessionState.RUNNING else 0.0
            cameras_health = {cid: w.get_health_metrics() for cid, w in self.workers.items()}
            return {
                "state": self.state.value,
                "session_id": self.session_id,
                "uptime_seconds": round(uptime, 1),
                "cameras": cameras_health,
                "probe_results": self.probe_results,
            }
