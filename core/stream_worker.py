#!/usr/bin/env python3
"""
Threaded RTSP Stream Worker
===========================
Captures frames asynchronously over RTSP (TCP transport) with zero-latency
1-frame buffer, accurate monotonic capture timestamps, automatic reconnection
with exponential backoff, hot-joining capability, and a non-blocking probe mode.
"""

import enum
import os
import threading
import time
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Force TCP transport for RTSP to prevent UDP packet tearing
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class StreamStatus(str, enum.Enum):
    OFFLINE = "OFFLINE"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    HOT_JOINED = "HOT_JOINED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class StreamWorker:
    """
    Asynchronous RTSP & Video Capture Worker.
    Keeps only the freshest frame in memory and timestamps every frame
    with time.monotonic() to eliminate network delay drift.
    """

    @staticmethod
    def sanitize_rtsp_url(url: str) -> str:
        """Sanitizes RTSP credentials by percent-encoding symbols like @ or # in passwords."""
        url = (url or "").strip()
        if url.startswith("rtsp://") and url.count("@") > 1:
            last_at = url.rfind("@")
            cred_part = url[7:last_at]
            rest_part = url[last_at:]
            if ":" in cred_part:
                user, pwd = cred_part.split(":", 1)
                pwd_encoded = pwd.replace("@", "%40").replace("#", "%23")
                return f"rtsp://{user}:{pwd_encoded}{rest_part}"
        return url

    def __init__(
        self,
        cam_id: str,
        name: str,
        rtsp_url: str,
        fallback_video: Optional[str] = None,
        downscale_width: int = 960,
    ):
        self.cam_id = cam_id
        self.name = name
        self.rtsp_url = self.sanitize_rtsp_url(rtsp_url)
        self.fallback_video = fallback_video
        self.downscale_width = downscale_width

        self.lock = threading.Lock()
        self.status = StreamStatus.OFFLINE
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Live frame buffer
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None
        self.latest_capture_ts: float = 0.0  # monotonic timestamp of capture
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self.reconnect_count: int = 0
        self.consecutive_failures: int = 0
        self.joined_late: bool = False
        self.last_error: str = ""
        self.video_writer = None
        self.file_mode = False
        self.finished = False
        self.playback_fps = 0.0
        self._consumed = threading.Event()
        self._consumed.set()

        # Placeholder JPEG for offline streams
        self._placeholder_jpeg = self._create_placeholder(f"[{self.name}] Camera Offline")

    def _create_placeholder(self, text: str) -> bytes:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (20, 24, 28)
        cv2.putText(img, text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 2, cv2.LINE_AA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()

    def start(self, is_hot_join: bool = False):
        """Starts the capture thread."""
        with self.lock:
            if self.running:
                return
            if self.thread and self.thread.is_alive():
                return
            self.running = True
            self.joined_late = is_hot_join
            self.status = StreamStatus.HOT_JOINED if is_hot_join else StreamStatus.CONNECTING
            self.thread = threading.Thread(target=self._capture_loop, name=f"Worker-{self.cam_id}", daemon=True)
            self.thread.start()

    def stop(self, timeout: float = 2.0):
        """Gracefully terminates capture thread with timeout."""
        with self.lock:
            self.running = False
            self.status = StreamStatus.STOPPED
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
        self.thread = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def probe(self, timeout_sec: float = 6.0, min_frames: int = 15) -> Tuple[bool, str, float, Tuple[int, int]]:
        """
        Tests if the RTSP stream is healthy before starting analysis.
        Returns: (success, message, measured_fps, (width, height))
        """
        source = self.rtsp_url or self.fallback_video
        if not source:
            return False, "No RTSP URL or fallback video specified", 0.0, (0, 0)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return False, "Failed to connect to stream source", 0.0, (0, 0)

        t_start = time.monotonic()
        frames_received = 0
        w, h = 0, 0
        last_frame = None

        while (time.monotonic() - t_start) < timeout_sec:
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_received += 1
                last_frame = frame
                if w == 0:
                    h, w = frame.shape[:2]
                if frames_received >= min_frames:
                    break
            else:
                time.sleep(0.05)

        cap.release()
        if last_frame is not None:
            with self.lock:
                self.latest_raw_frame = last_frame
                self.latest_capture_ts = time.monotonic()
                self.width = last_frame.shape[1]
                self.height = last_frame.shape[0]
        elapsed = time.monotonic() - t_start

        if frames_received < min_frames:
            return False, f"Timed out: received only {frames_received}/{min_frames} frames", 0.0, (w, h)

        fps = frames_received / max(elapsed, 0.001)
        return True, f"Stream healthy ({w}x{h} @ {fps:.1f} FPS)", fps, (w, h)

    def _capture_loop(self):
        """Main worker capture loop with exponential backoff reconnection."""
        backoff_sec = 1.0

        while self.running:
            source = self.rtsp_url or self.fallback_video
            self.file_mode = bool(source) and not str(source).startswith("rtsp://")
            if not source:
                with self.lock:
                    self.status = StreamStatus.FAILED
                    self.last_error = "No RTSP URL configured"
                time.sleep(2.0)
                continue

            with self.lock:
                self.status = StreamStatus.CONNECTING

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                with self.lock:
                    self.status = StreamStatus.FAILED
                    self.reconnect_count += 1
                    self.last_error = "Connection failed"
                    if self.file_mode:
                        self.finished = True
                        self.running = False
                if self.file_mode:
                    print(f"[Worker-{self.cam_id}] Could not open recorded video: {source}", flush=True)
                    return
                time.sleep(min(backoff_sec, 15.0))
                backoff_sec *= 1.5
                continue

            if self.file_mode:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self.playback_fps = fps if 1.0 <= fps <= 120.0 else 25.0

            # Connected successfully
            backoff_sec = 1.0
            with self.lock:
                self.status = StreamStatus.ONLINE if not self.joined_late else StreamStatus.HOT_JOINED
                self.consecutive_failures = 0
            if self.file_mode:
                print(
                    f"[Worker-{self.cam_id}] Playing recorded video {source} at {self.playback_fps:.1f} fps",
                    flush=True,
                )
            else:
                print(f"[Worker-{self.cam_id}] Live RTSP connected: {self.name}", flush=True)

            fps_timer = time.monotonic()
            fps_counter = 0

            while self.running:
                if self.file_mode and not self._consumed.wait(timeout=0.2):
                    continue

                ret, frame = cap.read()
                capture_ts = time.monotonic()

                if not ret or frame is None:
                    if self.file_mode:
                        with self.lock:
                            self.finished = True
                            self.running = False
                            self.status = StreamStatus.STOPPED
                        print(f"[Worker-{self.cam_id}] Recorded video finished.", flush=True)
                        cap.release()
                        return
                    with self.lock:
                        self.consecutive_failures += 1
                    if self.consecutive_failures > 30:
                        break  # trigger reconnect
                    time.sleep(0.01)
                    continue

                # Downscale 1080p/4K incoming frames for high-throughput pipeline
                h_orig, w_orig = frame.shape[:2]
                if self.downscale_width and w_orig > self.downscale_width:
                    scale = self.downscale_width / float(w_orig)
                    new_h = int(round(h_orig * scale))
                    new_w = self.downscale_width - (self.downscale_width % 2)
                    new_h = new_h - (new_h % 2)
                    proc_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    proc_frame = frame

                # Update FPS calculation
                fps_counter += 1
                now = time.monotonic()
                if (now - fps_timer) >= 1.0:
                    current_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now
                else:
                    current_fps = self.fps

                # JPEG streaming disabled by user
                raw_jpeg = None

                # Thread-safe buffer update (always 1-frame deep)
                with self.lock:
                    self.latest_raw_frame = proc_frame
                    if self.latest_annotated_frame is None and raw_jpeg:
                        self.latest_jpeg = raw_jpeg
                    self.latest_capture_ts = capture_ts
                    self.frame_count += 1
                    self.fps = current_fps
                    self.width = proc_frame.shape[1]
                    self.height = proc_frame.shape[0]
                    if self.file_mode:
                        self._consumed.clear()

            cap.release()
            time.sleep(1.0)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float, float, StreamStatus]:
        """Returns (raw_frame_copy, capture_timestamp, fps, status)."""
        with self.lock:
            if self.latest_raw_frame is None:
                return None, self.latest_capture_ts, self.fps, self.status
            return self.latest_raw_frame.copy(), self.latest_capture_ts, self.fps, self.status

    def release_file_frame(self) -> None:
        """Let a recorded-video reader fetch the next frame. Live RTSP ignores this."""
        if self.file_mode:
            self._consumed.set()

    def set_annotated_frame(self, frame: np.ndarray):
        """Stores the processed AI frame for the live web feed."""
        if frame is None:
            return
        
        with self.lock:
            self.latest_annotated_frame = frame

    def get_jpeg(self) -> bytes:
        """Returns the latest JPEG bytes for MJPEG web feed."""
        with self.lock:
            frame = self.latest_annotated_frame if self.latest_annotated_frame is not None else self.latest_raw_frame
            if frame is None:
                return self._placeholder_jpeg

            frame_copy = frame.copy()

            # Draw the Tripwire exactly where the slider sets it
            if self.cam_id == "office_balcony":
                import multi_rtsp_reid.config as cfg
                y_pct = getattr(cfg, 'TRIPWIRE_Y_PCT', 0.5)
                y_line = int(frame_copy.shape[0] * y_pct)
                cv2.line(frame_copy, (0, y_line), (frame_copy.shape[1], y_line), (0, 0, 255), 2)

            success, buf = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if success:
                return buf.tobytes()
            return self._placeholder_jpeg

    def get_health_metrics(self) -> Dict[str, Any]:
        """Returns status dictionary for dashboard health monitor."""
        with self.lock:
            return {
                "cam_id": self.cam_id,
                "name": self.name,
                "status": self.status.value,
                "fps": round(self.fps, 1),
                "resolution": f"{self.width}x{self.height}" if self.width > 0 else "N/A",
                "reconnects": self.reconnect_count,
                "joined_late": self.joined_late,
                "frame_count": self.frame_count,
                "last_error": self.last_error,
            }
