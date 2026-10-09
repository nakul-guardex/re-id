#!/usr/bin/env python3
"""
Web Dashboard Application & API Server
======================================
Serves the live 6-camera video feeds (MJPEG), REST API endpoints
for session lifecycle (Probe, Start, Stop), real-time camera health,
live gallery person cards, event logs, and reversible merge rollback.
"""

from pathlib import Path
import secrets
import time
from flask import Flask, Response, jsonify, render_template, request

from ..config import (
    API_TOKEN,
    DEFAULT_CAMERAS,
    EMA_CENTROID_ALPHA,
    LOGS_DIR,
    MATCH_MARGIN,
    MATCH_SIM_THRESHOLD,
    MAX_EXEMPLARS_PER_ID,
    PALETTE,
    REID_CHECKPOINT_PATH,
    STICKY_HYSTERESIS_BOOST,
    WEB_HOST,
    WEB_PORT,
    YOLO_MODEL_PATH,
)
from ..core.event_logger import SessionEventLogger
from ..core.identity_manager import GlobalIdentityManager
from ..core.quality_gate import ClearFrameQualityGate
from ..core.session_manager import SessionManager, SessionState
from ..core.tracker_engine import MultiCameraTrackerEngine
from ..core.tracklet_builder import TrackletManager
from ..models.reid_bnneck import ResNet50IBN_WithBNNeck
from ..pipeline.multicam_coordinator import MultiCameraCoordinator

# Initialize Flask App
APP_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

# -----------------------------------------------------------------------------
# Global Component Singletons
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  MULTI-CAMERA LIVE RE-ID SYSTEM: INITIALIZING COMPONENTS")
print("=" * 70)

event_logger = SessionEventLogger(LOGS_DIR)
quality_gate = ClearFrameQualityGate()
tracklet_mgr = TrackletManager()

identity_mgr = GlobalIdentityManager(
    palette=PALETTE,
    match_threshold=MATCH_SIM_THRESHOLD,
    match_margin=MATCH_MARGIN,
    sticky_boost=STICKY_HYSTERESIS_BOOST,
    max_exemplars=MAX_EXEMPLARS_PER_ID,
    ema_alpha=EMA_CENTROID_ALPHA,
)

# Same-origin cookie. Cross-site POSTs do not include it (SameSite=Lax).
_CSRF_TOKEN = API_TOKEN or secrets.token_hex(16)

tracker_engine = MultiCameraTrackerEngine(model_path=YOLO_MODEL_PATH)
reid_extractor = ResNet50IBN_WithBNNeck(checkpoint_path=REID_CHECKPOINT_PATH)

session_mgr = SessionManager(cameras_config=DEFAULT_CAMERAS)

coordinator = MultiCameraCoordinator(
    session_manager=session_mgr,
    tracker_engine=tracker_engine,
    reid_extractor=reid_extractor,
    identity_manager=identity_mgr,
    quality_gate=quality_gate,
    tracklet_manager=tracklet_mgr,
    event_logger=event_logger,
)

print("[Server] Core models & pipeline initialized in IDLE state.")
if not API_TOKEN:
    print("[Server] API_TOKEN is unset. Mutating routes require the dashboard session cookie.")
print("=" * 70 + "\n")


@app.before_request
def _require_csrf_on_post():
    if request.method != "POST":
        return None
    if request.cookies.get("omni_csrf") != _CSRF_TOKEN:
        return jsonify({"error": "missing or invalid session cookie"}), 401
    return None


@app.after_request
def _set_csrf_cookie(response):
    response.set_cookie(
        "omni_csrf",
        _CSRF_TOKEN,
        httponly=True,
        samesite="Lax",
    )
    return response


# -----------------------------------------------------------------------------
# HTML Views
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Main dashboard interface."""
    return render_template("index.html", cameras=DEFAULT_CAMERAS)


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns current state machine status and camera health."""
    status = session_mgr.get_status()
    status["enrolled_persons_count"] = len(identity_mgr.identities)
    status["gallery"] = identity_mgr.get_gallery_cards()
    status["events"] = list(reversed(identity_mgr.event_log))
    return jsonify(status)


@app.route("/api/probe", methods=["POST"])
def probe_cameras():
    """Runs parallel test probe on all configured camera streams."""
    try:
        data = request.get_json(silent=True) or {}
        res = session_mgr.probe_cameras(camera_ids=data.get("cameras"))
        return jsonify(res)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400


@app.route("/api/start", methods=["POST"])
def start_session():
    """Starts analysis from shared monotonic clock."""
    try:
        data = request.get_json(silent=True) if request.is_json else {}
        data = data or {}
        selected_cameras = data.get("cameras")
        session_id = session_mgr.start_session(camera_ids=selected_cameras)
        return jsonify({"status": "RUNNING", "session_id": session_id})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400


@app.route("/api/stop", methods=["POST"])
def stop_session():
    """Stops analysis session gracefully."""
    try:
        res = session_mgr.stop_session()
        return jsonify(res)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400


@app.route("/api/gallery", methods=["GET"])
def get_gallery():
    """Returns list of all active enrolled persons for UI sidebar."""
    cards = identity_mgr.get_gallery_cards()
    return jsonify({"gallery": cards})


@app.route("/api/events", methods=["GET"])
def get_events():
    """Returns the live enrollment and match event log."""
    return jsonify({"events": list(reversed(identity_mgr.event_log))})


@app.route("/api/camera/update", methods=["POST"])
def update_camera():
    """Updates RTSP URL for a camera dynamically."""
    data = request.get_json(silent=True) or {}
    cam_id = data.get("cam_id")
    rtsp_url = data.get("rtsp")
    if not cam_id or rtsp_url is None:
        return jsonify({"error": "cam_id and rtsp required"}), 400

    ok = session_mgr.update_camera_config(cam_id, rtsp_url)
    return jsonify({"success": ok})


@app.route("/api/tripwire", methods=["POST"])
def update_tripwire():
    """Updates the Tripwire Y Percentage."""
    data = request.get_json(silent=True) or {}
    y_pct = data.get("y_pct")
    if y_pct is None:
        return jsonify({"error": "y_pct required"}), 400
    try:
        pct = float(y_pct)
    except (TypeError, ValueError):
        return jsonify({"error": "y_pct must be a number"}), 400
    pct = min(100.0, max(0.0, pct))
    import multi_rtsp_reid.config as cfg
    cfg.TRIPWIRE_Y_PCT = pct / 100.0
    return jsonify({"success": True, "y_pct": cfg.TRIPWIRE_Y_PCT})


# -----------------------------------------------------------------------------
# MJPEG Live Video Feeds
# -----------------------------------------------------------------------------
def generate_mjpeg(cam_id: str):
    """Generator yielding JPEG frames for continuous MJPEG streaming."""
    worker = session_mgr.workers.get(cam_id)
    while True:
        if worker is not None:
            frame_bytes = worker.get_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS stream delivery


@app.route("/stream/<cam_id>")
def stream_camera(cam_id: str):
    """MJPEG stream endpoint for a camera tile."""
    if cam_id not in session_mgr.workers:
        return "Camera not found", 404
    return Response(
        generate_mjpeg(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/snapshot/<cam_id>")
def snapshot_camera(cam_id: str):
    """Returns the latest single JPEG snapshot for a camera tile."""
    worker = session_mgr.workers.get(cam_id)
    if worker is None:
        return "Camera not found", 404
    return Response(worker.get_jpeg(), mimetype="image/jpeg")


def create_app():
    return app


if __name__ == "__main__":
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
