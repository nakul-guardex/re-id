#!/usr/bin/env python3
"""
Configuration for Multi-Camera Live Re-ID System (v2)
=====================================================
All settings, quality gate thresholds, camera configs, and model parameters.
RTSP credentials and sensitive URLs are read from environment variables
or can be configured via the Web UI at runtime.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from .env. Existing environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Model Weights Configuration
# -----------------------------------------------------------------------------
# Check local project weights first, fallback to parent directory or default
_local_yolo = WEIGHTS_DIR / "yolo11s-seg.pt"
_repo_yolo = PROJECT_ROOT.parent / "yolo11s-seg.pt"

if _local_yolo.exists():
    YOLO_MODEL_PATH = str(_local_yolo)
elif _repo_yolo.exists():
    YOLO_MODEL_PATH = str(_repo_yolo)
else:
    YOLO_MODEL_PATH = "yolo11s-seg.pt"

# Re-ID Weights (Market-1501 ResNet50-IBN)
_cache_reid = Path.home() / ".cache" / "torch" / "checkpoints" / "market_bot_R50-ibn.pth"
_local_reid = WEIGHTS_DIR / "market_bot_R50-ibn.pth"
REID_CHECKPOINT_PATH = str(_local_reid if _local_reid.exists() else _cache_reid)
REID_CHECKPOINT_URL = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth"

# Color Neutral Fill: ImageNet mean in BGR (B: 103, G: 116, R: 124)
# When normalized with ImageNet mean/std, these pixels become ~0.0
IMAGENET_MEAN_BGR = (103, 116, 124)

# -----------------------------------------------------------------------------
# Camera Configuration (Default 6 RTSP streams)
# -----------------------------------------------------------------------------
# Optional local fallback video files for offline simulation/testing
FALLBACK_VIDEOS: Dict[str, str] = {}

# RTSP URLs come from environment variables only — do not hardcode credentials.
DEFAULT_CAMERAS: Dict[str, Dict[str, Any]] = {
    "office_balcony": {
        "id": "office_balcony",
        "name": "Balcony (Enrollment)",
        "rtsp": os.getenv("RTSP_URL_OFFICE_BALCONY", ""),
        "fallback_video": FALLBACK_VIDEOS.get("office_balcony"),
        "is_enrollment": True,
        "enabled": True,
    },
    "office_1": {
        "id": "office_1",
        "name": "Office Cam 01",
        "rtsp": os.getenv("RTSP_URL_OFFICE_1", ""),
        "fallback_video": FALLBACK_VIDEOS.get("office_1"),
        "is_enrollment": False,
        "enabled": False,
    },
    "office_2": {
        "id": "office_2",
        "name": "Office Cam 02",
        "rtsp": os.getenv("RTSP_URL_OFFICE_2", ""),
        "fallback_video": FALLBACK_VIDEOS.get("office_2"),
        "is_enrollment": False,
        "enabled": False,
    },
    "office_3": {
        "id": "office_3",
        "name": "Office Cam 03",
        "rtsp": os.getenv("RTSP_URL_OFFICE_3", ""),
        "fallback_video": FALLBACK_VIDEOS.get("office_3"),
        "is_enrollment": False,
        "enabled": False,
    },
    "office_4": {
        "id": "office_4",
        "name": "Office Cam 04",
        "rtsp": os.getenv("RTSP_URL_OFFICE_4", ""),
        "fallback_video": FALLBACK_VIDEOS.get("office_4"),
        "is_enrollment": False,
        "enabled": False,
    },
    "office_5": {
        "id": "office_5",
        "name": "Office Cam 05",
        "rtsp": os.getenv("RTSP_URL_OFFICE_5", ""),
        "fallback_video": FALLBACK_VIDEOS.get("office_5"),
        "is_enrollment": False,
        "enabled": False,
    },
}


def is_enrollment_camera(cam_id: str) -> bool:
    cam = DEFAULT_CAMERAS.get(cam_id) or {}
    return bool(cam.get("is_enrollment"))

# -----------------------------------------------------------------------------
# Ingestion & Video Pacing
# -----------------------------------------------------------------------------
TARGET_DETECTION_FPS = 5.0      # Target FPS for YOLO-seg detection per camera
INGESTION_DOWNSCALE_WIDTH = 960  # Downscale 1080p/4K incoming frames for fast inference
REORDER_BUFFER_WINDOW_SEC = 1.5  # Reorder buffer window in seconds based on capture timestamps
RTSP_PROBE_TIMEOUT_SEC = 6.0     # Probe timeout per camera during test
RTSP_PROBE_MIN_FRAMES = 15       # Minimum frames to declare stream healthy in probe

# -----------------------------------------------------------------------------
# Clear-Frame Quality Gate Thresholds
# -----------------------------------------------------------------------------
QG_MIN_HEIGHT = 90               # Minimum person crop height (px)
QG_MIN_WIDTH = 45                # Minimum person crop width (px)
QG_BORDER_MARGIN = 15            # Distance from image border (px) to reject half-cut bodies
QG_MASK_SOLIDITY_MIN = 0.20      # Minimum mask-to-box area ratio
QG_MASK_SOLIDITY_MAX = 0.85      # Maximum mask-to-box area ratio
QG_DET_CONFIDENCE = 0.45         # Minimum YOLO detection confidence
QG_BLUR_LAPLACIAN_VAR = 35.0     # Minimum Laplacian variance for sharpness
QG_MAX_OCCLUSION_IOU = 0.35      # Max IoU with adjacent persons to reject heavy occlusions
QG_TEMPORAL_SPACING_SEC = 0.40   # Minimum time gap between accepted crops for the same track
QG_MIN_GATE_PASSES = 3           # Minimum gate-passing crops required before identity decision

# -----------------------------------------------------------------------------
# Tracker-First & State Machine Config
# -----------------------------------------------------------------------------
TRACKER_TYPE = 'bytetrack'       # Options: 'bytetrack', 'pbf' (Particle Based Filtering)
REID_CHECK_INTERVAL_SEC = 2.0    # Wait time between Re-ID checks for UNKNOWN IDs
CONSECUTIVE_LOCK_FRAMES = 3      # Consecutive passes required to transition TENTATIVE -> LOCKED
MATCH_SIM_THRESHOLD = 0.65       # Cosine similarity threshold for global identity matching
MAX_EXEMPLARS_PER_ID = 5         # Maximum diverse appearance exemplars stored per Global ID

# Matching only. Identities are enrolled on the balcony and assigned on other cameras.
MATCH_MARGIN = 0.05              # Best match must beat the runner-up by this much
STICKY_HYSTERESIS_BOOST = 0.08   # Bonus so a track keeps its current identity
EMA_CENTROID_ALPHA = 0.15        # How fast an identity centroid follows new embeddings

# -----------------------------------------------------------------------------
# Balcony Enroller Config
# -----------------------------------------------------------------------------
TRIPWIRE_Y_PCT = 0.5             # Percentage of screen height (0.0 to 1.0) for tripwire line. Updatable from UI.
ENROLLMENT_FRAMES = 5            # Number of frames to capture when tripwire is crossed

# -----------------------------------------------------------------------------
# Visual Palettes & Dashboard
# -----------------------------------------------------------------------------
PALETTE = [
    (50, 230, 50),     # Lime Green (ID 1)
    (255, 220, 0),    # Cyan (ID 2)
    (220, 50, 220),   # Magenta (ID 3)
    (0, 165, 255),    # Orange (ID 4)
    (255, 90, 30),    # Blue (ID 5)
    (140, 255, 100),  # Mint (ID 6)
    (255, 120, 180),  # Pink (ID 7)
    (80, 200, 255),   # Sky (ID 8)
    (180, 120, 255),  # Violet (ID 9)
    (100, 255, 200),  # Turquoise (ID 10)
]
COLOR_TENTATIVE = (0, 200, 255)   # Amber badge for buffering/tentative tracks
COLOR_UNKNOWN = (150, 150, 150)   # Neutral gray

# Web Server Settings
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("PORT", "8000"))
# Optional shared secret for mutating dashboard APIs. When empty, routes stay open (local use).
API_TOKEN = os.getenv("API_TOKEN", "")
