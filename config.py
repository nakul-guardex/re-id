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

# Sensitive URLs are sourced from environment variables if present:
DEFAULT_CAMERAS: Dict[str, Dict[str, Any]] = {
    "office_balcony": {
        "id": "office_balcony",
        "name": "Balcony (Enrollment)",
        "rtsp": os.getenv(
            "RTSP_URL_OFFICE_BALCONY",
            "rtsp://admin:Cctv%401234@122.176.35.187:2554/cam/realmonitor?channel=01&subtype=0",
        ),
        "fallback_video": FALLBACK_VIDEOS.get("office_balcony"),
        "is_enrollment": True,
        "enabled": True,
    },
    "office_1": {
        "id": "office_1",
        "name": "Office Cam 01",
        "rtsp": os.getenv(
            "RTSP_URL_OFFICE_1",
            "rtsp://admin:aniket12@122.176.35.187:5554/Streaming/Channels/101",
        ),
        "fallback_video": FALLBACK_VIDEOS.get("office_1"),
        "is_enrollment": False,
        "enabled": True,
    },
    "office_2": {
        "id": "office_2",
        "name": "Office Cam 02",
        "rtsp": os.getenv(
            "RTSP_URL_OFFICE_2",
            "rtsp://admin:Cctv%401234@122.176.35.187:2554/cam/realmonitor?channel=02&subtype=0",
        ),
        "fallback_video": FALLBACK_VIDEOS.get("office_2"),
        "is_enrollment": False,
        "enabled": True,
    },
    "office_3": {
        "id": "office_3",
        "name": "Office Cam 03",
        "rtsp": os.getenv(
            "RTSP_URL_OFFICE_3",
            "rtsp://admin:Cctv%401234@122.176.35.187:2554/cam/realmonitor?channel=03&subtype=0",
        ),
        "fallback_video": FALLBACK_VIDEOS.get("office_3"),
        "is_enrollment": False,
        "enabled": True,
    },
    "office_4": {
        "id": "office_4",
        "name": "Office Cam 04",
        "rtsp": os.getenv(
            "RTSP_URL_OFFICE_4",
            "rtsp://admin:Admin%40123@122.176.35.187:554/cam/realmonitor?channel=3&subtype=0",
        ),
        "fallback_video": FALLBACK_VIDEOS.get("office_4"),
        "is_enrollment": False,
        "enabled": True,
    },
    "office_5": {
        "id": "office_5",
        "name": "Office Cam 05",
        "rtsp": os.getenv(
            "RTSP_URL_OFFICE_5",
            "rtsp://admin:Admin%40123@122.176.35.187:554/cam/realmonitor?channel=6&subtype=0",
        ),
        "fallback_video": FALLBACK_VIDEOS.get("office_5"),
        "is_enrollment": False,
        "enabled": True,
    },
}

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
# Identity Layer: Matching, Merging & Cannot-Links
# -----------------------------------------------------------------------------
MATCH_SIM_THRESHOLD = 0.70       # Cosine similarity threshold for global identity matching
MATCH_MARGIN = 0.05              # Minimum margin over 2nd best candidate to confirm match
STICKY_HYSTERESIS_BOOST = 0.08   # Bonus similarity to keep current confirmed identity
STICKY_RELEASE_COUNT = 15        # Consecutive low scores before releasing confirmed track

MERGE_SIM_THRESHOLD = 0.75       # Strict cosine similarity for dynamic auto-merging (higher than match)
RECONCILIATION_INTERVAL_SEC = 1.2 # Periodic interval for cross-camera reconciliation worker
CANNOT_LINK_TIME_TOLERANCE_SEC = 0.35 # Temporal tolerance for same-camera co-occurrence
MAX_EXEMPLARS_PER_ID = 10        # Maximum diverse appearance exemplars stored per Global ID
EMA_CENTROID_ALPHA = 0.15        # Centroid update momentum

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
WEB_HOST = "0.0.0.0"
WEB_PORT = int(os.getenv("PORT", "8000"))
