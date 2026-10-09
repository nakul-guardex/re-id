#!/usr/bin/env python3
"""
OmniReID Multi-Camera Live Re-ID System - Main Entry Point
==========================================================
Boots the background AI models (YOLO11-Seg + ResNet50-IBN With BNNeck)
and launches the interactive Web Dashboard in IDLE state.
RTSP streams and analysis will only begin upon explicit user command.

Usage:
  python run.py
"""

import sys
from pathlib import Path

# Ensure multi_rtsp_reid and parent directory are on sys.path
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from multi_rtsp_reid.config import WEB_HOST, WEB_PORT
from multi_rtsp_reid.web.app import app

def main():
    print("\n" + "=" * 75)
    print("  🚀 OMNIREID: 6-CAMERA LIVE PERSON RE-ID & FUSION SYSTEM (v2)")
    print(f"  Web Dashboard: http://localhost:{WEB_PORT} (Host: {WEB_HOST})")
    print("  State: IDLE (Analysis starts via Web Dashboard: Test -> Start)")
    print("=" * 75 + "\n")

    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
