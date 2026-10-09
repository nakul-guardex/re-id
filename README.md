# OmniReID (v2) - 6-Camera Live Person Re-ID System

A real-time Person Re-Identification & Cross-Camera Fusion system across **6 live RTSP cameras** running on an NVIDIA GPU VM (or local accelerator).

---

## 🌟 Key Features

1. **User-Controlled Session Lifecycle**:
   - Web UI driven: `IDLE → PROBING (Test Cameras) → READY → RUNNING (Start Analysis) → STOPPING → IDLE`.
   - Never starts streams automatically; starts all cameras from a unified monotonic clock $t_0$.
2. **Clear-Frame Quality Gate (First 3 Clear Frames Rule)**:
   - Filters out boundary cut-offs, motion blur (Laplacian variance), tiny specks (< 90px), low confidence (< 0.45), and heavy occlusions.
   - Requires $\ge 3$ temporally spaced ($> 0.4$s) clear crops before identity evaluation.
3. **Segmented Foreground Extraction**:
   - YOLO11-Seg extracts foreground masks; background is filled with ImageNet BGR mean `(103, 116, 124)` so background pixels evaluate to $\approx 0.0$ in PyTorch.
   - Extracts 2048-d appearance embeddings via **ResNet50-IBN With BNNeck** (FastReID checkpoint).
4. **In-Camera 1-to-1 Hungarian Exclusivity**:
   - Guarantees no two people in the same camera can ever receive the same Global ID.
   - Sticky hysteresis prevents track label flickering.
5. **Cross-Camera Multi-Exemplar Matching**:
   - Supports overlapping fields of view: assigns identical Global IDs to the same individual across cameras.
6. **Hard Spatial Co-Occurrence Blacklist & Reversible Merging**:
   - If two persons ever appear in the same camera at the same time, they are permanently barred from merging.
   - If two independently enrolled IDs match ($\text{sim} \ge 0.54$), they are merged with a full rollback log and a one-click **Undo** button on the UI!
7. **Glassmorphic Web Dashboard**:
   - 6-camera MJPEG video grid, FPS counters, live discovered target cards, event audit log, and camera RTSP config modal.

---

## 🚀 Quickstart

### 1. VM Installation
Run the one-click setup script on Ubuntu/Debian:
```bash
bash setup_vm.sh
```

### 2. Run the System
```bash
python run.py
```
Open your browser to: **`http://localhost:8000`**

### 3. Usage Workflow in the Web Dashboard:
1. Click **🔍 Test Cameras**: Probes all 6 RTSP feeds in parallel and measures live FPS.
2. Click **▶ Start Analysis**: Begins real-time detection, tracking, Re-ID, and cross-camera matching.
3. Click **⏹ Stop Analysis**: Gracefully shuts down workers and flushes logs.
4. If a merge event occurred, review the live log and click **↩ Undo Merge** if you ever wish to split them back.

---

## 📁 Project Structure

```
multi_rtsp_reid/
├── config.py                     # All thresholds, buffers, and camera configs
├── run.py                        # Main entry point (starts server in IDLE)
├── setup_vm.sh                   # One-click VM setup script
├── requirements.txt              # Production dependencies
├── models/
│   └── reid_bnneck.py            # FastReID ResNet50-IBN With BNNeck (2048-d)
├── core/
│   ├── session_manager.py        # State machine (IDLE -> PROBING -> RUNNING)
│   ├── stream_worker.py          # Zero-latency RTSP capture with monotonic timestamps
│   ├── quality_gate.py           # Clear-frame filter (size, blur, solidity, spacing)
│   ├── foreground_extractor.py   # Instance mask crop & ImageNet neutral mean fill
│   ├── tracker_engine.py         # YOLO11-Seg + independent ByteTrack per camera
│   ├── tracklet_builder.py       # Gate-passing sample accumulator + outlier rejection
│   ├── identity_manager.py       # Hungarian solver, co-occurrence blacklist, reversible merges
│   └── event_logger.py           # JSONL session event logger for replay/calibration
├── pipeline/
│   └── multicam_coordinator.py   # Master coordinator driving inference & rendering
├── tools/
│   ├── calibrate_thresholds.py   # Similarity distribution analyzer & threshold suggestions
│   └── evaluate.py               # ID switches, cannot-links, and stability metrics
└── web/
    ├── app.py                    # Flask server with REST API & MJPEG endpoints
    ├── templates/index.html      # Responsive 6-camera dashboard
    └── static/
        ├── css/dashboard.css     # Bespoke glassmorphism dark theme
        └── js/dashboard.js       # Real-time polling client controller
```
