import argparse
import time
import os
import cv2
import numpy as np

from config import (
    COLOR_TENTATIVE, COLOR_UNKNOWN, IMAGENET_MEAN_BGR, PALETTE,
    MATCH_SIM_THRESHOLD, MATCH_MARGIN, STICKY_HYSTERESIS_BOOST
)
from core.tracker_engine import MultiCameraTrackerEngine
from core.quality_gate import ClearFrameQualityGate
from core.tracklet_builder import TrackletManager
from core.identity_manager import GlobalIdentityManager
from core.foreground_extractor import extract_segmented_crop
from models.reid_bnneck import ResNet50IBN_WithBNNeck

FONT = cv2.FONT_HERSHEY_DUPLEX

def draw_styled_label(img: np.ndarray, box: np.ndarray, label: str, color: tuple):
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


def run_offline_batch(video_paths: dict, out_dir: str):
    print("==================================================")
    print("  OFFLINE INFERENCE MODE (100% Frame Processing)  ")
    print("==================================================")
    print("Loading models...")
    
    tracker_engine = MultiCameraTrackerEngine(
        model_path="weights/yolo11s-seg.pt",
        device="cuda"
    )
    reid_extractor = ResNet50IBN_WithBNNeck(
        checkpoint_path="weights/resnet50_ibn_a.pth",
        device="cuda"
    )
    quality_gate = ClearFrameQualityGate()
    tracklet_mgr = TrackletManager()
    identity_mgr = GlobalIdentityManager(
        palette=PALETTE,
        match_threshold=MATCH_SIM_THRESHOLD,
        match_margin=MATCH_MARGIN,
        sticky_boost=STICKY_HYSTERESIS_BOOST,
    )

    # Open video captures and writers
    caps = {}
    writers = {}
    
    os.makedirs(out_dir, exist_ok=True)
    
    for cid, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Failed to open {path}")
            continue
        caps[cid] = cap
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(out_dir, f"annotated_offline_{cid}.mp4")
        writers[cid] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"[{cid}] Opened {path} ({w}x{h} @ {fps}fps)")

    frame_idx = 0
    start_time = time.time()
    
    print("\nStarting frame-by-frame analysis...")
    while caps:
        active_cids = list(caps.keys())
        frames = {}
        for cid in active_cids:
            ret, frame = caps[cid].read()
            if not ret or frame is None:
                caps[cid].release()
                writers[cid].release()
                del caps[cid]
                del writers[cid]
                print(f"[{cid}] Reached end of video.")
            else:
                frames[cid] = frame
        
        if not frames:
            break
            
        # 1. Batch detection
        cids_batch = list(frames.keys())
        frames_batch = [frames[c] for c in cids_batch]
        results_batch = tracker_engine.detect_batch(frames_batch)
        
        pending_reid_queries = []
        cam_detections = {}
        
        # Synthetic monotonic timestamp (assume 10 fps base for quality gate spacing)
        cap_ts = float(frame_idx) * 0.1 
        
        # 2. Tracking and Quality Gate
        for i, cid in enumerate(cids_batch):
            frame = frames[cid]
            res = results_batch[i]
            dets = tracker_engine.track_camera_frame(cid, frame, res)
            cam_detections[cid] = dets
            
            all_boxes = [d.box for d in dets]
            
            for det in dets:
                tracklet = tracklet_mgr.get_or_create(cid, det.track_id)
                if tracklet.status == "CONFIRMED":
                    continue
                
                q_res = quality_gate.evaluate(
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
                    seg_crop = extract_segmented_crop(frame, det.box, det.mask_bool, bg_fill=IMAGENET_MEAN_BGR)
                    if seg_crop.size > 0:
                        pending_reid_queries.append((cid, det.track_id, seg_crop, det.box, q_res, cap_ts))
                        
        # 3. Batch Re-ID
        if pending_reid_queries:
            crops_to_embed = [item[2] for item in pending_reid_queries]
            embeddings = reid_extractor.extract_batch(crops_to_embed)
            
            for j, (cid, tid, crop, box, q_res, cap_ts_item) in enumerate(pending_reid_queries):
                emb = embeddings[j]
                if emb is not None:
                    tracklet = tracklet_mgr.get_or_create(cid, tid)
                    tracklet.add_sample(emb, cap_ts_item, box, crop, q_res.confidence)
                    
        # 4. Matching and Rendering
        for cid in cids_batch:
            frame = frames[cid]
            dets = cam_detections.get(cid, [])
            
            tracks_for_matching = []
            occupied_gids = set()

            for det in dets:
                tracklet = tracklet_mgr.get_or_create(cid, det.track_id)
                if tracklet.status == "CONFIRMED":
                    if tracklet.assigned_gid:
                        gid = identity_mgr.resolve_gid(tracklet.assigned_gid)
                        occupied_gids.add(gid)
                        identity_mgr.update_presence(gid, cid, det.track_id, cap_ts)
                    continue
                    
                if tracklet.sample_count >= 3:
                    proto = tracklet.get_prototype_embedding()
                    if proto is not None:
                        tracks_for_matching.append((det.track_id, proto, det.box, tracklet.assigned_gid))
                        
            if tracks_for_matching:
                assignments = identity_mgr.match_camera_tracks(
                    cid,
                    tracks_for_matching,
                    cap_ts,
                    occupied_gids=occupied_gids,
                    is_enrollment_cam=True,
                    commit_exemplars=True,
                )
                for tid, (gid, score) in assignments.items():
                    tracklet = tracklet_mgr.get_or_create(cid, tid)
                    resolved_gid = identity_mgr.resolve_gid(gid)
                    tracklet.assigned_gid = resolved_gid
                    tracklet.similarity_score = score
                    tracklet.status = "CONFIRMED"

            # Rendering
            annotated = frame.copy()
            mask_overlay = annotated.copy()
            
            for det in dets:
                tracklet = tracklet_mgr.get_or_create(cid, det.track_id)
                gid = identity_mgr.resolve_gid(tracklet.assigned_gid) if tracklet.assigned_gid else None
                
                if gid and gid in identity_mgr.identities:
                    ident = identity_mgr.identities[gid]
                    color = ident.color
                elif tracklet.sample_count < 3:
                    color = COLOR_TENTATIVE
                else:
                    color = COLOR_UNKNOWN
                    
                mask_overlay[det.mask_bool] = (mask_overlay[det.mask_bool] * 0.65 + np.array(color) * 0.35).astype(np.uint8)
                cnts, _ = cv2.findContours(det.mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask_overlay, cnts, -1, tuple(int(c) for c in color), 1, cv2.LINE_AA)
                
            cv2.addWeighted(mask_overlay, 1.0, annotated, 0.0, 0, annotated)
            
            for det in dets:
                tracklet = tracklet_mgr.get_or_create(cid, det.track_id)
                gid = identity_mgr.resolve_gid(tracklet.assigned_gid) if tracklet.assigned_gid else None
                
                if gid and gid in identity_mgr.identities:
                    ident = identity_mgr.identities[gid]
                    label = f"[{ident.name}] Sim: {tracklet.similarity_score:.2f} | #{det.track_id}"
                    color = ident.color
                elif tracklet.sample_count < 3:
                    label = f"Buffering ({tracklet.sample_count}/3) | #{det.track_id}"
                    color = COLOR_TENTATIVE
                else:
                    label = f"UNKNOWN | #{det.track_id}"
                    color = COLOR_UNKNOWN
                    
                draw_styled_label(annotated, det.box, label, color)
                
            writers[cid].write(annotated)
            
        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed
            print(f"Processed {frame_idx} frames. Pipeline speed: {fps_proc:.2f} iterations/sec. Enrolled IDs: {len(identity_mgr.identities)}", flush=True)

    print(f"\nAll done! Output saved to {out_dir}")
    
    print("\n==================================================")
    print("  TRANSCODING FOR BROWSER COMPATIBILITY (H.264)   ")
    print("==================================================")
    for cid in video_paths.keys():
        out_path = os.path.join(out_dir, f"annotated_offline_{cid}.mp4")
        h264_path = os.path.join(out_dir, f"h264_annotated_offline_{cid}.mp4")
        if os.path.exists(out_path):
            print(f"Transcoding {cid}...")
            os.system(f"ffmpeg -y -loglevel warning -i {out_path} -vcodec libx264 -pix_fmt yuv420p {h264_path}")
            print(f"Created {h264_path}")
    print("All transcoding complete! Files are ready to view.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Inference for Multi-Camera Re-ID")
    parser.add_argument("--inputs", nargs="+", required=True, help="Format: cam_id=path/to/video.mp4")
    parser.add_argument("--out_dir", default=".", help="Output directory for annotated videos")
    args = parser.parse_args()
    
    video_paths = {}
    for inp in args.inputs:
        if "=" not in inp:
            print(f"Invalid input format: {inp}. Must be cam_id=path.mp4")
            exit(1)
        cid, path = inp.split("=", 1)
        video_paths[cid] = path
        
    run_offline_batch(video_paths, args.out_dir)
