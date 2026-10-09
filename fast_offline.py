import time
import os
import cv2
import numpy as np

from config import (
    COLOR_TENTATIVE, COLOR_UNKNOWN, IMAGENET_MEAN_BGR, PALETTE,
    MATCH_SIM_THRESHOLD, MATCH_MARGIN, MERGE_SIM_THRESHOLD, STICKY_HYSTERESIS_BOOST,
    REID_CHECKPOINT_PATH, YOLO_MODEL_PATH
)
from core.tracker_engine import MultiCameraTrackerEngine
from core.quality_gate import ClearFrameQualityGate
from core.tracklet_builder import TrackletManager
from core.identity_manager import GlobalIdentityManager
from core.foreground_extractor import extract_segmented_crop
from models.reid_bnneck import ResNet50IBN_WithBNNeck

FONT = cv2.FONT_HERSHEY_DUPLEX

def downscale(frame, width=960):
    h, w = frame.shape[:2]
    if w > width:
        scale = width / float(w)
        nh = int(round(h * scale))
        nw = width - (width % 2)
        nh = nh - (nh % 2)
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame

def draw_styled_label(img, box, label, color):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h_img, w_img = img.shape[:2]
    c_int = tuple(int(c) for c in color)
    cv2.rectangle(img, (x1, y1), (x2, y2), c_int, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.44, 1)
    top = max(0, y1 - th - 8)
    right = min(w_img, x1 + tw + 10)
    cv2.rectangle(img, (x1, top), (right, top + th + 8), c_int, -1)
    text_color = (0, 0, 0) if sum(c_int) > 380 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 5, top + th + 3), FONT, 0.44, text_color, 1, cv2.LINE_AA)

def extract_seed_embedding(video_path: str, timestamp_sec: float, tracker, reid_model) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return None
        
    frame = downscale(frame)
    res = tracker.detect_batch([frame])[0]
    
    if res.boxes is None or len(res.boxes) == 0:
        return None
        
    boxes = res.boxes.cpu().numpy()
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = boxes.xyxy[0]
    
    if masks is not None:
        mask_bool = cv2.resize(masks[0], (w_frame, h_frame), interpolation=cv2.INTER_LINEAR) > 0.5
    else:
        mask_bool = np.zeros((h_frame, w_frame), dtype=bool)
        mask_bool[int(y1):int(y2), int(x1):int(x2)] = True
        
    box = np.array([x1, y1, x2, y2], dtype=np.float32)
    crop = extract_segmented_crop(frame, box, mask_bool, bg_fill=IMAGENET_MEAN_BGR)
    return reid_model.extract_batch([crop])[0]

def run_fast_experiment():
    print("Loading models (YOLO & ResNet)...")
    tracker_engine = MultiCameraTrackerEngine(model_path=YOLO_MODEL_PATH, device="cuda")
    reid_extractor = ResNet50IBN_WithBNNeck(checkpoint_path=REID_CHECKPOINT_PATH, device="cuda")
    
    seed_video = "1min_recordings/office_4_20261008_135031.mp4"
    seed_emb = extract_seed_embedding(seed_video, 8.0, tracker_engine, reid_extractor)
    
    if seed_emb is None:
        print("Failed to get seed embedding.")
        return
        
    identity_mgr = GlobalIdentityManager(
        palette=PALETTE, match_threshold=MATCH_SIM_THRESHOLD,
        match_margin=MATCH_MARGIN, merge_threshold=MERGE_SIM_THRESHOLD,
        sticky_boost=STICKY_HYSTERESIS_BOOST
    )
    
    target_gid = identity_mgr._create_identity(seed_emb)
    identity_mgr.identities[target_gid].name = "TARGET PERSON"
    identity_mgr.identities[target_gid].color = (0, 0, 255)
    print(f"🎯 Enrolled target identity: {target_gid}")
    
    video_paths = {
        "office_1": "1min_recordings/office_1_20261008_135031.mp4",
        "office_2": "1min_recordings/office_2_20261008_135031.mp4",
        "office_3": "1min_recordings/office_3_20261008_135031.mp4"
    }
    
    out_dir = "fast_experiment_results"
    os.makedirs(out_dir, exist_ok=True)
    quality_gate = ClearFrameQualityGate()
    tracklet_mgr = TrackletManager()
    
    caps, writers = {}, {}
    for cid, path in video_paths.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): continue
        caps[cid] = cap
        
        # Read a frame to get the downscaled resolution for the writer
        ret, test_frame = cap.read()
        if not ret: continue
        test_frame = downscale(test_frame)
        h, w = test_frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Rewind
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[cid] = cv2.VideoWriter(os.path.join(out_dir, f"{cid}_tracked.mp4"), fourcc, fps, (w, h))
        
    print("\nStarting batched inference...")
    frame_idx, start_time = 0, time.time()
    
    # Still skipping 1 frame, but with downscaled images it should be blazingly fast
    FRAME_SKIP = 2
    
    while caps:
        frames = {}
        for cid in list(caps.keys()):
            ret, frame = caps[cid].read()
            if not ret or frame is None:
                caps[cid].release()
                writers[cid].release()
                del caps[cid], writers[cid]
            else:
                frames[cid] = downscale(frame)
                
        if not frames: break
        frame_idx += 1
        
        if frame_idx % FRAME_SKIP != 0:
            for cid, frame in frames.items():
                writers[cid].write(frame)
            continue
            
        cids_batch = list(frames.keys())
        frames_batch = [frames[c] for c in cids_batch]
        results_batch = tracker_engine.detect_batch(frames_batch)
        
        pending_reid_queries = []
        cam_detections = {}
        cap_ts = float(frame_idx) * 0.1 
        
        for i, cid in enumerate(cids_batch):
            frame = frames[cid]
            dets = tracker_engine.track_camera_frame(cid, frame, results_batch[i])
            cam_detections[cid] = dets
            all_boxes = [d.box for d in dets]
            for det in dets:
                tracklet = tracklet_mgr.get_or_create(cid, det.track_id)
                if tracklet.status == "CONFIRMED": continue
                
                q_res = quality_gate.evaluate(cid, det.track_id, det.box, det.mask_bool, det.conf, frame.shape[:2], frame, [b for b in all_boxes if not np.array_equal(b, det.box)], cap_ts)
                if q_res.passed:
                    seg_crop = extract_segmented_crop(frame, det.box, det.mask_bool, bg_fill=IMAGENET_MEAN_BGR)
                    if seg_crop.size > 0:
                        pending_reid_queries.append((cid, det.track_id, seg_crop, det.box, q_res, cap_ts))
                        
        if pending_reid_queries:
            crops_to_embed = [item[2] for item in pending_reid_queries]
            embeddings = reid_extractor.extract_batch(crops_to_embed)
            for j, (cid, tid, crop, box, q_res, cap_ts_item) in enumerate(pending_reid_queries):
                if embeddings[j] is not None:
                    tracklet_mgr.get_or_create(cid, tid).add_sample(embeddings[j], cap_ts_item, box, crop, q_res.confidence)
                    
        for cid in cids_batch:
            frame = frames[cid]
            dets = cam_detections.get(cid, [])
            tracks_for_matching = []
            frame_assigned_gids = []
            
            for det in dets:
                tracklet = tracklet_mgr.get_or_create(cid, det.track_id)
                if tracklet.status == "CONFIRMED":
                    if tracklet.assigned_gid:
                        gid = identity_mgr.resolve_gid(tracklet.assigned_gid)
                        frame_assigned_gids.append(gid)
                        if gid in identity_mgr.identities:
                            identity_mgr.identities[gid].active_presence[cid] = (det.track_id, cap_ts)
                    continue
                if tracklet.sample_count >= 3:
                    proto = tracklet.get_prototype_embedding()
                    if proto is not None:
                        tracks_for_matching.append((det.track_id, proto, det.box, tracklet.assigned_gid))
                        
            if tracks_for_matching:
                assignments = identity_mgr.match_camera_tracks(cid, tracks_for_matching, cap_ts)
                for tid, (gid, score) in assignments.items():
                    tracklet = tracklet_mgr.get_or_create(cid, tid)
                    resolved_gid = identity_mgr.resolve_gid(gid)
                    tracklet.assigned_gid = resolved_gid
                    tracklet.similarity_score = score
                    tracklet.status = "CONFIRMED"
                    frame_assigned_gids.append(resolved_gid)
                    
            if len(frame_assigned_gids) >= 2:
                identity_mgr.record_co_occurrence(cid, frame_assigned_gids)
                
            if frame_idx % 30 == 0:
                identity_mgr.run_reconciliation_cycle()
                
            annotated = frame.copy()
            mask_overlay = annotated.copy()
            for det in dets:
                tracklet = tracklet_mgr.get_or_create(cid, det.track_id)
                gid = identity_mgr.resolve_gid(tracklet.assigned_gid) if tracklet.assigned_gid else None
                color = identity_mgr.identities[gid].color if gid and gid in identity_mgr.identities else (COLOR_TENTATIVE if tracklet.sample_count < 3 else COLOR_UNKNOWN)
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
                else:
                    label = f"Buffering ({tracklet.sample_count}/3)" if tracklet.sample_count < 3 else "UNKNOWN"
                    color = COLOR_TENTATIVE if tracklet.sample_count < 3 else COLOR_UNKNOWN
                draw_styled_label(annotated, det.box, label, color)
                
            writers[cid].write(annotated)
            
        if frame_idx % 100 == 0:
            fps_proc = frame_idx / (time.time() - start_time)
            print(f"Processed {frame_idx} frames. Speed: {fps_proc:.2f} fps. Active IDs: {len(identity_mgr.identities)}", flush=True)

    print("Experiment finished!")

if __name__ == '__main__':
    run_fast_experiment()
