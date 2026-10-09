import cv2
import numpy as np
from core.tracker_engine import MultiCameraTrackerEngine
from core.foreground_extractor import extract_segmented_crop
from config import IMAGENET_MEAN_BGR

video_path = "1min_recordings/office_4_20261008_135031.mp4"
timestamp_sec = 8.0
out_img = "/home/ubuntu/.gemini/antigravity-ide/brain/64708744-6a0a-4ebc-8034-2654458ac0ab/target_crop.jpg"

print("Loading YOLO for crop extraction...")
tracker = MultiCameraTrackerEngine(model_path="weights/yolo11s-seg.pt", device="cuda")

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
ret, frame = cap.read()
cap.release()

if ret:
    res = tracker.detect_batch([frame])[0]
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
    
    cv2.imwrite(out_img, crop)
    print(f"✅ Saved crop to {out_img}")
else:
    print("❌ Failed to read frame.")
