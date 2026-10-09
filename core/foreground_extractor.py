#!/usr/bin/env python3
"""
Segmented Foreground Person Extraction
======================================
Crops the bounding box and masks out non-person background pixels using
the instance segmentation mask. Background pixels are replaced with the
neutral ImageNet BGR mean (103, 116, 124) before embedding extraction,
preventing office furniture, walls, and lighting shifts from leaking into Re-ID features.
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

# Neutral ImageNet BGR mean fill (OpenCV channel order):
# B: 0.406*255=103.5 -> 103, G: 0.456*255=116.3 -> 116, R: 0.485*255=123.7 -> 124
# After torchvision.transforms.Normalize, these values map to ~0.0 (zero contribution).
IMAGENET_MEAN_BGR = (103, 116, 124)


def extract_segmented_crop(
    frame_bgr: np.ndarray,
    box_xyxy: Union[np.ndarray, List[int], Tuple[int, int, int, int]],
    full_mask_bool: np.ndarray,
    bg_fill: Tuple[int, int, int] = IMAGENET_MEAN_BGR,
    erode_kernel_size: int = 0,
) -> np.ndarray:
    """
    Extracts a bounding box crop with background pixels masked out.
    
    Args:
        frame_bgr: Source image in BGR format.
        box_xyxy: Bounding box [x1, y1, x2, y2].
        full_mask_bool: Binary boolean mask corresponding to frame dimensions (H, W).
        bg_fill: BGR color used to fill background (default: ImageNet mean).
        erode_kernel_size: Optional kernel size to erode mask edges and remove background bleed.
        
    Returns:
        Segmented BGR crop numpy array, or empty array if coordinates invalid.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype)

    crop_img = frame_bgr[y1:y2, x1:x2].copy()
    crop_mask = full_mask_bool[y1:y2, x1:x2]

    if erode_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
        crop_mask_uint8 = (crop_mask.astype(np.uint8) * 255)
        crop_mask_uint8 = cv2.erode(crop_mask_uint8, kernel, iterations=1)
        crop_mask = crop_mask_uint8 > 127

    # Fill non-person background pixels with neutral mean
    crop_img[~crop_mask] = bg_fill

    return crop_img


def extract_batch_segmented_crops(
    frame_bgr: np.ndarray,
    boxes: List[Union[np.ndarray, List[int]]],
    masks: List[np.ndarray],
    bg_fill: Tuple[int, int, int] = IMAGENET_MEAN_BGR,
    erode_kernel_size: int = 0,
) -> List[np.ndarray]:
    """Batch extraction of segmented crops for all detections in a frame."""
    crops = []
    for b, m in zip(boxes, masks):
        crops.append(extract_segmented_crop(frame_bgr, b, m, bg_fill, erode_kernel_size))
    return crops
