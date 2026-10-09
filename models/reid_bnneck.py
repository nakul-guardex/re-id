#!/usr/bin/env python3
"""
ResNet50-IBN with BNNeck Feature Extractor (FastReID Architecture)
=================================================================
Loads pretrained weights from Market-1501 checkpoint (market_bot_R50-ibn.pth).
Produces 2048-dimensional L2-normalized feature vectors f_i = BatchNorm1d(f_t).
"""

import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    from torchreid.reid.models.resnet_ibn_a import resnet50_ibn_a
except ImportError:
    # If torchreid is not top-level, check repo path
    from torchreid.models.resnet_ibn_a import resnet50_ibn_a


def resolve_device(device_str: Optional[str] = None) -> torch.device:
    """Detect available accelerator: CUDA, Apple MPS, or CPU."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_checkpoint_path(custom_path: Optional[str] = None, download_url: Optional[str] = None) -> Path:
    """Find or download Market-1501 ResNet50-IBN weights."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    
    weights_dir = Path.home() / ".cache" / "torch" / "checkpoints"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = weights_dir / "market_bot_R50-ibn.pth"
    
    if not ckpt_path.exists():
        url = download_url or "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth"
        print(f"[Model] Downloading Market-1501 ResNet50-IBN checkpoint from {url}...")
        urllib.request.urlretrieve(url, ckpt_path)
        print(f"[Model] Checkpoint saved to {ckpt_path}")
    return ckpt_path


def load_state_dict_file(path: Path) -> dict:
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(path), map_location="cpu")
    return raw["model"] if isinstance(raw, dict) and "model" in raw else raw


def build_backbone(sd: dict, fastreid_arch: bool = True) -> torch.nn.Module:
    """Builds ResNet50-IBN-a matching FastReID's trained configuration."""
    model = resnet50_ibn_a(num_classes=1000, pretrained=False)

    if fastreid_arch:
        # FastReID: LAST_STRIDE = 1 -> layer4 keeps 16x8 resolution for 256x128 input
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        # FastReID: MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) with padding=0
        model.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

    expected = model.state_dict()
    ckpt_bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    matched = {k: v for k, v in ckpt_bb.items() if k in expected and expected[k].shape == v.shape}
    needed = [k for k in expected if not k.startswith(("fc.", "classifier."))]
    missing = [k for k in needed if k not in matched]

    if missing:
        raise RuntimeError(f"Backbone weights incomplete! Missing {len(missing)} tensors, e.g. {missing[:5]}")

    model.load_state_dict(matched, strict=False)
    return model


class ResNet50IBN_WithBNNeck:
    """
    Appearance Feature Extractor:
    Preprocesses 256x128 crops and returns 2048-d L2-normalized feature embeddings.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None, fastreid_arch: bool = True):
        self.device = resolve_device(device)
        self.fastreid_arch = fastreid_arch
        
        # Bicubic resize (FastReID standard), conversion to tensor, ImageNet normalization
        self.transform = T.Compose([
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        ckpt = get_checkpoint_path(checkpoint_path)
        sd = load_state_dict_file(ckpt)
        self.model = build_backbone(sd, fastreid_arch).eval().to(self.device)

        # Load BNNeck layer
        need = ["weight", "bias", "running_mean", "running_var"]
        missing = [f"heads.bnneck.{n}" for n in need if f"heads.bnneck.{n}" not in sd]
        if missing:
            raise RuntimeError(f"BNNeck weights not found in checkpoint: {missing}")

        self.bnneck = torch.nn.BatchNorm1d(2048, affine=True)
        self.bnneck.load_state_dict({
            "weight": sd["heads.bnneck.weight"],
            "bias": sd["heads.bnneck.bias"],
            "running_mean": sd["heads.bnneck.running_mean"],
            "running_var": sd["heads.bnneck.running_var"],
            "num_batches_tracked": sd.get("heads.bnneck.num_batches_tracked", torch.tensor(0)),
        })
        self.bnneck.eval().to(self.device)
        self._warned = False
        print(f"[ResNet50IBN_WithBNNeck] Initialized on {self.device} (2048-d, FastReID BNNeck).")

    def _warn_once(self, where: str, ex: Exception):
        if not self._warned:
            print(f"[ResNet50IBN] Extraction error in {where}: {type(ex).__name__}: {ex}")
            self._warned = True

    def extract_batch(self, bgr_crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Batched GPU inference:
        Takes a list of BGR numpy crops and returns a list of 2048-d L2-normalized float32 vectors.
        Returns None for empty or invalid crops.
        """
        out = [None] * len(bgr_crops)
        tensors, valid_idxs = [], []

        for i, c in enumerate(bgr_crops):
            if c is None or getattr(c, "size", 0) == 0 or c.shape[0] < 10 or c.shape[1] < 10:
                continue
            try:
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(Image.fromarray(rgb)))
                valid_idxs.append(i)
            except Exception as ex:
                self._warn_once("preprocess", ex)

        if not tensors:
            return out

        try:
            with torch.no_grad():
                batch = torch.stack(tensors).to(self.device)
                feat = self.model(batch)                      # Raw pooled feature (N, 2048)
                feat_bn = self.bnneck(feat).float().cpu().numpy() # BNNeck feature (N, 2048)
        except Exception as ex:
            self._warn_once("forward", ex)
            return out

        for j, i in enumerate(valid_idxs):
            v = feat_bn[j]
            norm = np.linalg.norm(v)
            out[i] = (v / (norm + 1e-12)).astype(np.float32)

        return out

    def extract(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """Single crop inference."""
        res = self.extract_batch([bgr_crop])
        return res[0]
