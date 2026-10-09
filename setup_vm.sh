#!/usr/bin/env bash
# ==============================================================================
# Multi-Camera Live Re-ID System - One-Click VM Setup Script (NVIDIA CUDA)
# ==============================================================================
set -e

echo "=== [1/4] Installing System Dependencies & FFmpeg ==="
sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    curl \
    build-essential \
    python3-dev \
    python3-pip

echo "=== [2/4] Installing PyTorch with CUDA Support ==="
# Check if nvidia-smi exists
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Detected! Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected. Installing standard PyTorch..."
    pip install torch torchvision
fi

echo "=== [3/4] Installing Project Python Dependencies ==="
pip install -r requirements.txt

echo "=== [4/4] Pre-Downloading Model Weights ==="
mkdir -p weights
# Download YOLO11s-seg if not present
if [ ! -f "weights/yolo11s-seg.pt" ]; then
    echo "Downloading YOLO11s-Seg weights..."
    curl -L -o weights/yolo11s-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt
fi

# Download FastReID Market-1501 ResNet50-IBN weights
if [ ! -f "weights/market_bot_R50-ibn.pth" ]; then
    echo "Downloading Market-1501 ResNet50-IBN weights..."
    mkdir -p ~/.cache/torch/checkpoints
    curl -L -o ~/.cache/torch/checkpoints/market_bot_R50-ibn.pth https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth
    cp ~/.cache/torch/checkpoints/market_bot_R50-ibn.pth weights/market_bot_R50-ibn.pth
fi

echo "=============================================================================="
echo " Setup Complete! You can now start the web application by running:"
echo " python run.py"
echo "=============================================================================="
