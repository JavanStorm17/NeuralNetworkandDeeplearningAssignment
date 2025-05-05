#!/bin/bash
# Fast training script for DETRAC and VisDrone datasets
# Completes within 2 hours

set -e  # Exit on error

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start timing
TOTAL_START_TIME=$(date +%s)

# Enable maximum GPU memory compact allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

echo "========================================================"
echo "🚀 Starting optimized training (2-hour time budget)"
echo "========================================================"

# Step 1: Train YOLOv8 on DETRAC and VisDrone datasets
echo "Step 1: Training YOLOv8 model (1 hour time budget)"
YOLO_START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python train_yolo_optimized.py

YOLO_END_TIME=$(date +%s)
YOLO_DURATION=$((YOLO_END_TIME - YOLO_START_TIME))
echo "YOLOv8 training completed in $((YOLO_DURATION / 60)) minutes and $((YOLO_DURATION % 60)) seconds"

# Calculate remaining time budget
ELAPSED_TIME=$(($(date +%s) - TOTAL_START_TIME))
REMAINING_BUDGET=$((7200 - ELAPSED_TIME))  # 7200 seconds = 2 hours

# Adjust RL timesteps based on remaining time
if [ $REMAINING_BUDGET -lt 1800 ]; then
    # Less than 30 minutes left - reduce timesteps
    export RL_TIMESTEPS=80000
    echo "Time budget tight, reducing RL training to $RL_TIMESTEPS timesteps"
elif [ $REMAINING_BUDGET -lt 3600 ]; then
    # Between 30-60 minutes left
    export RL_TIMESTEPS=150000
    echo "Using $RL_TIMESTEPS timesteps for RL training"
else
    # More than 60 minutes left - use default
    export RL_TIMESTEPS=200000
    echo "Using default $RL_TIMESTEPS timesteps for RL training"
fi

# Step 2: Train Reinforcement Learning model using the trained YOLO model
echo "Step 2: Training RL model with trained YOLO detector"
RL_START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python train_optimized.py

RL_END_TIME=$(date +%s)
RL_DURATION=$((RL_END_TIME - RL_START_TIME))
echo "RL training completed in $((RL_DURATION / 60)) minutes and $((RL_DURATION % 60)) seconds"

# Calculate total time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo "========================================================"
echo "✅ Complete training pipeline finished successfully!"
echo "Total time: $((TOTAL_DURATION / 60)) minutes and $((TOTAL_DURATION % 60)) seconds"
echo "========================================================"

# Make sure we're under 2 hours
if [ $TOTAL_DURATION -le 7200 ]; then
    echo "✅ Training completed within 2-hour budget"
else
    echo "⚠️ Training took longer than 2-hour budget"
    echo "   Consider further optimizations"
fi