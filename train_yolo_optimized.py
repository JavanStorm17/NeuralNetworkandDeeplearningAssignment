#!/usr/bin/env python3
"""
Optimized YOLO training on DETRAC and VisDrone datasets
Designed to complete within 2 hours
"""
import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

# Configuration
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT/'traffic.yaml'
OUTPUT_DIR = ROOT/'runs'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Adjust based on GPU memory
IMAGE_SIZE = 640  # Reduced from standard 1280 for faster training
EPOCHS = 30  # Reduced for faster training
PATIENCE = 10  # Early stopping patience
WORKERS = min(8, os.cpu_count() or 1)

# Load config
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

def train_yolo():
    """Train YOLOv8 model on DETRAC and VisDrone datasets"""
    print(f"🚀 Starting optimized YOLOv8 training on {DEVICE}")
    print(f"📊 Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    # Start with a pre-trained nano model for faster convergence
    model = YOLO('yolov8n.pt')
    
    # Train with optimized parameters
    results = model.train(
        data=CONFIG_FILE,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        patience=PATIENCE,
        project=str(OUTPUT_DIR),
        name='optimized_train',
        amp=True,  # Mixed precision for faster training
        close_mosaic=10,  # Disable mosaic augmentation in final epochs
        cos_lr=True,  # Cosine learning rate scheduler
        lr0=0.01,  # Higher initial learning rate
        lrf=0.01,  # Final learning rate
        optimizer='AdamW',  # Better optimizer
        dropout=0.0,  # Disable dropout for faster training
        seed=42,  # For reproducibility
        verbose=True,
        exist_ok=True
    )
    
    # Export to optimized format
    model.export(format='onnx')
    print(f"✅ YOLO training completed → {model.export()}")
    
    return model

def validate_model(model):
    """Validate the trained model"""
    print("🔍 Validating model on validation set")
    metrics = model.val(data=CONFIG_FILE)
    print(f"📈 Validation metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Track training time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    if torch.cuda.is_available():
        start_time.record()
    
    # Train model
    model = train_yolo()
    
    # Validate model
    metrics = validate_model(model)
    
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        print(f"⏱️ Total time: {start_time.elapsed_time(end_time) / 1000:.2f} seconds")