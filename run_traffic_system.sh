#!/bin/bash
# Script to run the traffic direction system

set -e  # Exit on error

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for required models
YOLO_MODEL="runs/optimized_train/weights/best.pt"
RL_MODEL="ppo_traffic_optimized.zip"

# Check Python environment
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8 available')"

# Display menu
echo "========================================================"
echo "🚦 Traffic Direction System"
echo "========================================================"
echo "Choose an option:"
echo "1. Run simulation mode"
echo "2. Run with webcam"
echo "3. Run with dashboard interface"
echo "4. Run training pipeline first"
echo "5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Starting traffic direction simulation..."
        python direct_traffic.py --mode simulation --steps 2000
        ;;
    2)
        echo "Starting traffic direction with webcam..."
        python direct_traffic.py --mode camera
        ;;
    3)
        echo "Starting traffic dashboard..."
        python traffic_dashboard.py
        ;;
    4)
        echo "Running training pipeline first..."
        ./train_fast.sh
        echo "Training completed. Now running traffic direction system..."
        python direct_traffic.py --mode simulation --steps 1000
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo "Done!"