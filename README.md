# Neural Network and Deep Learning Assignment

A traffic control system using YOLOv8 for object detection and reinforcement learning for traffic optimization.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd NeuralNetworkandDeeplearningAssignment
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the interactive script to start the traffic system:

```
./run_traffic_system.sh
```

### Available Options:

1. **Run simulation mode** - Run the traffic direction system in simulation mode
2. **Run with webcam** - Run the traffic system using a connected webcam
3. **Run with dashboard interface** - Start the visual dashboard for traffic monitoring
4. **Run training pipeline first** - Train the models before running the system
5. **Exit** - Exit the program

## Training

To train the models from scratch:

```
./train_fast.sh
```

This script runs an optimized training pipeline that:
1. Trains YOLOv8 for object detection
2. Trains the reinforcement learning model for traffic control
3. Completes within a 2-hour time budget

## File Structure

- `direct_traffic.py` - Main traffic direction system
- `traffic_dashboard.py` - Visual dashboard for traffic monitoring
- `train_yolo_optimized.py` - YOLOv8 training script
- `train_optimized.py` - RL model training script
- `train_fast.sh` - Complete training pipeline
- `run_traffic_system.sh` - User-friendly interface to run the system