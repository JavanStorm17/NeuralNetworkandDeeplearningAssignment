#!/usr/bin/env python3
"""
Optimized training script for reinforcement learning using DETRAC and VisDrone datasets
Designed to complete within 2 hours
"""
import os
import torch
import multiprocessing as mp
from pathlib import Path
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from compat import OldGymToGymnasium, DictObsToArray, ScalarActionToDict, DictRewardToFloat
from ultralytics import YOLO

# Configuration
ROOT = Path(__file__).resolve().parent
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ENVS = max(1, min(8, mp.cpu_count() - 1))  # Use all cores except one for system
TIMESTEPS = 200_000  # Reduced from 500k or 1M
BATCH_SIZE = 1024  # Larger batch size for better GPU utilization
NET_FILE = ROOT/'sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml'
ROUTE_FILE = ROOT/'sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml'
YOLO_MODEL_PATH = ROOT/'yolov8n.pt'  # Use nano model for speed

print(f"🚗 Optimized RL training with {NUM_ENVS} parallel environments on {GPU_DEVICE}")
print(f"💿 Using YOLOv8 model: {YOLO_MODEL_PATH}")

class VisionEncoder:
    """Efficient vision encoder using YOLOv8 with caching"""
    def __init__(self, model_path=YOLO_MODEL_PATH, conf=0.25, device=GPU_DEVICE):
        print(f"📷 Loading YOLOv8 model from {model_path}")
        self.det = YOLO(model_path).to(device)
        self.det.fuse()  # Optimize model
        self.device = device
        self.conf = conf
        self.cache = {}  # Simple frame cache to avoid redundant processing
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000

    @torch.no_grad()
    def __call__(self, frame):
        """Return fixed-length state vector with caching for efficiency"""
        # Use frame hash for caching
        frame_hash = hash(frame.tobytes()) if hasattr(frame, 'tobytes') else hash(str(frame))
        
        if frame_hash in self.cache:
            self.cache_hits += 1
            return self.cache[frame_hash]
        
        # Process frame with YOLO
        out = self.det.predict(frame, conf=self.conf, verbose=False, device=self.device)[0]
        boxes, cls, speed = out.boxes.xyxy, out.boxes.cls, out.boxes.conf
        
        # Extract vehicle counts
        cars = (cls == 2).sum()    # YOLO label 2 = car
        buses = (cls == 5).sum()   # bus
        trucks = (cls == 7).sum()  # truck
        vans = (cls == 5).sum()    # van (using same as bus for simplicity)
        mean_speed = speed.mean().item() if speed.numel() else 0.0
        
        # Create state vector
        state = torch.tensor([cars, buses, trucks, vans, mean_speed], 
                           dtype=torch.float32, device=self.device)
        
        # Cache the result
        self.cache_misses += 1
        if len(self.cache) >= self.max_cache_size:
            # Remove random item if cache full
            self.cache.pop(next(iter(self.cache)))
        self.cache[frame_hash] = state
        
        return state

def make_env():
    """Create a single environment instance with vision encoder"""
    raw_env = SumoEnvironment(
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILE),
        use_gui=False,
        num_seconds=3600,  # Shorter episodes for faster training
        delta_time=5,
        yellow_time=3,
        min_green=8,
        reward_fn="diff-waiting-time",
        additional_sumo_cmd=['--device.rerouting.probability', '0.2']
    )
    
    # Apply wrappers
    env = OldGymToGymnasium(raw_env)
    env = DictObsToArray(env)
    env = ScalarActionToDict(env)
    env = DictRewardToFloat(env)
    
    return env

def main():
    """Main training function with optimized settings"""
    # For reproducibility
    torch.manual_seed(42)
    
    # Create vectorized environment
    print(f"🔄 Creating {NUM_ENVS} parallel environments")
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    
    # Create PPO model with optimized hyperparameters
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,  # ReLU faster than Tanh
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Smaller network
    )
    
    print("🧠 Creating PPO model with optimized hyperparameters")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=BATCH_SIZE,
        learning_rate=3e-4,  # Slightly higher learning rate
        gamma=0.99,
        ent_coef=0.01,  # Encourage exploration
        clip_range=0.2,
        n_epochs=10,
        device=GPU_DEVICE,
        verbose=1
    )
    
    # Train the model
    print(f"🏋️ Training for {TIMESTEPS} timesteps")
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
        log_interval=10  # More frequent logging
    )
    
    # Save the model
    model_path = "ppo_traffic_optimized"
    model.save(model_path)
    print(f"✅ RL training finished → {model_path}.zip")
    env.close()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Robust across OSes
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    if torch.cuda.is_available():
        # Track GPU time
        start_time.record()
    
    main()
    
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        print(f"⏱️ Training time: {start_time.elapsed_time(end_time) / 1000:.2f} seconds")
    
    print("Training completed successfully!")