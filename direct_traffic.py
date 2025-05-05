#!/usr/bin/env python3
"""
Traffic direction system using trained YOLO and RL models
Uses vision detection to optimize traffic signal timings
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment
from compat import OldGymToGymnasium, DictObsToArray, ScalarActionToDict, DictRewardToFloat

# Configuration
ROOT = Path(__file__).resolve().parent
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL_PATH = ROOT/'runs/optimized_train/weights/best.pt'
RL_MODEL_PATH = ROOT/'ppo_traffic_optimized.zip'
NET_FILE = ROOT/'sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml'
ROUTE_FILE = ROOT/'sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml'
USE_GUI = True  # Set to True to visualize SUMO simulation

class VisionEncoder:
    """Vision encoder using YOLOv8 model to detect vehicles"""
    def __init__(self, model_path=YOLO_MODEL_PATH, conf=0.25, device=GPU_DEVICE):
        print(f"📷 Loading YOLOv8 model from {model_path}")
        self.det = YOLO(model_path).to(device)
        self.det.fuse()  # Optimize model
        self.device = device
        self.conf = conf
        self.cache = {}
        self.max_cache_size = 100

    @torch.no_grad()
    def __call__(self, frame):
        """Process frame to extract vehicle information"""
        # Use frame hash for caching
        frame_hash = hash(frame.tobytes()) if hasattr(frame, 'tobytes') else hash(str(frame))
        
        if frame_hash in self.cache:
            return self.cache[frame_hash]
        
        # Process frame with YOLO
        out = self.det.predict(frame, conf=self.conf, verbose=False, device=self.device)[0]
        boxes, cls, speed = out.boxes.xyxy, out.boxes.cls, out.boxes.conf
        
        # Extract vehicle counts by class
        cars = (cls == 2).sum()    # YOLO label 2 = car
        buses = (cls == 5).sum()   # bus
        trucks = (cls == 7).sum()  # truck
        vans = (cls == 5).sum()    # van (using same as bus for simplicity)
        mean_speed = speed.mean().item() if speed.numel() else 0.0
        
        # Create state vector
        state = torch.tensor([cars, buses, trucks, vans, mean_speed], 
                           dtype=torch.float32, device=self.device)
        
        # Cache the result
        if len(self.cache) >= self.max_cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[frame_hash] = state
        
        return state

class TrafficController:
    """Traffic light controller using trained RL model and vision detection"""
    def __init__(self, rl_model_path=RL_MODEL_PATH, vision_model_path=YOLO_MODEL_PATH):
        # Load vision encoder
        self.vision_encoder = VisionEncoder(model_path=vision_model_path)
        
        # Load SUMO environment
        self.env = self._create_env()
        
        # Load trained RL model
        print(f"🧠 Loading RL model from {rl_model_path}")
        self.model = PPO.load(rl_model_path, device=GPU_DEVICE)
        
        print("✅ Traffic controller initialized successfully")

    def _create_env(self):
        """Create SUMO environment with wrappers"""
        raw_env = SumoEnvironment(
            net_file=str(NET_FILE),
            route_file=str(ROUTE_FILE),
            use_gui=USE_GUI,
            num_seconds=3600,
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
    
    def process_camera_feed(self, frame):
        """Process camera frame and get vehicle information"""
        return self.vision_encoder(frame)
    
    def get_traffic_action(self, state):
        """Get traffic light action from RL model based on state"""
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def run_simulation(self, num_steps=1000):
        """Run traffic simulation for specified number of steps"""
        print(f"🚦 Starting traffic control simulation for {num_steps} steps")
        obs, _ = self.env.reset()
        total_reward = 0
        
        for step in range(num_steps):
            # Get action from RL model
            action = self.get_traffic_action(obs)
            
            # Apply action to environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Print info every 100 steps
            if step % 100 == 0:
                print(f"Step {step}/{num_steps}, Total Reward: {total_reward:.2f}")
            
            if terminated or truncated:
                print("Environment terminated, resetting...")
                obs, _ = self.env.reset()
        
        print(f"✅ Simulation completed, Final Reward: {total_reward:.2f}")
        self.env.close()
        return total_reward
    
    def process_live_video(self, video_path=0):
        """Process live video feed and control traffic in real-time
        video_path: 0 for webcam, or path to video file
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            return
        
        print("📹 Starting live video processing (press 'q' to quit)")
        
        # Reset environment
        obs, _ = self.env.reset()
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with YOLO
            vehicle_state = self.process_camera_feed(frame)
            
            # Combine with SUMO state (this is a simplified example)
            # In a real system, you'd need to map camera view to SUMO state
            combined_state = obs  # Replace with actual fusion logic
            
            # Get action from RL model
            action = self.get_traffic_action(combined_state)
            
            # Apply action to environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Display frame with annotations
            cv2.putText(frame, f"Cars: {vehicle_state[0]:.0f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Action: {action}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Traffic Control', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Reset if necessary
            if terminated or truncated:
                obs, _ = self.env.reset()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        self.env.close()

def main():
    """Main function to run traffic controller"""
    # Check for model files
    rl_model_file = Path(RL_MODEL_PATH)
    yolo_model_file = Path(YOLO_MODEL_PATH)
    
    if not rl_model_file.exists():
        print(f"⚠️ RL model not found at {rl_model_file}")
        print("Run train_optimized.py first to generate the model.")
        return
    
    if not yolo_model_file.exists():
        print(f"⚠️ YOLO model not found at {yolo_model_file}")
        print("Run train_yolo_optimized.py first or specify a different model path.")
        return
    
    # Initialize controller
    controller = TrafficController(
        rl_model_path=rl_model_file,
        vision_model_path=yolo_model_file
    )
    
    # Choose mode of operation
    import argparse
    parser = argparse.ArgumentParser(description='Traffic Direction System')
    parser.add_argument('--mode', choices=['simulation', 'live', 'camera'], 
                        default='simulation', help='Mode of operation')
    parser.add_argument('--steps', type=int, default=1000, 
                        help='Number of steps for simulation')
    parser.add_argument('--video', type=str, default="0", 
                        help='Video source (0 for webcam, or path to file)')
    args = parser.parse_args()
    
    if args.mode == 'simulation':
        controller.run_simulation(num_steps=args.steps)
    elif args.mode in ['live', 'camera']:
        video_source = 0 if args.video == "0" else args.video
        controller.process_live_video(video_path=video_source)

if __name__ == "__main__":
    main()