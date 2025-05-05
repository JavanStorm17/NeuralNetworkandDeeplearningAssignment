#!/usr/bin/env python3
"""
Traffic monitoring dashboard for visualizing and controlling traffic
"""
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from pathlib import Path
from direct_traffic import TrafficController, VisionEncoder

class TrafficDashboard:
    """Interactive dashboard for traffic monitoring and control"""
    
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.root.title("Traffic Monitoring System")
        self.root.geometry("1200x800")
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.style.configure('TFrame', background='#2E3B4E')
        self.style.configure('TLabel', background='#2E3B4E', foreground='white')
        self.style.configure('TButton', background='#3498DB', foreground='white')
        
        self.setup_ui()
        
        # Traffic statistics
        self.stats = {
            'cars': [],
            'buses': [],
            'trucks': [],
            'wait_times': [],
            'actions': []
        }
        
        # Video sources
        self.video_running = False
        self.cap = None
        
        # Start simulation thread
        self.simulation_running = False
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - video feed and controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=10)
        
        # Video feed
        self.video_frame = ttk.Frame(top_frame, width=640, height=480, relief=tk.SUNKEN, borderwidth=2)
        self.video_frame.pack(side=tk.LEFT, padx=10)
        self.video_frame.pack_propagate(False)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls
        controls_frame = ttk.Frame(top_frame)
        controls_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(controls_frame, text="Operation Mode")
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.mode_var = tk.StringVar(value="simulation")
        ttk.Radiobutton(mode_frame, text="Simulation", variable=self.mode_var, 
                        value="simulation").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Live Camera", variable=self.mode_var, 
                        value="camera").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Video File", variable=self.mode_var, 
                        value="file").pack(side=tk.LEFT, padx=5)
        
        # Traffic light control
        light_frame = ttk.LabelFrame(controls_frame, text="Traffic Light Control")
        light_frame.pack(fill=tk.X, pady=5)
        
        self.light_state = tk.StringVar(value="Auto")
        ttk.Radiobutton(light_frame, text="Automatic (AI)", variable=self.light_state, 
                        value="Auto").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(light_frame, text="Manual", variable=self.light_state, 
                        value="Manual").pack(side=tk.LEFT, padx=5)
        
        # Manual control buttons
        manual_frame = ttk.Frame(light_frame)
        manual_frame.pack(pady=5, fill=tk.X)
        
        self.north_south_btn = ttk.Button(manual_frame, text="North-South Green", 
                                         command=lambda: self.manual_action(0))
        self.north_south_btn.pack(side=tk.LEFT, padx=5)
        
        self.east_west_btn = ttk.Button(manual_frame, text="East-West Green", 
                                       command=lambda: self.manual_action(1))
        self.east_west_btn.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(action_frame, text="Start", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(action_frame, text="Stop", command=self.stop_system)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Status
        status_frame = ttk.LabelFrame(controls_frame, text="System Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.StringVar(value="System Ready")
        ttk.Label(status_frame, textvariable=self.status_text, font=('Arial', 12)).pack(pady=5)
        
        # Bottom section - statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Traffic Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Vehicle counts
        counts_frame = ttk.Frame(stats_frame)
        counts_frame.pack(fill=tk.X, pady=5)
        
        # Create count labels
        self.car_count = tk.StringVar(value="Cars: 0")
        self.bus_count = tk.StringVar(value="Buses: 0")
        self.truck_count = tk.StringVar(value="Trucks: 0")
        self.total_count = tk.StringVar(value="Total Vehicles: 0")
        self.wait_time = tk.StringVar(value="Avg Wait Time: 0s")
        
        ttk.Label(counts_frame, textvariable=self.car_count, font=('Arial', 12)).pack(side=tk.LEFT, padx=20)
        ttk.Label(counts_frame, textvariable=self.bus_count, font=('Arial', 12)).pack(side=tk.LEFT, padx=20)
        ttk.Label(counts_frame, textvariable=self.truck_count, font=('Arial', 12)).pack(side=tk.LEFT, padx=20)
        ttk.Label(counts_frame, textvariable=self.total_count, font=('Arial', 12)).pack(side=tk.LEFT, padx=20)
        ttk.Label(counts_frame, textvariable=self.wait_time, font=('Arial', 12)).pack(side=tk.LEFT, padx=20)
        
        # Graphs frame
        self.graphs_frame = ttk.Frame(stats_frame)
        self.graphs_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Initial empty graph
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.tight_layout(pad=3.0)
        self.canvas_graph = self.embed_matplotlib(self.graphs_frame, self.fig)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def embed_matplotlib(self, parent, figure):
        """Embed matplotlib figure in tkinter"""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(figure, parent)
        canvas.draw()
        return canvas
    
    def update_video_feed(self):
        """Update video feed with detection overlay"""
        if not self.video_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return
        
        # Process frame with YOLO
        vehicle_state = self.controller.process_camera_feed(frame)
        
        # Update stats
        self.stats['cars'].append(vehicle_state[0].item())
        self.stats['buses'].append(vehicle_state[1].item())
        self.stats['trucks'].append(vehicle_state[2].item())
        
        # Update count labels
        self.car_count.set(f"Cars: {int(vehicle_state[0].item())}")
        self.bus_count.set(f"Buses: {int(vehicle_state[1].item())}")
        self.truck_count.set(f"Trucks: {int(vehicle_state[2].item())}")
        total = int(sum(vehicle_state[:4].tolist()))
        self.total_count.set(f"Total Vehicles: {total}")
        
        # Limit stats history
        max_history = 100
        if len(self.stats['cars']) > max_history:
            for key in self.stats:
                if len(self.stats[key]) > max_history:
                    self.stats[key] = self.stats[key][-max_history:]
        
        # Convert to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img = img.resize((640, 480), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep reference
        
        # Update statistics
        self.update_statistics()
        
        # Schedule next update
        self.root.after(50, self.update_video_feed)
    
    def update_statistics(self):
        """Update statistics graphs"""
        # Clear previous plots
        for ax in self.ax:
            ax.clear()
        
        # Vehicle count history
        if self.stats['cars']:
            x = range(len(self.stats['cars']))
            self.ax[0].plot(x, self.stats['cars'], label='Cars', color='blue')
            self.ax[0].plot(x, self.stats['buses'], label='Buses', color='green')
            self.ax[0].plot(x, self.stats['trucks'], label='Trucks', color='red')
            self.ax[0].set_title('Vehicle Count History')
            self.ax[0].set_xlabel('Time')
            self.ax[0].set_ylabel('Count')
            self.ax[0].legend()
        
        # Wait time history
        if self.stats['wait_times']:
            self.ax[1].plot(range(len(self.stats['wait_times'])), 
                          self.stats['wait_times'], color='orange')
            self.ax[1].set_title('Average Wait Time')
            self.ax[1].set_xlabel('Time')
            self.ax[1].set_ylabel('Wait Time (s)')
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas_graph.draw()
    
    def start_video(self, source=0):
        """Start video capture"""
        if self.video_running:
            self.stop_video()
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.status_text.set(f"Error: Could not open video source {source}")
            return False
        
        self.video_running = True
        self.update_video_feed()
        return True
    
    def stop_video(self):
        """Stop video capture"""
        self.video_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def start_system(self):
        """Start the traffic monitoring system"""
        mode = self.mode_var.get()
        
        if mode == "simulation":
            # Start simulation
            self.status_text.set("Starting simulation...")
            self.simulation_running = True
            threading.Thread(target=self.run_simulation, daemon=True).start()
            
        elif mode == "camera":
            # Start live camera feed
            if self.start_video(0):
                self.status_text.set("Live camera feed active")
            
        elif mode == "file":
            # For demo, use a sample video if available
            sample_video = Path(__file__).parent / "sample_traffic.mp4"
            if sample_video.exists():
                if self.start_video(str(sample_video)):
                    self.status_text.set("Video playback active")
            else:
                self.status_text.set("Sample video not found")
                return
        
        # Update button states
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
    
    def stop_system(self):
        """Stop the traffic monitoring system"""
        # Stop video capture
        self.stop_video()
        
        # Stop simulation
        self.simulation_running = False
        
        self.status_text.set("System stopped")
        
        # Update button states
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def run_simulation(self):
        """Run traffic simulation in a separate thread"""
        try:
            # Reset stats
            for key in self.stats:
                self.stats[key] = []
            
            # Get initial state from environment
            obs, _ = self.controller.env.reset()
            step = 0
            
            while self.simulation_running and step < 1000:
                # Get action from model or manual override
                if self.light_state.get() == "Auto":
                    action = self.controller.get_traffic_action(obs)
                else:
                    # Use last manual action
                    action = self.stats['actions'][-1] if self.stats['actions'] else 0
                
                # Apply action
                obs, reward, terminated, truncated, info = self.controller.env.step(action)
                
                # Update stats
                waiting_times = sum(info.get('step_waiting_time', {}).values())
                avg_wait = waiting_times / max(1, len(info.get('step_waiting_time', {})))
                
                self.stats['wait_times'].append(avg_wait)
                self.stats['actions'].append(action)
                
                # Generate synthetic vehicle counts for visualization
                # In a real system, these would come from actual detections
                cars = np.random.poisson(max(1, int(20 - avg_wait/2)))
                buses = np.random.poisson(max(1, int(5 - avg_wait/10)))
                trucks = np.random.poisson(max(1, int(3 - avg_wait/15)))
                
                self.stats['cars'].append(cars)
                self.stats['buses'].append(buses)
                self.stats['trucks'].append(trucks)
                
                # Update UI (tkinter updates must be in main thread)
                self.root.after(0, lambda: self.car_count.set(f"Cars: {cars}"))
                self.root.after(0, lambda: self.bus_count.set(f"Buses: {buses}"))
                self.root.after(0, lambda: self.truck_count.set(f"Trucks: {trucks}"))
                self.root.after(0, lambda: self.total_count.set(f"Total Vehicles: {cars+buses+trucks}"))
                self.root.after(0, lambda: self.wait_time.set(f"Avg Wait Time: {avg_wait:.1f}s"))
                self.root.after(0, lambda: self.status_text.set(f"Simulation running - Step {step}"))
                
                # Update statistics graph
                self.root.after(0, self.update_statistics)
                
                # Reset if needed
                if terminated or truncated:
                    obs, _ = self.controller.env.reset()
                
                step += 1
                time.sleep(0.1)  # Slow down simulation for visualization
            
            if not self.simulation_running:
                self.root.after(0, lambda: self.status_text.set("Simulation stopped"))
            else:
                self.root.after(0, lambda: self.status_text.set("Simulation completed"))
                self.simulation_running = False
                
            # Reset buttons
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_text.set(f"Error: {str(e)}"))
            self.simulation_running = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
    
    def manual_action(self, action):
        """Apply manual traffic light action"""
        if self.light_state.get() == "Manual":
            self.stats['actions'].append(action)
            self.status_text.set(f"Manual action applied: {action}")
        else:
            self.status_text.set("Switch to Manual mode first")

def main():
    """Main function"""
    # Check if required model paths exist
    root_path = Path(__file__).resolve().parent
    yolo_path = root_path/'runs/optimized_train/weights/best.pt'
    rl_path = root_path/'ppo_traffic_optimized.zip'
    
    # Use default YOLO model if trained one doesn't exist
    if not yolo_path.exists():
        yolo_path = 'yolov8n.pt'
        print(f"Using default YOLO model: {yolo_path}")
    
    # Use default or fallback RL model if trained one doesn't exist
    if not rl_path.exists():
        # This is just a placeholder, in real system you'd need a valid model
        print("RL model not found, using simulation only mode")
    
    # Initialize controller
    controller = TrafficController(
        rl_model_path=rl_path,
        vision_model_path=yolo_path
    )
    
    # Create and run dashboard
    root = tk.Tk()
    app = TrafficDashboard(root, controller)
    root.mainloop()

if __name__ == "__main__":
    main()