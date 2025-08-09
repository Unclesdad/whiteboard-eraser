#!/usr/bin/env python3
"""
Raspberry Pi Camera Module 3 Whiteboard Edge Detector
Continuously captures images and detects whiteboard edges, logging results to CSV
"""

import cv2
import numpy as np
import time
import csv
import os
import threading
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, List, Tuple
import signal
import sys

# Import our optimized detector
from whiteboard_tracker4 import WhiteboardDetector

class CameraWhiteboardDetector:
    def __init__(self, 
                 camera_id: int = 0,
                 capture_interval: float = 2.0,
                 detection_interval: float = 5.0,
                 csv_filename: str = "whiteboard_edges.csv",
                 image_width: int = 640,
                 image_height: int = 480,
                 save_images: bool = False):
        """
        Initialize camera detector
        
        Args:
            camera_id: Camera device ID (usually 0 for Pi Camera)
            capture_interval: Seconds between image captures
            detection_interval: Seconds between edge detections
            csv_filename: CSV file to log results
            image_width: Capture width (lower = faster processing)
            image_height: Capture height (lower = faster processing)
            save_images: Whether to save processed images
        """
        self.camera_id = camera_id
        self.capture_interval = capture_interval
        self.detection_interval = detection_interval
        self.csv_filename = csv_filename
        self.image_width = image_width
        self.image_height = image_height
        self.save_images = save_images
        
        # Initialize detector
        self.detector = WhiteboardDetector(debug=False)
        
        # Threading components
        self.image_queue = Queue(maxsize=5)  # Small buffer to prevent memory issues
        self.running = False
        self.capture_thread = None
        self.detection_thread = None
        
        # Camera
        self.camera = None
        
        # Statistics
        self.stats = {
            'images_captured': 0,
            'images_processed': 0,
            'edges_detected': 0,
            'start_time': time.time()
        }
        
        # Initialize CSV file
        self._init_csv()
        
    def _init_csv(self):
        """Initialize CSV file with headers"""
        file_exists = os.path.exists(self.csv_filename)
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow([
                    'timestamp',
                    'iso_timestamp', 
                    'edges_found',
                    'edge_1_rho',
                    'edge_1_theta', 
                    'edge_1_angle_deg',
                    'edge_2_rho',
                    'edge_2_theta',
                    'edge_2_angle_deg',
                    'processing_time_ms'
                ])
        
        print(f"CSV logging initialized: {self.csv_filename}")
    
    def _init_camera(self) -> bool:
        """Initialize camera with Pi-optimized settings"""
        try:
            # Try to initialize camera
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties for Pi performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            self.camera.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS for stability
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            
            # Test capture
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("Error: Could not capture test frame")
                return False
            
            print(f"Camera initialized: {test_frame.shape[1]}x{test_frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def _capture_loop(self):
        """Continuous image capture thread"""
        print("Starting capture loop...")
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    timestamp = time.time()
                    
                    # Try to add to queue (non-blocking)
                    try:
                        self.image_queue.put((frame.copy(), timestamp), block=False)
                        self.stats['images_captured'] += 1
                    except:
                        # Queue full, skip this frame
                        pass
                else:
                    print("Warning: Failed to capture frame")
                    time.sleep(0.1)  # Brief pause on capture failure
                
                time.sleep(self.capture_interval)
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(1)  # Longer pause on error
    
    def _detection_loop(self):
        """Edge detection processing thread"""
        print("Starting detection loop...")
        
        last_detection_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for detection
                if current_time - last_detection_time < self.detection_interval:
                    time.sleep(0.1)
                    continue
                
                # Get latest image from queue
                frame = None
                capture_timestamp = None
                
                # Get the most recent frame, discard older ones
                while True:
                    try:
                        frame, capture_timestamp = self.image_queue.get(timeout=0.1)
                    except Empty:
                        break
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process the frame
                start_time = time.time()
                edges = self.detector.detect_whiteboard_edges(frame)
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                self.stats['images_processed'] += 1
                
                # Log results
                self._log_results(capture_timestamp, edges, processing_time)
                
                # Save image if requested
                if self.save_images and edges:
                    self._save_processed_image(frame, edges, capture_timestamp)
                
                # Update stats
                if edges and len(edges) > 0:
                    self.stats['edges_detected'] += 1
                
                last_detection_time = current_time
                
                # Print status
                if self.stats['images_processed'] % 10 == 0:
                    self._print_status()
                
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(1)
    
    def _log_results(self, timestamp: float, edges: Optional[List[Tuple[float, float]]], processing_time: float):
        """Log detection results to CSV"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            iso_timestamp = dt.isoformat()
            
            # Prepare row data
            row = [
                timestamp,
                iso_timestamp,
                len(edges) if edges else 0,
                '', '', '',  # Edge 1 data
                '', '', '',  # Edge 2 data
                f"{processing_time:.2f}"
            ]
            
            # Fill in edge data if available
            if edges:
                for i, (rho, theta) in enumerate(edges[:2]):  # Max 2 edges
                    angle_deg = np.degrees(theta)
                    start_idx = 3 + (i * 3)
                    row[start_idx] = f"{rho:.2f}"
                    row[start_idx + 1] = f"{theta:.4f}"
                    row[start_idx + 2] = f"{angle_deg:.2f}"
            
            # Write to CSV
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
            
            # Print detection result
            if edges:
                print(f"[{dt.strftime('%H:%M:%S')}] Detected {len(edges)} edge(s) in {processing_time:.1f}ms")
                for i, (rho, theta) in enumerate(edges):
                    angle = np.degrees(theta)
                    edge_type = "horizontal" if abs(angle) < 45 or abs(angle) > 135 else "vertical"
                    print(f"  Edge {i+1}: {edge_type}, rho={rho:.1f}, angle={angle:.1f}°")
            else:
                print(f"[{dt.strftime('%H:%M:%S')}] No edges detected ({processing_time:.1f}ms)")
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def _save_processed_image(self, frame: np.ndarray, edges: List[Tuple[float, float]], timestamp: float):
        """Save processed image with detected edges"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            filename = f"whiteboard_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # Draw edges on image
            result_img = frame.copy()
            h, w = frame.shape[:2]
            colors = [(0, 0, 255), (0, 255, 0)]  # Red, Green
            
            for i, (rho, theta) in enumerate(edges):
                color = colors[i % len(colors)]
                
                # Calculate line endpoints
                a, b = np.cos(theta), np.sin(theta)
                
                # Find intersections with image boundaries
                intersections = []
                
                # Check edges
                if abs(a) > 0.001:
                    y_left = rho / b if abs(b) > 0.001 else None
                    if y_left is not None and 0 <= y_left <= h:
                        intersections.append((0, int(y_left)))
                    
                    y_right = (rho - w * a) / b if abs(b) > 0.001 else None
                    if y_right is not None and 0 <= y_right <= h:
                        intersections.append((w, int(y_right)))
                
                if abs(b) > 0.001:
                    x_top = rho / a if abs(a) > 0.001 else None
                    if x_top is not None and 0 <= x_top <= w:
                        intersections.append((int(x_top), 0))
                    
                    x_bottom = (rho - h * b) / a if abs(a) > 0.001 else None
                    if x_bottom is not None and 0 <= x_bottom <= w:
                        intersections.append((int(x_bottom), h))
                
                # Draw line if we have intersections
                if len(intersections) >= 2:
                    cv2.line(result_img, intersections[0], intersections[1], color, 2)
            
            cv2.imwrite(filename, result_img)
            print(f"  Saved: {filename}")
            
        except Exception as e:
            print(f"Image save error: {e}")
    
    def _print_status(self):
        """Print current status"""
        runtime = time.time() - self.stats['start_time']
        capture_rate = self.stats['images_captured'] / runtime if runtime > 0 else 0
        detection_rate = self.stats['images_processed'] / runtime if runtime > 0 else 0
        
        print(f"\n=== STATUS ===")
        print(f"Runtime: {runtime:.1f}s")
        print(f"Images captured: {self.stats['images_captured']} ({capture_rate:.2f}/s)")
        print(f"Images processed: {self.stats['images_processed']} ({detection_rate:.2f}/s)")
        print(f"Edges detected: {self.stats['edges_detected']}")
        print(f"Queue size: {self.image_queue.qsize()}")
        print("==============\n")
    
    def start(self):
        """Start the camera detector"""
        print("Starting Raspberry Pi Whiteboard Edge Detector...")
        
        # Initialize camera
        if not self._init_camera():
            print("Failed to initialize camera")
            return False
        
        # Set running flag
        self.running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        
        self.capture_thread.start()
        self.detection_thread.start()
        
        print(f"Camera detector started!")
        print(f"Capture interval: {self.capture_interval}s")
        print(f"Detection interval: {self.detection_interval}s") 
        print(f"CSV log: {self.csv_filename}")
        print(f"Image size: {self.image_width}x{self.image_height}")
        print(f"Save images: {self.save_images}")
        print("Press Ctrl+C to stop...\n")
        
        return True
    
    def stop(self):
        """Stop the camera detector"""
        print("\nStopping camera detector...")
        
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        self._print_status()
        print("Camera detector stopped.")
    
    def run_forever(self):
        """Run until interrupted"""
        if not self.start():
            return
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received...")
        finally:
            self.stop()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutdown signal received...")
    sys.exit(0)

def main():
    """Main function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Configuration
    config = {
        'camera_id': 0,              # Pi Camera
        'capture_interval': 1.0,     # Capture every 1 second
        'detection_interval': 3.0,   # Process every 3 seconds
        'csv_filename': 'whiteboard_edges.csv',
        'image_width': 640,          # Lower resolution for Pi performance
        'image_height': 480,
        'save_images': False         # Set to True to save processed images
    }
    
    print("Raspberry Pi Whiteboard Edge Detector")
    print("=" * 40)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 40)
    
    # Create and run detector
    detector = CameraWhiteboardDetector(**config)
    detector.run_forever()

if __name__ == "__main__":
    main()