#!/usr/bin/env python3
"""
Whiteboard Detection Streaming Server for Raspberry Pi 5 with Camera Module 3

This script captures video from the Pi Camera Module 3, processes each frame
with whiteboard edge detection and marking detection, overlays the results,
and streams to a local web interface accessible over the network.

Requirements:
- Raspberry Pi 5 with Camera Module 3
- picamera2 (pre-installed on Pi OS)
- Flask
- OpenCV
- NumPy

Usage:
    python3 whiteboard_stream.py

Then access http://[pi-ip]:5000 from any computer on the network.
"""

import cv2
import numpy as np
import time
import threading
from flask import Flask, render_template, Response
import logging
from typing import Optional, Tuple, List

# Import the whiteboard detection class
from whiteboard_tracker4 import WhiteboardDetector

# Suppress Flask's default logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class WhiteboardStreamer:
    def __init__(self, resolution=(640, 480), framerate=10):
        """
        Initialize the whiteboard streaming system
        
        Args:
            resolution: Camera resolution tuple (width, height)
            framerate: Target frames per second
        """
        self.resolution = resolution
        self.framerate = framerate
        self.camera = None
        self.detector = WhiteboardDetector(debug=False)  # Disable debug for performance
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        # Try to import picamera2
        try:
            from picamera2 import Picamera2
            self.Picamera2 = Picamera2
            print("picamera2 imported successfully")
        except ImportError:
            print("ERROR: picamera2 not available. This script requires a Raspberry Pi with picamera2 installed.")
            self.Picamera2 = None
    
    def setup_camera(self):
        """Initialize and configure the Pi Camera"""
        if self.Picamera2 is None:
            return False
            
        try:
            self.camera = self.Picamera2()
            
            # Configure camera for continuous capture
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            
            print(f"Camera configured: {self.resolution[0]}x{self.resolution[1]} @ {self.framerate} FPS")
            return True
            
        except Exception as e:
            print(f"Failed to setup camera: {e}")
            return False
    
    def start_camera(self):
        """Start the camera capture"""
        if self.camera is None:
            return False
            
        try:
            self.camera.start()
            print("Camera started successfully")
            return True
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop and cleanup camera resources"""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
                print("Camera stopped")
            except Exception as e:
                print(f"Error stopping camera: {e}")
    
    def create_overlay_visualization(self, image: np.ndarray, 
                                   detected_lines: List[Tuple[float, float]], 
                                   markings: np.ndarray) -> np.ndarray:
        """
        Create visualization overlay with detected edges and markings
        
        Args:
            image: Original BGR image
            detected_lines: List of (rho, theta) edge line parameters
            markings: Binary mask of detected markings
            
        Returns:
            Image with overlays drawn
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # Overlay markings in cyan (semi-transparent)
        if markings is not None and np.any(markings):
            marking_overlay = np.zeros_like(image)
            marking_overlay[markings > 0] = [255, 255, 0]  # Cyan in BGR
            result = cv2.addWeighted(result, 0.8, marking_overlay, 0.3, 0)
        
        # Draw detected edge lines
        if detected_lines:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
            
            for i, (rho, theta) in enumerate(detected_lines):
                color = colors[i % len(colors)]
                
                # Calculate line endpoints for drawing
                a, b = np.cos(theta), np.sin(theta)
                
                # Find intersections with image boundaries
                intersections = []
                
                # Left edge (x = 0)
                if abs(b) > 0.001:
                    y_left = rho / b
                    if 0 <= y_left <= h:
                        intersections.append((0, int(y_left)))
                
                # Right edge (x = w)
                if abs(b) > 0.001:
                    y_right = (rho - w * a) / b
                    if 0 <= y_right <= h:
                        intersections.append((w-1, int(y_right)))
                
                # Top edge (y = 0)
                if abs(a) > 0.001:
                    x_top = rho / a
                    if 0 <= x_top <= w:
                        intersections.append((int(x_top), 0))
                
                # Bottom edge (y = h)
                if abs(a) > 0.001:
                    x_bottom = (rho - h * b) / a
                    if 0 <= x_bottom <= w:
                        intersections.append((int(x_bottom), h-1))
                
                # Draw line if we have at least 2 intersection points
                if len(intersections) >= 2:
                    cv2.line(result, intersections[0], intersections[1], color, 2)
                    
                    # Add edge line label
                    label_pos = intersections[0]
                    edge_type = "H" if abs(np.degrees(theta)) < 45 or abs(np.degrees(theta)) > 135 else "V"
                    cv2.putText(result, f"Edge {i+1}({edge_type})", 
                              (label_pos[0] + 5, label_pos[1] + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with whiteboard detection
        
        Args:
            frame: RGB image from camera
            
        Returns:
            Processed frame with overlays
        """
        # Convert RGB to BGR for OpenCV processing
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run whiteboard detection
        detection_result = self.detector.detect_whiteboard_edges(bgr_frame)
        
        if detection_result is not None:
            detected_lines, markings, marking_regions = detection_result
            
            # Create visualization overlay
            processed_frame = self.create_overlay_visualization(bgr_frame, detected_lines, markings)
            
            # Add status text
            status_text = f"Edges: {len(detected_lines)}, Markings: {len(marking_regions)} regions"
            cv2.putText(processed_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # No detection - just show original frame with status
            processed_frame = bgr_frame.copy()
            cv2.putText(processed_frame, "No whiteboard detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add FPS counter
        cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return processed_frame
    
    def capture_loop(self):
        """Main capture and processing loop"""
        print("Starting capture loop...")
        
        while self.running:
            try:
                # Capture frame from camera
                frame = self.camera.capture_array()
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Update shared frame buffer
                with self.frame_lock:
                    self.current_frame = processed_frame
                
                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Control frame rate
                time.sleep(1.0 / self.framerate)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[bytes]:
        """
        Get the current processed frame as JPEG bytes for streaming
        
        Returns:
            JPEG encoded frame bytes, or None if no frame available
        """
        with self.frame_lock:
            if self.current_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', self.current_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    return buffer.tobytes()
        return None
    
    def start_streaming(self):
        """Start the camera capture and processing thread"""
        if not self.setup_camera():
            return False
        
        if not self.start_camera():
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("Streaming started")
        return True
    
    def stop_streaming(self):
        """Stop streaming and cleanup"""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        self.stop_camera()
        print("Streaming stopped")

# Global streamer instance
streamer = WhiteboardStreamer()

# Flask web application
app = Flask(__name__)

def generate_frames():
    """Generator function for video streaming"""
    while True:
        frame = streamer.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # No frame available, wait a bit
            time.sleep(0.1)

@app.route('/')
def index():
    """Main page with video stream"""
    return render_template('whiteboard_stream.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Status information endpoint"""
    return {
        'running': streamer.running,
        'fps': streamer.fps,
        'resolution': streamer.resolution
    }

def main():
    """Main function to start the streaming server"""
    print("Whiteboard Detection Streaming Server")
    print("=====================================")
    print("Setting up camera and detection system...")
    
    # Start the camera streaming
    if not streamer.start_streaming():
        print("Failed to start camera streaming")
        return
    
    try:
        print(f"\nStreaming server starting on port 5000")
        print(f"Camera resolution: {streamer.resolution[0]}x{streamer.resolution[1]}")
        print(f"Target framerate: {streamer.framerate} FPS")
        print("\nAccess the stream from any device on your network:")
        print("http://[raspberry-pi-ip]:5000")
        print("\nPress Ctrl+C to stop")
        
        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        streamer.stop_streaming()

if __name__ == '__main__':
    main()