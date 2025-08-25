#!/usr/bin/env python3
"""
Raspberry Pi 5 Whiteboard Camera Stream
Captures images from Pi Camera Module 3, processes with WhiteboardTracker5,
and streams debug images to a local website accessible on the network.
"""

import cv2
import numpy as np
import time
import threading
import socket
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, List, Tuple, Generator
import signal
import sys

# Flask imports
from flask import Flask, Response, render_template_string

# Import our whiteboard tracker
from whiteboard_tracker5 import WhiteboardTracker5

class WhiteboardCameraStream:
    def __init__(self, 
                 camera_id: int = 0,
                 stream_width: int = 640,
                 stream_height: int = 480,
                 processing_width: int = 320,
                 capture_fps: int = 10,
                 processing_fps: int = 3,
                 flask_port: int = 5000):
        """
        Initialize the whiteboard camera stream
        
        Args:
            camera_id: Camera device ID (0 for Pi Camera)
            stream_width: Output stream resolution width
            stream_height: Output stream resolution height  
            processing_width: Processing resolution width (smaller = faster)
            capture_fps: Camera capture frame rate
            processing_fps: Whiteboard processing frame rate
            flask_port: Flask server port
        """
        self.camera_id = camera_id
        self.stream_width = stream_width
        self.stream_height = stream_height
        self.processing_width = processing_width
        self.capture_fps = capture_fps
        self.processing_fps = processing_fps
        self.flask_port = flask_port
        
        # Calculate processing height maintaining aspect ratio
        self.processing_height = int(processing_width * stream_height / stream_width)
        
        # Initialize whiteboard tracker with debug mode for visualization
        self.tracker = WhiteboardTracker5(debug=True)
        
        # Camera
        self.camera = None
        
        # Threading components
        self.frame_queue = Queue(maxsize=5)
        self.debug_frame_queue = Queue(maxsize=2)  # Smaller queue for processed frames
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'edges_detected': 0,
            'markings_detected': 0,
            'start_time': time.time(),
            'last_processing_time': 0
        }
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
        
    def get_local_ip(self) -> str:
        """Get the local IP address"""
        try:
            # Connect to a dummy address to find local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "localhost"
    
    def init_camera(self) -> bool:
        """Initialize Pi Camera Module 3"""
        try:
            # Try different camera backends for Pi Camera
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                self.camera = cv2.VideoCapture(self.camera_id, backend)
                if self.camera.isOpened():
                    break
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {self.camera_id}")
                return False
            
            # Configure camera for Pi Camera Module 3
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.stream_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stream_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.capture_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Auto settings for better whiteboard detection
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)      # Auto focus if available
            
            # Test capture
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("Error: Could not capture test frame")
                return False
            
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"Pi Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def capture_loop(self):
        """Continuous camera capture thread"""
        print("Starting camera capture loop...")
        frame_interval = 1.0 / self.capture_fps
        
        while self.running:
            try:
                start_time = time.time()
                
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    timestamp = time.time()
                    
                    # Add to processing queue (non-blocking)
                    try:
                        self.frame_queue.put((frame.copy(), timestamp), block=False)
                        self.stats['frames_captured'] += 1
                    except:
                        # Queue full, skip frame
                        pass
                else:
                    print("Warning: Failed to capture frame")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(1)
    
    def processing_loop(self):
        """Whiteboard detection and debug image generation thread"""
        print("Starting whiteboard processing loop...")
        processing_interval = 1.0 / self.processing_fps
        last_processing_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to process
                if current_time - last_processing_time < processing_interval:
                    time.sleep(0.1)
                    continue
                
                # Get latest frame, discard older ones
                frame = None
                timestamp = None
                
                while True:
                    try:
                        frame, timestamp = self.frame_queue.get(timeout=0.1)
                    except Empty:
                        break
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame for whiteboard detection
                debug_frame = self.process_frame(frame)
                
                if debug_frame is not None:
                    # Add to debug frame queue
                    try:
                        self.debug_frame_queue.put(debug_frame, block=False)
                    except:
                        # Queue full, skip frame
                        pass
                
                last_processing_time = current_time
                self.stats['frames_processed'] += 1
                
                # Print status periodically
                if self.stats['frames_processed'] % 30 == 0:
                    self.print_stats()
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(1)
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process frame with whiteboard detection and create debug visualization"""
        try:
            processing_start = time.time()
            
            # Resize for processing if needed
            if frame.shape[1] != self.processing_width:
                processing_frame = cv2.resize(frame, (self.processing_width, self.processing_height))
                scale_factor = frame.shape[1] / self.processing_width
            else:
                processing_frame = frame.copy()
                scale_factor = 1.0
            
            # Detect whiteboard edges and markings
            result = self.tracker.detect_whiteboard_edges_rpi5(processing_frame, target_width=self.processing_width)
            
            # Create debug visualization on original resolution frame
            debug_frame = frame.copy()
            edge_lines = []
            marking_boxes = []
            
            if result is not None:
                edge_lines, marking_boxes = result
                
                # Scale results back to original frame size if needed
                if scale_factor != 1.0:
                    scaled_lines = []
                    for rho, theta in edge_lines:
                        scaled_rho = rho * scale_factor
                        scaled_lines.append((scaled_rho, theta))
                    edge_lines = scaled_lines
                    
                    scaled_boxes = []
                    for x, y in marking_boxes:
                        scaled_x = int(x * scale_factor)
                        scaled_y = int(y * scale_factor)
                        scaled_boxes.append((scaled_x, scaled_y))
                    marking_boxes = scaled_boxes
                
                # Update statistics
                self.stats['edges_detected'] = len(edge_lines)
                self.stats['markings_detected'] = len(marking_boxes)
            
            # Draw debug visualization
            self.draw_debug_overlay(debug_frame, edge_lines, marking_boxes)
            
            # Record processing time
            self.stats['last_processing_time'] = (time.time() - processing_start) * 1000
            
            return debug_frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame  # Return original frame on error
    
    def draw_debug_overlay(self, frame: np.ndarray, edge_lines: List[Tuple[float, float]], 
                          marking_boxes: List[Tuple[int, int]]):
        """Draw debug overlay on frame (modifies frame in-place)"""
        h, w = frame.shape[:2]
        
        # Draw detected edge lines (red and green like original debug images)
        colors = [(0, 0, 255), (0, 255, 0)]  # Red, Green in BGR
        for i, (rho, theta) in enumerate(edge_lines[:2]):  # Max 2 lines for clarity
            color = colors[i % len(colors)]
            a, b = np.cos(theta), np.sin(theta)
            
            # Calculate line endpoints
            if abs(b) > 0.001:  # Not nearly vertical
                y1 = int(rho / b) if abs(b) > 0.001 else 0
                y2 = int((rho - w * a) / b) if abs(b) > 0.001 else h
                x1, x2 = 0, w - 1
                
                # Clamp to image bounds
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            else:  # Nearly vertical line
                x_line = int(rho / a) if abs(a) > 0.001 else 0
                x_line = max(0, min(w - 1, x_line))
                cv2.line(frame, (x_line, 0), (x_line, h - 1), color, 3)
        
        # Draw 20x20 marking boxes (green rectangles with yellow centers)
        for x, y in marking_boxes:
            # Ensure coordinates are within frame
            x = max(10, min(w - 11, x))
            y = max(10, min(h - 11, y))
            
            # Draw 20x20 box
            cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Add text overlay with stats
        stats_text = f"Edges: {len(edge_lines)}, Markings: {len(marking_boxes)}"
        processing_text = f"Processing: {self.stats['last_processing_time']:.1f}ms"
        
        # White background for text readability
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        cv2.putText(frame, processing_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(frame, processing_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    def generate_frames(self) -> Generator[bytes, None, None]:
        """Generate JPEG frames for Flask streaming"""
        while self.running:
            try:
                # Get latest debug frame
                debug_frame = None
                while True:
                    try:
                        debug_frame = self.debug_frame_queue.get(timeout=0.5)
                    except Empty:
                        break
                
                if debug_frame is not None:
                    # Encode as JPEG
                    ret, buffer = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Frame generation error: {e}")
                time.sleep(1)
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page with video stream"""
            local_ip = self.get_local_ip()
            html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Whiteboard Camera Stream - RPi5</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .info {
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .video-container {
            position: relative;
            display: inline-block;
            max-width: 100%;
        }
        #videoStream {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .stats {
            margin-top: 20px;
            font-family: monospace;
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            text-align: left;
            display: inline-block;
        }
        .legend {
            margin-top: 15px;
            font-size: 14px;
            color: #666;
        }
        .legend span {
            display: inline-block;
            margin: 0 15px;
        }
        .red-line { color: red; font-weight: bold; }
        .green-line { color: green; font-weight: bold; }
        .green-box { color: green; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔲 Whiteboard Edge Detection Stream</h1>
        <div class="info">
            Raspberry Pi 5 + Camera Module 3<br>
            Network Address: {{ local_ip }}:{{ port }}
        </div>
        
        <div class="video-container">
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Live Camera Stream">
        </div>
        
        <div class="legend">
            <span class="red-line">— Red Line: Edge 1</span>
            <span class="green-line">— Green Line: Edge 2</span>
            <span class="green-box">□ Green Box: Markings</span>
        </div>
        
        <div class="stats">
            <div>Stream Resolution: {{ stream_width }}x{{ stream_height }}</div>
            <div>Processing Resolution: {{ processing_width }}x{{ processing_height }}</div>
            <div>Capture FPS: {{ capture_fps }}</div>
            <div>Processing FPS: {{ processing_fps }}</div>
        </div>
    </div>
    
    <script>
        // Auto-refresh if stream fails
        document.getElementById('videoStream').onerror = function() {
            setTimeout(function() {
                document.getElementById('videoStream').src = "{{ url_for('video_feed') }}?" + new Date().getTime();
            }, 5000);
        };
    </script>
</body>
</html>
            """
            return render_template_string(html_template, 
                                        local_ip=local_ip,
                                        port=self.flask_port,
                                        stream_width=self.stream_width,
                                        stream_height=self.stream_height,
                                        processing_width=self.processing_width,
                                        processing_height=self.processing_height,
                                        capture_fps=self.capture_fps,
                                        processing_fps=self.processing_fps)
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def print_stats(self):
        """Print current statistics"""
        runtime = time.time() - self.stats['start_time']
        capture_rate = self.stats['frames_captured'] / runtime if runtime > 0 else 0
        processing_rate = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        
        print(f"\n=== WHITEBOARD STREAM STATS ===")
        print(f"Runtime: {runtime:.1f}s")
        print(f"Frames captured: {self.stats['frames_captured']} ({capture_rate:.1f}/s)")
        print(f"Frames processed: {self.stats['frames_processed']} ({processing_rate:.1f}/s)")
        print(f"Current detection: {self.stats['edges_detected']} edges, {self.stats['markings_detected']} markings")
        print(f"Processing time: {self.stats['last_processing_time']:.1f}ms")
        print("===============================\n")
    
    def start(self):
        """Start the camera stream system"""
        print("Starting Whiteboard Camera Stream...")
        print("=" * 50)
        
        # Initialize camera
        if not self.init_camera():
            print("Failed to initialize camera")
            return False
        
        # Start processing threads
        self.running = True
        
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        
        self.capture_thread.start()
        self.processing_thread.start()
        
        # Get network info
        local_ip = self.get_local_ip()
        
        print(f"Camera: Pi Camera Module 3")
        print(f"Stream: {self.stream_width}x{self.stream_height} @ {self.capture_fps}FPS")
        print(f"Processing: {self.processing_width}x{self.processing_height} @ {self.processing_fps}FPS")
        print(f"Local access: http://localhost:{self.flask_port}")
        print(f"Network access: http://{local_ip}:{self.flask_port}")
        print("=" * 50)
        print("Access the stream from any device on your network!")
        print("Press Ctrl+C to stop...")
        
        return True
    
    def stop(self):
        """Stop the camera stream system"""
        print("\nStopping whiteboard camera stream...")
        
        self.running = False
        
        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        self.print_stats()
        print("Camera stream stopped.")
    
    def run(self):
        """Run the Flask app"""
        if not self.start():
            return
        
        try:
            # Run Flask app (this blocks)
            self.app.run(host='0.0.0.0', port=self.flask_port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received...")
        finally:
            self.stop()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\nShutdown signal received...")
    sys.exit(0)


def main():
    """Main function"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Raspberry Pi 5 Whiteboard Camera Stream")
    print("Captures from Pi Camera Module 3 and streams debug visualization")
    print()
    
    # Configuration
    config = {
        'camera_id': 0,                 # Pi Camera Module 3
        'stream_width': 640,            # Output stream resolution
        'stream_height': 480,
        'processing_width': 320,        # Processing resolution (faster)
        'capture_fps': 10,              # Camera capture frame rate
        'processing_fps': 3,            # Whiteboard processing frame rate
        'flask_port': 5000              # Web server port
    }
    
    # Create and run stream
    stream = WhiteboardCameraStream(**config)
    stream.run()


if __name__ == "__main__":
    main()