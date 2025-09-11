#!/usr/bin/env python3
"""
Whiteboard Position Tracking Test
Temporary test script for 18"x13.5" whiteboard with camera 1.5" above surface.
Integrates triangulation methods from rc_car_controller.py with camera streaming.
"""

import cv2
import numpy as np
import time
import threading
import socket
import math
from queue import Queue, Empty
from typing import Optional, List, Tuple, Generator, Dict
import signal
import sys

# Flask imports
from flask import Flask, Response, render_template_string

# Pi Camera import
from picamera2 import Picamera2

# Import our whiteboard tracker
from whiteboard_tracker5 import WhiteboardTracker5

class WhiteboardPositionTracker:
    def __init__(self, 
                 camera_id: int = 0,
                 stream_width: int = 640,
                 stream_height: int = 480,
                 processing_width: int = 320,
                 capture_fps: int = 10,
                 processing_fps: int = 5,
                 flask_port: int = 5000):
        """
        Initialize whiteboard position tracker for 18"x13.5" test setup
        
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
        
        # Test whiteboard dimensions (18" x 13.5" = 457.2mm x 342.9mm)
        self.whiteboard_width_mm = 457.2   # 18 inches
        self.whiteboard_height_mm = 342.9  # 13.5 inches
        
        # Camera configuration for test setup
        self.camera_height_mm = 38.1  # 1.5 inches above whiteboard surface
        self.camera_fov_horizontal = 62.0  # degrees (Pi Camera Module 3)
        self.camera_fov_vertical = 48.0    # degrees (Pi Camera Module 3)
        
        # Edge classification tolerance
        self.edge_classification_tolerance = 15.0  # degrees
        
        # Initialize whiteboard tracker with debug mode
        self.tracker = WhiteboardTracker5(debug=False)  # Disable debug for cleaner output
        
        # Camera
        self.camera = None
        
        # Threading components
        self.frame_queue = Queue(maxsize=5)
        self.debug_frame_queue = Queue(maxsize=2)
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        
        # Position tracking
        self.current_position = {'x': None, 'y': None, 'confidence': 0.0}
        self.position_history = []
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'edges_detected': 0,
            'markings_detected': 0,
            'position_updates': 0,
            'start_time': time.time(),
            'last_processing_time': 0
        }
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
    
    def get_local_ip(self) -> str:
        """Get the local IP address"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "localhost"
    
    def init_camera(self) -> bool:
        """Initialize Pi Camera Module 3 using picamera2"""
        try:
            # Initialize Picamera2
            self.camera = Picamera2()
            
            # Create configuration for video streaming
            config = self.camera.create_video_configuration(
                main={"size": (self.stream_width, self.stream_height)}
            )
            
            # Configure the camera
            self.camera.configure(config)
            
            # Add controls to improve edge detection
            controls = {
                "AeEnable": True,       # Enable auto exposure
                "AwbEnable": True,      # Enable auto white balance  
                "Brightness": 0.0,      # Neutral brightness
                "Contrast": 1.1,        # Slight contrast boost for edge detection
            }
            
            # Apply controls
            self.camera.set_controls(controls)
            
            # Start the camera
            self.camera.start()
            
            # Give camera time to stabilize
            time.sleep(2)
            
            # Test capture
            try:
                test_frame_rgb = self.camera.capture_array()
                if test_frame_rgb is None or test_frame_rgb.size == 0:
                    print("Error: Could not capture test frame")
                    return False
                
                # Convert to BGR for OpenCV compatibility
                test_frame_bgr = cv2.cvtColor(test_frame_rgb, cv2.COLOR_RGB2BGR)
                
                actual_height, actual_width = test_frame_bgr.shape[:2]
                print(f"Pi Camera Module 3 initialized: {actual_width}x{actual_height}")
                print(f"Test whiteboard: {self.whiteboard_width_mm:.1f}x{self.whiteboard_height_mm:.1f}mm")
                print(f"Camera height: {self.camera_height_mm:.1f}mm")
                return True
                
            except Exception as capture_error:
                print(f"Test capture failed: {capture_error}")
                return False
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            print("Make sure the camera is properly connected and enabled")
            return False
    
    def classify_detected_edges(self, edges: List[Tuple[float, float]], gyro_heading: float, image_shape: Tuple[int, int]) -> dict:
        """Classify detected edges as top/bottom/left/right using line parameters"""
        if not edges:
            return {}
        
        h, w = image_shape
        center_x, center_y = w / 2, h / 2
        classified_edges = {}
        
        for rho, theta in edges:
            # For test setup without gyro, use image-relative classification
            # Convert line angle to degrees
            angle_deg = math.degrees(theta)
            
            # Classify based on line angle and position
            tolerance = self.edge_classification_tolerance
            
            # Determine if line is more horizontal or vertical
            is_horizontal = (abs(angle_deg) < tolerance or 
                           abs(angle_deg - 180) < tolerance or
                           abs(angle_deg - 360) < tolerance)
            
            is_vertical = (abs(angle_deg - 90) < tolerance or 
                         abs(angle_deg - 270) < tolerance)
            
            if is_horizontal:
                # Horizontal line - could be top or bottom edge
                # Use rho and line position to determine which
                if abs(math.sin(theta)) > 0.001:  # Avoid division by zero
                    y_at_center = rho / math.sin(theta)
                    if y_at_center < center_y:
                        # Line is in upper part of image
                        edge_type = 'top'
                    else:
                        # Line is in lower part of image  
                        edge_type = 'bottom'
                else:
                    # Nearly horizontal line, use rho sign
                    edge_type = 'bottom' if rho > 0 else 'top'
                    
            elif is_vertical:
                # Vertical line - could be left or right edge
                # Calculate where line intersects horizontal center of image
                if abs(math.cos(theta)) > 0.001:  # Avoid division by zero
                    x_at_center = rho / math.cos(theta)
                    if x_at_center < center_x:
                        # Line is in left part of image
                        edge_type = 'left'
                    else:
                        # Line is in right part of image
                        edge_type = 'right'
                else:
                    # Nearly vertical line, use rho sign
                    edge_type = 'right' if rho > 0 else 'left'
            else:
                # Diagonal line - classify based on angle
                if 0 < angle_deg < 90:
                    # Positive slope diagonal - could be left edge going up-right
                    edge_type = 'left'
                elif 90 < angle_deg < 180:
                    # Negative slope diagonal - could be right edge going down-right
                    edge_type = 'right'
                else:
                    continue  # Skip ambiguous diagonals
            
            # Store the best edge of each type
            if edge_type not in classified_edges:
                classified_edges[edge_type] = (rho, theta, angle_deg)
            else:
                # Keep the edge with smaller absolute rho (closer to image center, usually clearer)
                current_rho = classified_edges[edge_type][0]
                if abs(rho) < abs(current_rho):
                    classified_edges[edge_type] = (rho, theta, angle_deg)
        
        return classified_edges
    
    def calculate_distance_to_edges(self, classified_edges: dict) -> dict:
        """Calculate real-world distance to each classified whiteboard edge"""
        edge_distances = {}
        
        # Camera parameters for test setup (much closer to whiteboard)
        camera_height = self.camera_height_mm  # 38.1mm (1.5 inches)
        h_fov_rad = math.radians(self.camera_fov_horizontal)
        v_fov_rad = math.radians(self.camera_fov_vertical)
        
        image_width = self.processing_width
        image_height = self.processing_height
        
        # Focal length equivalents in pixels
        focal_length_x = (image_width / 2) / math.tan(h_fov_rad / 2)
        focal_length_y = (image_height / 2) / math.tan(v_fov_rad / 2)
        
        for edge_type, (rho, theta, angle_deg) in classified_edges.items():
            distance_mm = 0.0
            
            if edge_type in ['top', 'bottom']:
                # Horizontal edge - calculate distance using vertical perspective
                if abs(math.sin(theta)) > 0.001:
                    y_intersect = rho / math.sin(theta)  # y-coordinate in pixels
                    
                    # Convert to distance from image center
                    y_from_center = y_intersect - (image_height / 2)
                    
                    # Convert pixel offset to angle
                    angle_from_horizontal = math.atan(y_from_center / focal_length_y)
                    
                    # Calculate horizontal distance on whiteboard
                    if abs(angle_from_horizontal) > 0.001:
                        distance_mm = camera_height / math.tan(abs(angle_from_horizontal))
                    else:
                        distance_mm = 500.0  # Reasonable distance for small whiteboard
                    
                    # Sanity check for small whiteboard
                    distance_mm = min(distance_mm, 800.0)  # Cap at reasonable distance
                    
            elif edge_type in ['left', 'right']:
                # Vertical edge - calculate distance using horizontal perspective  
                if abs(math.cos(theta)) > 0.001:
                    x_intersect = rho / math.cos(theta)  # x-coordinate in pixels
                    
                    # Convert to distance from image center
                    x_from_center = x_intersect - (image_width / 2)
                    
                    # Convert pixel offset to angle
                    angle_from_center = math.atan(x_from_center / focal_length_x)
                    
                    # Calculate distance using horizontal geometry
                    if abs(angle_from_center) > 0.001:
                        # Use similar approach but adapted for horizontal distance
                        distance_mm = camera_height / math.tan(abs(angle_from_center))
                    else:
                        distance_mm = 500.0
                    
                    # Apply scaling factor for horizontal distance calculation
                    # (adjusted for closer camera and smaller whiteboard)
                    distance_mm *= 1.2  # Scaling factor for test setup
                    
                    # Sanity check
                    distance_mm = min(distance_mm, 800.0)
            
            edge_distances[edge_type] = max(distance_mm, 20.0)  # Minimum distance of 2cm
        
        return edge_distances
    
    def triangulate_position_from_edges(self, edge_distances: dict) -> Tuple[float, float, float]:
        """Calculate absolute position from distances to whiteboard edges using triangulation"""
        if not edge_distances:
            return None, None, 0.0
        
        estimated_x = None
        estimated_y = None
        confidence = 0.0
        
        # Available edges for triangulation
        has_top = 'top' in edge_distances
        has_bottom = 'bottom' in edge_distances
        has_left = 'left' in edge_distances
        has_right = 'right' in edge_distances
        
        # Calculate X position from left/right edges
        if has_left and has_right:
            # Both left and right visible - high confidence X position
            left_dist = edge_distances['left']
            right_dist = edge_distances['right']
            total_width = left_dist + right_dist
            
            # Validate against known whiteboard width (more tolerant for test setup)
            if abs(total_width - self.whiteboard_width_mm) < 150:  # 15cm tolerance
                estimated_x = left_dist
                confidence += 0.5
            else:
                # Use individual edges with lower confidence
                if left_dist < right_dist:
                    estimated_x = left_dist
                else:
                    estimated_x = self.whiteboard_width_mm - right_dist
                confidence += 0.3
        elif has_left:
            # Only left edge visible
            estimated_x = edge_distances['left']
            confidence += 0.25
        elif has_right:
            # Only right edge visible
            estimated_x = self.whiteboard_width_mm - edge_distances['right']
            confidence += 0.25
        
        # Calculate Y position from top/bottom edges
        if has_top and has_bottom:
            # Both top and bottom visible - high confidence Y position
            top_dist = edge_distances['top']
            bottom_dist = edge_distances['bottom']
            total_height = top_dist + bottom_dist
            
            # Validate against known whiteboard height
            if abs(total_height - self.whiteboard_height_mm) < 100:  # 10cm tolerance
                estimated_y = top_dist
                confidence += 0.5
            else:
                # Use individual edges with lower confidence
                if top_dist < bottom_dist:
                    estimated_y = top_dist
                else:
                    estimated_y = self.whiteboard_height_mm - bottom_dist
                confidence += 0.3
        elif has_top:
            # Only top edge visible
            estimated_y = edge_distances['top']
            confidence += 0.25
        elif has_bottom:
            # Only bottom edge visible
            estimated_y = self.whiteboard_height_mm - edge_distances['bottom']
            confidence += 0.25
        
        # Apply boundary constraints
        if estimated_x is not None:
            estimated_x = max(0, min(estimated_x, self.whiteboard_width_mm))
        
        if estimated_y is not None:
            estimated_y = max(0, min(estimated_y, self.whiteboard_height_mm))
        
        # Bonus confidence for having multiple edges
        edge_count = len(edge_distances)
        if edge_count >= 3:
            confidence += 0.2
        elif edge_count >= 2:
            confidence += 0.1
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        return estimated_x, estimated_y, confidence
    
    def update_position(self, edge_lines: List[Tuple[float, float]]):
        """Update camera position based on detected edges"""
        if not edge_lines:
            return
        
        try:
            # Step 1: Classify edges (without gyro, use image-relative classification)
            image_shape = (self.processing_height, self.processing_width)
            classified_edges = self.classify_detected_edges(edge_lines, 0.0, image_shape)
            
            if not classified_edges:
                return
            
            # Step 2: Calculate distances to classified edges
            edge_distances = self.calculate_distance_to_edges(classified_edges)
            
            if not edge_distances:
                return
            
            # Step 3: Triangulate position
            estimated_x, estimated_y, confidence = self.triangulate_position_from_edges(edge_distances)
            
            if estimated_x is not None and estimated_y is not None and confidence > 0.1:
                # Update position
                self.current_position = {
                    'x': estimated_x,
                    'y': estimated_y, 
                    'confidence': confidence
                }
                
                # Add to history (keep last 10 positions)
                self.position_history.append({
                    'x': estimated_x,
                    'y': estimated_y,
                    'confidence': confidence,
                    'timestamp': time.time(),
                    'edges': list(classified_edges.keys()),
                    'distances': edge_distances
                })
                
                if len(self.position_history) > 10:
                    self.position_history.pop(0)
                
                self.stats['position_updates'] += 1
                
        except Exception as e:
            print(f"Position update error: {e}")
    
    def capture_loop(self):
        """Continuous camera capture thread"""
        print("Starting camera capture loop...")
        frame_interval = 1.0 / self.capture_fps
        
        while self.running:
            try:
                start_time = time.time()
                
                # Capture frame using picamera2
                frame_rgb = self.camera.capture_array()
                if frame_rgb is not None and frame_rgb.size > 0:
                    # Convert RGB to BGR for OpenCV compatibility
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    timestamp = time.time()
                    
                    # Add to processing queue (non-blocking)
                    try:
                        self.frame_queue.put((frame_bgr.copy(), timestamp), block=False)
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
        """Whiteboard detection and position tracking thread"""
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
                if self.stats['frames_processed'] % 15 == 0:
                    self.print_stats()
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(1)
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process frame with whiteboard detection and position tracking"""
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
                
                # Update position based on detected edges
                original_edge_lines = []
                for rho, theta in edge_lines:
                    # Convert back to processing resolution for position calculation
                    original_rho = rho / scale_factor
                    original_edge_lines.append((original_rho, theta))
                
                self.update_position(original_edge_lines)
                
                # Update statistics
                self.stats['edges_detected'] = len(edge_lines)
                self.stats['markings_detected'] = len(marking_boxes)
            
            # Draw debug visualization with position overlay
            self.draw_debug_overlay(debug_frame, edge_lines, marking_boxes)
            
            # Record processing time
            self.stats['last_processing_time'] = (time.time() - processing_start) * 1000
            
            return debug_frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame  # Return original frame on error
    
    def draw_debug_overlay(self, frame: np.ndarray, edge_lines: List[Tuple[float, float]], 
                          marking_boxes: List[Tuple[int, int]]):
        """Draw debug overlay with position information"""
        h, w = frame.shape[:2]
        
        # Draw detected edge lines
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
        
        # Draw marking boxes
        for x, y in marking_boxes:
            # Ensure coordinates are within frame
            x = max(10, min(w - 11, x))
            y = max(10, min(h - 11, y))
            
            # Draw 20x20 box
            cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 255), 2)
            # Draw center point
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # Draw position information
        pos = self.current_position
        
        # Position text
        if pos['x'] is not None and pos['y'] is not None:
            position_text = f"Position: ({pos['x']:.1f}, {pos['y']:.1f})mm"
            confidence_text = f"Confidence: {pos['confidence']:.2f}"
            
            # Draw position crosshair on whiteboard (if we can estimate screen position)
            if len(edge_lines) >= 1:
                # Draw crosshair at estimated position
                try:
                    # Simple approximation: map position to screen coordinates
                    screen_x = int((pos['x'] / self.whiteboard_width_mm) * w * 0.8 + w * 0.1)
                    screen_y = int((pos['y'] / self.whiteboard_height_mm) * h * 0.8 + h * 0.1)
                    
                    # Draw crosshair
                    cv2.drawMarker(frame, (screen_x, screen_y), (255, 0, 255), cv2.MARKER_CROSS, 20, 3)
                except:
                    pass
        else:
            position_text = "Position: Unknown"
            confidence_text = "Confidence: 0.00"
        
        # Stats text
        stats_text = f"Edges: {len(edge_lines)}, Markings: {len(marking_boxes)}"
        processing_text = f"Processing: {self.stats['last_processing_time']:.1f}ms"
        
        # Draw text with background for readability
        texts = [stats_text, processing_text, position_text, confidence_text]
        y_offset = 30
        
        for text in texts:
            # White background
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
            # Black text
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
    
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
            """Main page with video stream and position info"""
            local_ip = self.get_local_ip()
            
            # Get current position for display
            pos = self.current_position
            position_info = {
                'x_mm': pos['x'] if pos['x'] is not None else 0,
                'y_mm': pos['y'] if pos['y'] is not None else 0,
                'x_in': (pos['x'] / 25.4) if pos['x'] is not None else 0,
                'y_in': (pos['y'] / 25.4) if pos['y'] is not None else 0,
                'confidence': pos['confidence']
            }
            
            html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Whiteboard Position Test - RPi5</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="2">
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
        .position-info {
            background-color: #e8f4f8;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            font-size: 18px;
            font-weight: bold;
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
        .cyan-box { color: cyan; font-weight: bold; }
        .position-marker { color: magenta; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📍 Whiteboard Position Tracking Test</h1>
        <div class="info">
            18" × 13.5" Test Whiteboard • Camera Height: 1.5" • RPi5 + Camera Module 3<br>
            Network Address: {{ local_ip }}:{{ port }}
        </div>
        
        <div class="position-info">
            📍 Camera Position: ({{ "%.1f" | format(position_info.x_mm) }}mm, {{ "%.1f" | format(position_info.y_mm) }}mm)<br>
            📏 In Inches: ({{ "%.2f" | format(position_info.x_in) }}", {{ "%.2f" | format(position_info.y_in) }}") • Confidence: {{ "%.0f" | format(position_info.confidence * 100) }}%
        </div>
        
        <div class="video-container">
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Live Camera Stream">
        </div>
        
        <div class="legend">
            <span class="red-line">— Red Line: Edge 1</span>
            <span class="green-line">— Green Line: Edge 2</span>
            <span class="cyan-box">□ Cyan Box: Markings</span>
            <span class="position-marker">✚ Magenta Cross: Camera Position</span>
        </div>
        
        <div class="stats">
            <div><strong>Test Setup:</strong></div>
            <div>Whiteboard: {{ whiteboard_width }}mm × {{ whiteboard_height }}mm ({{ "%.1f" | format(whiteboard_width/25.4) }}" × {{ "%.1f" | format(whiteboard_height/25.4) }}")</div>
            <div>Camera Height: {{ camera_height }}mm ({{ "%.1f" | format(camera_height/25.4) }}")</div>
            <div style="margin-top: 10px; border-top: 1px solid #ddd; padding-top: 10px;">
                <div><strong>Stream Info:</strong></div>
                <div>Resolution: {{ stream_width }}×{{ stream_height }} @ {{ capture_fps }}fps</div>
                <div>Processing: {{ processing_width }}×{{ processing_height }} @ {{ processing_fps }}fps</div>
            </div>
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
                                        position_info=position_info,
                                        whiteboard_width=self.whiteboard_width_mm,
                                        whiteboard_height=self.whiteboard_height_mm,
                                        camera_height=self.camera_height_mm,
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
        
        pos = self.current_position
        position_str = f"({pos['x']:.1f}, {pos['y']:.1f})mm" if pos['x'] is not None else "Unknown"
        
        print(f"\n=== POSITION TRACKING STATS ===")
        print(f"Runtime: {runtime:.1f}s")
        print(f"Frames captured: {self.stats['frames_captured']} ({capture_rate:.1f}/s)")
        print(f"Frames processed: {self.stats['frames_processed']} ({processing_rate:.1f}/s)")
        print(f"Current detection: {self.stats['edges_detected']} edges, {self.stats['markings_detected']} markings")
        print(f"Current position: {position_str} (confidence: {pos['confidence']:.2f})")
        print(f"Position updates: {self.stats['position_updates']}")
        print(f"Processing time: {self.stats['last_processing_time']:.1f}ms")
        print("===============================\n")
    
    def start(self):
        """Start the position tracking system"""
        print("Starting Whiteboard Position Tracking Test...")
        print("=" * 60)
        
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
        print(f"Test Setup: {self.whiteboard_width_mm:.1f}×{self.whiteboard_height_mm:.1f}mm whiteboard")
        print(f"Camera Height: {self.camera_height_mm:.1f}mm ({self.camera_height_mm/25.4:.1f}\")")
        print(f"Stream: {self.stream_width}×{self.stream_height} @ {self.capture_fps}fps")
        print(f"Processing: {self.processing_width}×{self.processing_height} @ {self.processing_fps}fps")
        print(f"Local access: http://localhost:{self.flask_port}")
        print(f"Network access: http://{local_ip}:{self.flask_port}")
        print("=" * 60)
        print("🎯 Camera position will be displayed in real-time!")
        print("Press Ctrl+C to stop...")
        
        return True
    
    def stop(self):
        """Stop the position tracking system"""
        print("\nStopping position tracking...")
        
        self.running = False
        
        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        # Stop and close camera
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                print(f"Camera cleanup error: {e}")
        
        self.print_stats()
        print("Position tracking stopped.")
    
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
    """Main function for position tracking test"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Raspberry Pi 5 Whiteboard Position Tracking Test")
    print("Triangulates camera position using detected whiteboard edges")
    print("Adapted for 18\" × 13.5\" whiteboard with camera 1.5\" above surface")
    print()
    
    # Configuration for test setup
    config = {
        'camera_id': 0,                 # Pi Camera Module 3
        'stream_width': 640,            # Output stream resolution
        'stream_height': 480,
        'processing_width': 320,        # Processing resolution (faster)
        'capture_fps': 10,              # Camera capture frame rate
        'processing_fps': 5,            # Position tracking frame rate (higher than whiteboard processing)
        'flask_port': 5000,             # Web server port
    }
    
    # Create and run position tracker
    tracker = WhiteboardPositionTracker(**config)
    tracker.run()


if __name__ == "__main__":
    main()