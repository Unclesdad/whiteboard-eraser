#!/usr/bin/env python3
"""
Simple Camera Stream with Basic Marking Detection
Uses the simplified detection approach
"""

import io
import socketserver
import threading
import time
from threading import Condition
from http import server
from dataclasses import dataclass, field
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# Import our simple marking detector
from simple_marking_detector import SimpleMarkingDetector

# Configuration - LOW POWER MODE for battery operation
CAMERA_WIDTH = 960   # Reduced from 1920 to save power
CAMERA_HEIGHT = 540  # Reduced from 1080 to save power

# Processing resolution (downscale for speed while keeping full FOV)
PROCESSING_WIDTH = 640   # Reduced from 1280 to save processing power
PROCESSING_HEIGHT = 360  # Reduced from 720 to save processing power

PAGE = """\
<html>
<head>
<title>Whiteboard Detection with Spatial Mapping</title>
<style>
body { font-family: Arial, sans-serif; background-color: #f0f0f0; margin: 0; padding: 10px; }
.container { text-align: center; max-width: 1400px; margin: 0 auto; }
.stats { background-color: white; padding: 10px; margin: 10px auto; border-radius: 5px; }
.video-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin: 20px auto;
    max-width: 1400px;
}
.video-panel {
    background-color: white;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.video {
    width: 100%;
    height: auto;
    border: 2px solid #333;
    border-radius: 5px;
    display: block;
}
.video-label {
    text-align: center;
    margin-top: 8px;
    font-weight: bold;
    font-size: 14px;
}
.video-description {
    text-align: center;
    margin-top: 5px;
    font-size: 11px;
    color: #666;
}
h1 { margin: 10px 0; font-size: 24px; }
</style>
</head>
<body>
<div class="container">
<h1>Whiteboard Detection with Spatial Mapping</h1>
<div class="stats">
<p><strong>Approach:</strong> Brightness threshold → Bottom blob detection → Dark spot detection → Temporal tracking</p>
<p><strong>Markings:</strong> <span id="markings">0</span> | <strong>FPS:</strong> <span id="fps">0.0</span></p>
</div>

<div class="video-grid">
<div class="video-panel">
<img src="stream.mjpg" class="video">
<div class="video-label">1. Annotated Camera Feed</div>
<div class="video-description">Yellow boundary = detected surface, Colored boxes = markings</div>
</div>

<div class="video-panel">
<img src="brightness.mjpg" class="video">
<div class="video-label">2. Raw Brightness Mask</div>
<div class="video-description">Raw brightness thresholding (all pixels above threshold)</div>
</div>

<div class="video-panel">
<img src="surface.mjpg" class="video">
<div class="video-label">3. Final Surface Blob</div>
<div class="video-description">Bottom-connected blob selected as whiteboard surface</div>
</div>

<div class="video-panel">
<img src="map.mjpg" class="video">
<div class="video-label">4. Spatial Map (Top-Down)</div>
<div class="video-description">800x600mm view • Green=Established, Yellow=Medium, Orange=New</div>
</div>
</div>

<div class="stats">
<p><strong>Spatial Map:</strong> Shows markings tracked over time relative to car position.
Markings reinforce with repeated detection (merge within 30mm), fade after 10 absent frames.</p>
<p><strong>Camera:</strong> 7.5cm height, 20.5° down angle, converts pixels → mm displacement → car coordinates</p>
</div>
</div>
<script>
setInterval(function() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('markings').textContent = data.markings;
            document.getElementById('fps').textContent = data.fps.toFixed(1);
        })
        .catch(err => console.log('Status update failed'));
}, 2000);
</script>
</body>
</html>
"""

@dataclass
class TrackedMarking:
    """Represents a marking tracked over time"""
    x: float  # mm from car center
    y: float  # mm from car center
    confidence: float
    observation_count: int = 1
    frames_absent: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

class SimpleDetectionOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.brightness_frame = None
        self.surface_frame = None
        self.map_frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.brightness_condition = Condition()
        self.surface_condition = Condition()
        self.map_condition = Condition()

        # Simple detection system - configure for processing resolution
        self.detector = SimpleMarkingDetector(
            image_width=PROCESSING_WIDTH,
            image_height=PROCESSING_HEIGHT,
            debug=True
        )
        self.detection_thread = None
        self.detection_running = False
        self.current_markings = []
        self.detection_fps = 0.0

        # Raw frame buffer
        self.raw_frame = None
        self.raw_condition = Condition()

        # Temporal tracking of markings
        self.tracked_markings = []  # List of TrackedMarking
        self.tracking_lock = threading.Lock()
        self.merge_distance_mm = 30.0  # Merge markings within 30mm
        self.forget_threshold = 10  # Forget after 10 absent frames

        print("✓ Simple detection system initialized")

    def start_detection(self):
        """Start the detection processing thread"""
        self.detection_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("✓ Simple detection thread started")

    def stop_detection(self):
        """Stop the detection processing"""
        self.detection_running = False
        if self.detection_thread:
            self.detection_thread.join()

    def _detection_loop(self):
        """Main detection processing loop"""
        detection_times = []

        while self.detection_running:
            try:
                # Wait for new raw frame
                with self.raw_condition:
                    if self.raw_frame is None:
                        self.raw_condition.wait(timeout=0.1)
                        continue

                    frame = self.raw_frame.copy()

                # Downscale frame for faster processing while keeping full FOV
                if frame.shape[1] != PROCESSING_WIDTH or frame.shape[0] != PROCESSING_HEIGHT:
                    processing_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
                else:
                    processing_frame = frame

                # Run simple detection on downscaled frame
                start_time = time.time()
                markings = self.detector.detect_markings(processing_frame)
                detection_time = time.time() - start_time

                # Update stats
                detection_times.append(detection_time)
                if len(detection_times) > 30:
                    detection_times.pop(0)

                self.detection_fps = 1.0 / np.mean(detection_times) if detection_times else 0.0

                # Scale markings back to original resolution for visualization
                scale_x = CAMERA_WIDTH / PROCESSING_WIDTH
                scale_y = CAMERA_HEIGHT / PROCESSING_HEIGHT
                scaled_markings = []

                for marking in markings:
                    from simple_marking_detector import Marking
                    scaled_marking = Marking(
                        x=marking.x * scale_x,
                        y=marking.y * scale_y,
                        area=marking.area * scale_x * scale_y,
                        confidence=marking.confidence,
                        bbox=(int(marking.bbox[0] * scale_x),
                              int(marking.bbox[1] * scale_y),
                              int(marking.bbox[2] * scale_x),
                              int(marking.bbox[3] * scale_y))
                    )
                    scaled_markings.append(scaled_marking)

                self.current_markings = scaled_markings

                # Convert markings to car coordinates and update tracking
                car_markings = []
                for marking in markings:
                    cam_x, cam_y = self.detector.pixel_to_camera_relative_mm(marking.x, marking.y)
                    car_x, car_y = self.detector.camera_relative_to_car_center_mm(cam_x, cam_y)
                    car_markings.append((car_x, car_y, marking.confidence))

                self._update_tracked_markings(car_markings)

                # Create annotated frame using original resolution
                annotated_frame = self.detector.visualize_detections(frame, scaled_markings)

                # Create brightness mask visualization using processing resolution
                brightness_frame = self._create_brightness_visualization(processing_frame)
                brightness_frame = cv2.resize(brightness_frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

                # Create final surface visualization using processing resolution
                surface_frame = self._create_surface_visualization(processing_frame)
                surface_frame = cv2.resize(surface_frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

                # Create map visualization
                map_frame = self._create_map_visualization()

                # Convert all four to JPEG
                _, jpeg_data = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, brightness_jpeg_data = cv2.imencode('.jpg', brightness_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, surface_jpeg_data = cv2.imencode('.jpg', surface_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, map_jpeg_data = cv2.imencode('.jpg', map_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Update streaming buffers
                with self.condition:
                    self.frame = jpeg_data.tobytes()
                    self.condition.notify_all()

                with self.brightness_condition:
                    self.brightness_frame = brightness_jpeg_data.tobytes()
                    self.brightness_condition.notify_all()

                with self.surface_condition:
                    self.surface_frame = surface_jpeg_data.tobytes()
                    self.surface_condition.notify_all()

                with self.map_condition:
                    self.map_frame = map_jpeg_data.tobytes()
                    self.map_condition.notify_all()

                # Increased delay for battery power savings (was 0.05)
                time.sleep(0.1)  # Reduces to ~10 FPS max

            except Exception as e:
                print(f"Simple detection error: {e}")
                time.sleep(0.1)

    def _create_brightness_visualization(self, frame):
        """Create visualization of raw brightness thresholding"""
        # Correct camera orientation
        corrected = self.detector.rotate_image_180(frame.copy())

        # Get raw brightness mask (replicate the detector's logic)
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        brightness_threshold = 180
        _, brightness_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Convert mask to 3-channel for display
        brightness_vis = cv2.cvtColor(brightness_mask, cv2.COLOR_GRAY2BGR)

        # Add text info
        white_pixels = np.sum(brightness_mask) / 255.0
        cv2.putText(brightness_vis, f"Threshold: {brightness_threshold}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(brightness_vis, f"White pixels: {white_pixels:.0f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(brightness_vis, "Raw brightness mask", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        return brightness_vis

    def _create_surface_visualization(self, frame):
        """Create visualization of final detected surface blob"""
        # Correct camera orientation
        corrected = self.detector.rotate_image_180(frame.copy())

        # Get the final surface mask
        surface_mask = self.detector.find_white_surface(corrected)

        # Convert mask to 3-channel for display
        surface_vis = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)

        # Draw boundary of detected surface
        contours, _ = cv2.findContours(surface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(surface_vis, [largest_contour], -1, (0, 255, 255), 2)

        # Add text info
        surface_area = np.sum(surface_mask) / 255.0
        cv2.putText(surface_vis, f"Surface: {surface_area:.0f}px", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(surface_vis, "Bottom-connected blob", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(surface_vis, "Final surface mask", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        return surface_vis

    def _update_tracked_markings(self, car_markings):
        """Update temporal tracking of markings with reinforcement and forgetting"""
        with self.tracking_lock:
            # Mark all tracked markings as not seen this frame
            for tracked in self.tracked_markings:
                tracked.frames_absent += 1

            # Process new detections
            for car_x, car_y, confidence in car_markings:
                # Find if this marking matches an existing tracked marking
                matched = False
                for tracked in self.tracked_markings:
                    distance = np.sqrt((tracked.x - car_x)**2 + (tracked.y - car_y)**2)
                    if distance <= self.merge_distance_mm:
                        # Reinforce existing marking
                        total_weight = tracked.confidence * tracked.observation_count + confidence
                        tracked.x = (tracked.x * tracked.confidence * tracked.observation_count +
                                   car_x * confidence) / total_weight
                        tracked.y = (tracked.y * tracked.confidence * tracked.observation_count +
                                   car_y * confidence) / total_weight
                        tracked.confidence = min(1.0, total_weight / (tracked.observation_count + 1))
                        tracked.observation_count += 1
                        tracked.frames_absent = 0
                        tracked.last_seen = time.time()
                        matched = True
                        break

                if not matched:
                    # Create new tracked marking
                    new_tracked = TrackedMarking(
                        x=car_x,
                        y=car_y,
                        confidence=confidence
                    )
                    self.tracked_markings.append(new_tracked)

            # Remove markings that have been absent too long
            self.tracked_markings = [
                t for t in self.tracked_markings
                if t.frames_absent < self.forget_threshold
            ]

    def _create_map_visualization(self):
        """Create top-down 2D map visualization of tracked markings"""
        # Map dimensions in pixels
        map_width = 640
        map_height = 480

        # View area in mm (800mm x 600mm centered on car)
        view_width_mm = 800.0
        view_height_mm = 600.0

        # Scale: pixels per mm
        scale_x = map_width / view_width_mm
        scale_y = map_height / view_height_mm

        # Create white background
        map_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

        # Draw grid (100mm spacing)
        grid_spacing_mm = 100.0
        for x_mm in np.arange(-view_width_mm/2, view_width_mm/2, grid_spacing_mm):
            x_px = int((x_mm + view_width_mm/2) * scale_x)
            cv2.line(map_img, (x_px, 0), (x_px, map_height), (220, 220, 220), 1)

        for y_mm in np.arange(-view_height_mm/2, view_height_mm/2, grid_spacing_mm):
            y_px = int((y_mm + view_height_mm/2) * scale_y)
            cv2.line(map_img, (0, y_px), (map_width, y_px), (220, 220, 220), 1)

        # Draw axes
        center_x = map_width // 2
        center_y = map_height // 2
        cv2.line(map_img, (center_x, 0), (center_x, map_height), (180, 180, 180), 2)
        cv2.line(map_img, (0, center_y), (map_width, center_y), (180, 180, 180), 2)

        # Draw tracked markings
        with self.tracking_lock:
            for tracked in self.tracked_markings:
                # Convert car coordinates to map pixels
                # Car center is at map center, forward (positive Y) is up
                map_x = int(center_x + tracked.x * scale_x)
                map_y = int(center_y - tracked.y * scale_y)  # Flip Y for screen coordinates

                # Skip if out of bounds
                if map_x < 0 or map_x >= map_width or map_y < 0 or map_y >= map_height:
                    continue

                # Color based on confidence and observation count
                if tracked.observation_count >= 5:
                    color = (0, 200, 0)  # Green - well established
                elif tracked.observation_count >= 2:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 128, 255)  # Orange - new detection

                # Fade if not seen recently
                if tracked.frames_absent > 0:
                    fade_factor = 1.0 - (tracked.frames_absent / self.forget_threshold)
                    color = tuple(int(c * fade_factor + 255 * (1 - fade_factor)) for c in color)

                # Draw marking
                radius = max(3, int(15 * scale_x))  # ~15mm marking radius
                cv2.circle(map_img, (map_x, map_y), radius, color, -1)
                cv2.circle(map_img, (map_x, map_y), radius, (100, 100, 100), 1)

                # Draw observation count
                if tracked.observation_count > 1:
                    cv2.putText(map_img, f"{tracked.observation_count}",
                              (map_x + radius + 2, map_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Draw car at center (triangle pointing up/forward)
        car_size = 30
        car_points = np.array([
            [center_x, center_y - car_size],  # Front point
            [center_x - car_size//2, center_y + car_size//2],  # Back left
            [center_x + car_size//2, center_y + car_size//2]   # Back right
        ], np.int32)
        cv2.fillPoly(map_img, [car_points], (255, 100, 100))
        cv2.polylines(map_img, [car_points], True, (200, 0, 0), 2)

        # Add legend and info
        cv2.putText(map_img, "Map View (800x600mm)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        with self.tracking_lock:
            active_count = len([t for t in self.tracked_markings if t.frames_absent == 0])
            total_count = len(self.tracked_markings)

        cv2.putText(map_img, f"Markings: {active_count} active / {total_count} total",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(map_img, "Green=Established, Yellow=Medium, Orange=New",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Add scale reference
        scale_length_mm = 100.0
        scale_length_px = int(scale_length_mm * scale_x)
        scale_y_pos = map_height - 20
        cv2.line(map_img, (10, scale_y_pos), (10 + scale_length_px, scale_y_pos), (0, 0, 0), 2)
        cv2.putText(map_img, "100mm", (15, scale_y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return map_img

    def write(self, buf):
        """Handle incoming camera data"""
        if buf.startswith(b'\xff\xd8'):
            # New JPEG frame - decode for processing
            try:
                nparr = np.frombuffer(self.buffer.getvalue(), np.uint8)
                if len(nparr) > 0:
                    decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if decoded_frame is not None:
                        with self.raw_condition:
                            self.raw_frame = decoded_frame
                            self.raw_condition.notify()
            except Exception as e:
                print(f"Frame decode error: {e}")

            self.buffer.truncate()
            self.buffer.seek(0)

        return self.buffer.write(buf)

class SimpleStreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/status':
            import json
            status_data = {
                'markings': len(output.current_markings),
                'fps': output.detection_fps
            }
            content = json.dumps(status_data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    if frame:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                print(f'Removed streaming client {self.client_address}: {e}')
        elif self.path == '/brightness.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.brightness_condition:
                        output.brightness_condition.wait()
                        frame = output.brightness_frame
                    if frame:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                print(f'Removed brightness streaming client {self.client_address}: {e}')
        elif self.path == '/surface.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.surface_condition:
                        output.surface_condition.wait()
                        frame = output.surface_frame
                    if frame:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                print(f'Removed surface streaming client {self.client_address}: {e}')
        elif self.path == '/map.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.map_condition:
                        output.map_condition.wait()
                        frame = output.map_frame
                    if frame:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                print(f'Removed map streaming client {self.client_address}: {e}')
        else:
            self.send_error(404)
            self.end_headers()

class SimpleStreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    global output, picam2

    print("🔧 Starting Simple Whiteboard Detection Test...")

    try:
        # Initialize camera
        print("📷 Starting camera...")
        picam2 = Picamera2(camera_num=0)

        # Create configuration with manual controls
        config = picam2.create_video_configuration(main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)})
        picam2.configure(config)
        print(f"✓ Camera configured for {CAMERA_WIDTH}x{CAMERA_HEIGHT} resolution")

        # Set manual camera controls for consistent brightness
        controls = {
            "AeEnable": False,          # Disable auto-exposure
            "AwbEnable": True,          # Enable auto white balance for color consistency
            "ExposureTime": 8000,       # Reduced exposure time to save power 
            "AnalogueGain": 2.5,        # Reduced gain to save power
            "Brightness": 0.725,          # High brightness (0.0 to 1.0)
            "Contrast": 2,              # Higher contrast for better marking distinction
        }
        picam2.set_controls(controls)
        print(f"✓ Camera controls set: {controls}")

        # Initialize simple detection output
        output = SimpleDetectionOutput()

        # Start camera recording
        picam2.start_recording(MJPEGEncoder(), FileOutput(output))
        print("✓ Camera recording started")

        # Start detection processing
        output.start_detection()

        # Start web server
        address = ('', 8000)
        server = SimpleStreamingServer(address, SimpleStreamingHandler)

        print("🌐 Simple detection server starting...")
        print(f"📺 Open your browser: http://localhost:8000")
        print("🎯 Should show yellow line for whiteboard edge + detected markings")
        print("⚡ Press Ctrl+C to stop")
        print()

        server.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'output' in globals():
            output.stop_detection()
        if 'picam2' in globals():
            picam2.stop_recording()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()