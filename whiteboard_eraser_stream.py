#!/usr/bin/env python3
"""
Whiteboard Eraser Car with Live HTTP Streaming
Combines autonomous operation with real-time web-based monitoring
"""

import cv2
import numpy as np
import time
import threading
import signal
import sys
import io
import socketserver
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from threading import Condition
from http import server

# Import whiteboard eraser modules
from whiteboard_eraser_main import WhiteboardEraserMain, EraserConfig, EraserState
from simple_marking_detector import SimpleMarkingDetector, Marking

# Import camera
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    CAMERA_AVAILABLE = True
except ImportError:
    print("Warning: picamera2 not available. Running without camera.")
    CAMERA_AVAILABLE = False

# Configuration
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 360
HTTP_PORT = 8000

# HTML Page with 5 panels (4 visual + 1 status)
PAGE = """\
<html>
<head>
<title>Autonomous Whiteboard Eraser - Live View</title>
<style>
body { font-family: Arial, sans-serif; background-color: #f0f0f0; margin: 0; padding: 10px; }
.container { text-align: center; max-width: 1400px; margin: 0 auto; }
.header { background-color: white; padding: 10px; margin: 10px auto; border-radius: 5px; }
.video-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
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
.video-panel.wide {
    grid-column: span 2;
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
.status-panel {
    background-color: white;
    padding: 15px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: left;
}
.status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
}
.status-item {
    padding: 8px;
    background-color: #f5f5f5;
    border-radius: 3px;
}
.status-label {
    font-weight: bold;
    font-size: 12px;
    color: #666;
}
.status-value {
    font-size: 16px;
    margin-top: 4px;
}
.state-display {
    font-size: 24px;
    font-weight: bold;
    padding: 15px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 5px;
    margin: 10px 0;
    text-align: center;
}
h1 { margin: 10px 0; font-size: 24px; }
</style>
</head>
<body>
<div class="container">
<h1>🤖 Autonomous Whiteboard Eraser - Live View</h1>
<div class="header">
<div class="state-display" id="stateDisplay">INITIALIZING</div>
</div>

<div class="video-grid">
<div class="video-panel">
<img src="stream.mjpg" class="video">
<div class="video-label">1. Annotated Camera Feed</div>
<div class="video-description">Live marking detection with surface boundary</div>
</div>

<div class="video-panel">
<img src="brightness.mjpg" class="video">
<div class="video-label">2. Brightness Mask</div>
<div class="video-description">Threshold visualization</div>
</div>

<div class="video-panel">
<img src="surface.mjpg" class="video">
<div class="video-label">3. Surface Detection</div>
<div class="video-description">Bottom-connected blob</div>
</div>

<div class="video-panel wide">
<img src="map.mjpg" class="video">
<div class="video-label">4. Spatial Map (1000x900mm)</div>
<div class="video-description">Top-down view • Green=Established, Yellow=Medium, Orange=New</div>
</div>

<div class="status-panel">
<div class="video-label">5. System Status</div>
<div class="status-grid">
<div class="status-item">
<div class="status-label">POSITION</div>
<div class="status-value" id="position">0, 0, 0°</div>
</div>
<div class="status-item">
<div class="status-label">MARKINGS</div>
<div class="status-value" id="markings">0 detected / 0 erased</div>
</div>
<div class="status-item">
<div class="status-label">PROGRESS</div>
<div class="status-value" id="progress">0%</div>
</div>
<div class="status-item">
<div class="status-label">TARGET</div>
<div class="status-value" id="target">None</div>
</div>
<div class="status-item">
<div class="status-label">CAR SPEED</div>
<div class="status-value" id="speed">0 mm/s</div>
</div>
<div class="status-item">
<div class="status-label">RUNTIME</div>
<div class="status-value" id="runtime">0s</div>
</div>
</div>
</div>
</div>

</div>
<script>
setInterval(function() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('stateDisplay').textContent = data.state.toUpperCase();
            document.getElementById('position').textContent =
                `${data.position.x}mm, ${data.position.y}mm, ${data.position.theta}°`;
            document.getElementById('markings').textContent =
                `${data.markings.detected} detected / ${data.markings.erased} erased`;
            document.getElementById('progress').textContent = `${data.progress}%`;
            document.getElementById('target').textContent = data.target || 'None';
            document.getElementById('speed').textContent = `${data.speed} mm/s`;
            document.getElementById('runtime').textContent = `${data.runtime}s`;
        })
        .catch(err => console.log('Status update failed'));
}, 1000);
</script>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    """
    Handles camera frame output with detection processing and HTTP streaming
    Integrates SimpleMarkingDetector with multi-panel visualization
    """

    def __init__(self, eraser_system):
        self.frame = None
        self.brightness_frame = None
        self.surface_frame = None
        self.map_frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.brightness_condition = Condition()
        self.surface_condition = Condition()
        self.map_condition = Condition()

        # Reference to main eraser system for shared data
        self.eraser_system = eraser_system

        # Simple detection system
        self.detector = SimpleMarkingDetector(
            camera_height_mm=75.0,
            camera_angle_deg=20.5,
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

        # Temporal tracking (same as simple_camera_stream.py)
        self.tracked_markings = []
        self.tracking_lock = threading.Lock()
        self.merge_distance_mm = 30.0
        self.forget_threshold = 10

        print("✓ Streaming output initialized")

    def start_detection(self):
        """Start the detection processing thread"""
        self.detection_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("✓ Detection thread started")

    def stop_detection(self):
        """Stop the detection processing thread"""
        self.detection_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)

    def _detection_loop(self):
        """Main detection processing loop - runs in separate thread"""
        while self.detection_running:
            # Wait for new frame
            with self.raw_condition:
                self.raw_condition.wait(timeout=0.5)
                frame = self.raw_frame

            if frame is None:
                continue

            try:
                # Process at lower resolution
                processing_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))

                # Detect markings
                start_time = time.time()
                markings = self.detector.detect_markings(processing_frame)
                detection_time = time.time() - start_time
                self.detection_fps = 1.0 / detection_time if detection_time > 0 else 0.0

                # Scale markings back to original resolution
                scale_x = CAMERA_WIDTH / PROCESSING_WIDTH
                scale_y = CAMERA_HEIGHT / PROCESSING_HEIGHT

                scaled_markings = []
                for marking in markings:
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

                # Convert to car coordinates and update tracking
                car_markings = []
                for marking in markings:
                    cam_x, cam_y = self.detector.pixel_to_camera_relative_mm(marking.x, marking.y)
                    car_x, car_y = self.detector.camera_relative_to_car_center_mm(cam_x, cam_y)
                    car_markings.append((car_x, car_y, marking.confidence))

                self._update_tracked_markings(car_markings)

                # Also update eraser system's global map if available
                if self.eraser_system and hasattr(self.eraser_system, 'global_map'):
                    # Convert car-relative to global coordinates
                    pose = self.eraser_system.localization.get_pose()
                    global_markings = []
                    for car_x, car_y, conf in car_markings:
                        # Transform from car frame to global frame
                        cos_theta = np.cos(pose.theta)
                        sin_theta = np.sin(pose.theta)
                        global_x = pose.x + car_x * cos_theta - car_y * sin_theta
                        global_y = pose.y + car_x * sin_theta + car_y * cos_theta
                        global_markings.append((global_x, global_y, conf))

                    self.eraser_system.global_map.add_markings(global_markings)

                # Create visualizations
                annotated_frame = self.detector.visualize_detections(frame, scaled_markings)
                brightness_frame = self._create_brightness_visualization(processing_frame)
                brightness_frame = cv2.resize(brightness_frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                surface_frame = self._create_surface_visualization(processing_frame)
                surface_frame = cv2.resize(surface_frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                map_frame = self._create_map_visualization()

                # Convert to JPEG and update streams
                _, jpeg_data = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, brightness_jpeg = cv2.imencode('.jpg', brightness_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, surface_jpeg = cv2.imencode('.jpg', surface_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, map_jpeg = cv2.imencode('.jpg', map_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                with self.condition:
                    self.frame = jpeg_data.tobytes()
                    self.condition.notify_all()

                with self.brightness_condition:
                    self.brightness_frame = brightness_jpeg.tobytes()
                    self.brightness_condition.notify_all()

                with self.surface_condition:
                    self.surface_frame = surface_jpeg.tobytes()
                    self.surface_condition.notify_all()

                with self.map_condition:
                    self.map_frame = map_jpeg.tobytes()
                    self.map_condition.notify_all()

                time.sleep(0.1)  # 10 FPS max

            except Exception as e:
                print(f"Detection error: {e}")
                import traceback
                traceback.print_exc()

    def _update_tracked_markings(self, car_markings):
        """Update temporal tracking (same as simple_camera_stream.py)"""
        with self.tracking_lock:
            for tracked in self.tracked_markings:
                tracked.frames_absent += 1

            for car_x, car_y, confidence in car_markings:
                matched = False
                for tracked in self.tracked_markings:
                    distance = np.sqrt((tracked.x - car_x)**2 + (tracked.y - car_y)**2)
                    if distance <= self.merge_distance_mm:
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
                    from dataclasses import dataclass, field
                    @dataclass
                    class TrackedMarking:
                        x: float
                        y: float
                        confidence: float
                        observation_count: int = 1
                        frames_absent: int = 0
                        first_seen: float = field(default_factory=time.time)
                        last_seen: float = field(default_factory=time.time)

                    new_tracked = TrackedMarking(x=car_x, y=car_y, confidence=confidence)
                    self.tracked_markings.append(new_tracked)

            self.tracked_markings = [
                t for t in self.tracked_markings
                if t.frames_absent < self.forget_threshold
            ]

    def _create_brightness_visualization(self, image):
        """Create brightness mask visualization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        bright_vis = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
        white_pixels = np.sum(bright_mask > 0)
        cv2.putText(bright_vis, f"White pixels: {white_pixels}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bright_vis, "Raw brightness mask", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        return bright_vis

    def _create_surface_visualization(self, image):
        """Create surface detection visualization"""
        surface_mask = self.detector.find_white_surface(image)
        surface_vis = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)
        surface_pixels = np.sum(surface_mask > 0)
        cv2.putText(surface_vis, f"Surface: {surface_pixels}px", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(surface_vis, "Final surface mask", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        return surface_vis

    def _create_map_visualization(self):
        """Create spatial map visualization (same as simple_camera_stream.py)"""
        map_width = 640
        map_height = 480
        view_width_mm = 1000.0
        view_height_mm = 900.0

        scale_x = map_width / view_width_mm
        scale_y = map_height / view_height_mm

        map_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

        # Grid
        grid_spacing_mm = 100.0
        for x_mm in np.arange(-view_width_mm/2, view_width_mm/2, grid_spacing_mm):
            x_px = int((x_mm + view_width_mm/2) * scale_x)
            cv2.line(map_img, (x_px, 0), (x_px, map_height), (220, 220, 220), 1)
        for y_mm in np.arange(-view_height_mm/2, view_height_mm/2, grid_spacing_mm):
            y_px = int((y_mm + view_height_mm/2) * scale_y)
            cv2.line(map_img, (0, y_px), (map_width, y_px), (220, 220, 220), 1)

        # Axes
        center_x = map_width // 2
        center_y = map_height // 2
        cv2.line(map_img, (center_x, 0), (center_x, map_height), (180, 180, 180), 2)
        cv2.line(map_img, (0, center_y), (map_width, center_y), (180, 180, 180), 2)

        # Draw tracked markings
        skipped_count = 0
        with self.tracking_lock:
            for tracked in self.tracked_markings:
                map_x = int(center_x + tracked.x * scale_x)
                map_y = int(center_y - tracked.y * scale_y)

                if map_x < 0 or map_x >= map_width or map_y < 0 or map_y >= map_height:
                    skipped_count += 1
                    continue

                if tracked.observation_count >= 5:
                    color = (0, 200, 0)
                elif tracked.observation_count >= 2:
                    color = (0, 255, 255)
                else:
                    color = (0, 128, 255)

                if tracked.frames_absent > 0:
                    fade_factor = 1.0 - (tracked.frames_absent / self.forget_threshold)
                    color = tuple(int(c * fade_factor + 255 * (1 - fade_factor)) for c in color)

                radius = max(3, int(15 * scale_x))
                cv2.circle(map_img, (map_x, map_y), radius, color, -1)
                cv2.circle(map_img, (map_x, map_y), radius, (100, 100, 100), 1)

                if tracked.observation_count > 1:
                    cv2.putText(map_img, f"{tracked.observation_count}",
                              (map_x + radius + 2, map_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Car
        car_size = 30
        car_points = np.array([
            [center_x, center_y - car_size],
            [center_x - car_size//2, center_y + car_size//2],
            [center_x + car_size//2, center_y + car_size//2]
        ], np.int32)
        cv2.fillPoly(map_img, [car_points], (255, 100, 100))
        cv2.polylines(map_img, [car_points], True, (200, 0, 0), 2)

        # Info
        cv2.putText(map_img, f"Map View ({int(view_width_mm)}x{int(view_height_mm)}mm)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        with self.tracking_lock:
            active_count = len([t for t in self.tracked_markings if t.frames_absent == 0])
            total_count = len(self.tracked_markings)

            debug_y_offset = 80
            if total_count > 0:
                x_positions = [t.x for t in self.tracked_markings]
                y_positions = [t.y for t in self.tracked_markings]
                x_range = f"X:[{min(x_positions):.0f},{max(x_positions):.0f}]"
                y_range = f"Y:[{min(y_positions):.0f},{max(y_positions):.0f}]"
                cv2.putText(map_img, f"{x_range} {y_range}",
                           (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
                debug_y_offset += 20

        cv2.putText(map_img, f"Markings: {active_count} active / {total_count} total",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if skipped_count > 0:
            cv2.putText(map_img, f"({skipped_count} out of view)",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 0, 0), 1)

        cv2.putText(map_img, "Green=Established, Yellow=Medium, Orange=New",
                   (10, debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Scale
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
            # New JPEG frame
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


class StreamingHandler(server.BaseHTTPRequestHandler):
    """HTTP request handler for streaming"""

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
            # Get status from eraser system
            eraser = streaming_output.eraser_system
            if eraser:
                with eraser.state_lock:
                    state = eraser.state.value

                pose = eraser.localization.get_pose()
                progress = eraser.global_map.estimate_completion_progress()
                car_status = eraser.car_controller.get_status()

                status_data = {
                    'state': state,
                    'position': {
                        'x': int(pose.x),
                        'y': int(pose.y),
                        'theta': int(np.degrees(pose.theta))
                    },
                    'markings': {
                        'detected': progress['total_detected'],
                        'erased': progress['total_erased']
                    },
                    'progress': int(progress['progress_percent']),
                    'target': None,  # TODO: add current target
                    'speed': int(car_status.linear_velocity),
                    'runtime': int(time.time() - eraser.start_time)
                }
            else:
                status_data = {
                    'state': 'disconnected',
                    'position': {'x': 0, 'y': 0, 'theta': 0},
                    'markings': {'detected': 0, 'erased': 0},
                    'progress': 0,
                    'target': None,
                    'speed': 0,
                    'runtime': 0
                }

            content = json.dumps(status_data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self._stream_mjpeg(streaming_output.condition, lambda: streaming_output.frame)
        elif self.path == '/brightness.mjpg':
            self._stream_mjpeg(streaming_output.brightness_condition, lambda: streaming_output.brightness_frame)
        elif self.path == '/surface.mjpg':
            self._stream_mjpeg(streaming_output.surface_condition, lambda: streaming_output.surface_frame)
        elif self.path == '/map.mjpg':
            self._stream_mjpeg(streaming_output.map_condition, lambda: streaming_output.map_frame)
        else:
            self.send_error(404)
            self.end_headers()

    def _stream_mjpeg(self, condition, get_frame):
        """Helper to stream MJPEG"""
        self.send_response(200)
        self.send_header('Age', 0)
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        try:
            while True:
                with condition:
                    condition.wait()
                    frame = get_frame()
                if frame:
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
        except Exception as e:
            print(f'Removed streaming client {self.client_address}: {e}')

    def log_message(self, format, *args):
        """Suppress default logging"""
        return


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# Global output object (needed for HTTP handler)
streaming_output = None


def main():
    """Main entry point"""
    global streaming_output

    print("=" * 60)
    print("Autonomous Whiteboard Eraser with Live Streaming")
    print("=" * 60)

    if not CAMERA_AVAILABLE:
        print("ERROR: Camera not available!")
        return 1

    # Initialize eraser system
    config = EraserConfig()
    config.camera_width = CAMERA_WIDTH
    config.camera_height = CAMERA_HEIGHT
    eraser = WhiteboardEraserMain(config=config, debug=True)

    # Initialize camera
    print("\nInitializing camera...")
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
        controls={"FrameRate": 10}
    )
    picam2.configure(camera_config)

    # Create streaming output
    streaming_output = StreamingOutput(eraser)

    # Start camera
    print("Starting camera...")
    picam2.start_recording(MJPEGEncoder(), FileOutput(streaming_output))

    # Start detection thread
    streaming_output.start_detection()

    # Start HTTP server in background thread
    print(f"\nStarting HTTP server on port {HTTP_PORT}...")
    address = ('', HTTP_PORT)
    server = StreamingServer(address, StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"\n✓ Streaming at http://<pi-ip>:{HTTP_PORT}")
    print("✓ All systems ready!\n")

    # Initialize and run eraser system
    try:
        eraser.initialize()
        eraser.run()
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    finally:
        print("Stopping systems...")
        streaming_output.stop_detection()
        picam2.stop_recording()
        eraser.shutdown()
        server.shutdown()
        print("✓ Clean shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())