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
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# Import our simple marking detector
from simple_marking_detector import SimpleMarkingDetector

PAGE = """\
<html>
<head>
<title>Simple Whiteboard Detection Test</title>
<style>
body { font-family: Arial, sans-serif; background-color: #f0f0f0; }
.container { text-align: center; margin: 20px; }
.stats { background-color: white; padding: 10px; margin: 10px auto; border-radius: 5px; width: 1950px; }
.video-container { display: flex; justify-content: center; gap: 15px; margin: 20px 0; }
.video { border: 2px solid #333; border-radius: 5px; }
.video-label { text-align: center; margin-top: 10px; font-weight: bold; }
</style>
</head>
<body>
<div class="container">
<h1>Simple Whiteboard Detection Test - Three Panel View</h1>
<div class="stats">
<p><strong>Approach:</strong> Brightness threshold → Find bottom blob → Detect dark spots</p>
<p><strong>Markings:</strong> <span id="markings">0</span> | <strong>FPS:</strong> <span id="fps">0.0</span></p>
</div>
<div class="video-container">
<div>
<img src="stream.mjpg" width="640" height="480" class="video">
<div class="video-label">1. Annotated Camera Feed</div>
</div>
<div>
<img src="brightness.mjpg" width="640" height="480" class="video">
<div class="video-label">2. Raw Brightness Mask</div>
</div>
<div>
<img src="surface.mjpg" width="640" height="480" class="video">
<div class="video-label">3. Final Surface Blob</div>
</div>
</div>
<div class="stats">
<p><strong>Panel 1:</strong> Yellow boundary = detected surface, Colored boxes = markings</p>
<p><strong>Panel 2:</strong> Raw brightness thresholding (all pixels above threshold)</p>
<p><strong>Panel 3:</strong> Bottom-connected blob selected as whiteboard surface</p>
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

class SimpleDetectionOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.brightness_frame = None
        self.surface_frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.brightness_condition = Condition()
        self.surface_condition = Condition()

        # Simple detection system
        self.detector = SimpleMarkingDetector(debug=True)
        self.detection_thread = None
        self.detection_running = False
        self.current_markings = []
        self.detection_fps = 0.0

        # Raw frame buffer
        self.raw_frame = None
        self.raw_condition = Condition()

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

                # Run simple detection
                start_time = time.time()
                markings = self.detector.detect_markings(frame)
                detection_time = time.time() - start_time

                # Update stats
                detection_times.append(detection_time)
                if len(detection_times) > 30:
                    detection_times.pop(0)

                self.detection_fps = 1.0 / np.mean(detection_times) if detection_times else 0.0
                self.current_markings = markings

                # Create annotated frame
                annotated_frame = self.detector.visualize_detections(frame, markings)

                # Create brightness mask visualization
                brightness_frame = self._create_brightness_visualization(frame)

                # Create final surface visualization
                surface_frame = self._create_surface_visualization(frame)

                # Convert all three to JPEG
                _, jpeg_data = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, brightness_jpeg_data = cv2.imencode('.jpg', brightness_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, surface_jpeg_data = cv2.imencode('.jpg', surface_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

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

                # Small delay
                time.sleep(0.05)

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
        picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))

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