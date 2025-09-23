#!/usr/bin/env python3
"""
Camera Stream with Marking Detection Overlay
Broadcasts camera feed with real-time marking detection overlays
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

# Import our marking detector
from marking_detector import MarkingDetector

PAGE = """\
<html>
<head>
<title>Whiteboard Eraser - Camera with Marking Detection</title>
<style>
body { font-family: Arial, sans-serif; background-color: #f0f0f0; }
.container { text-align: center; margin: 20px; }
.stats { background-color: white; padding: 10px; margin: 10px auto; border-radius: 5px; width: 640px; }
.video { border: 2px solid #333; border-radius: 5px; }
</style>
</head>
<body>
<div class="container">
<h1>Whiteboard Eraser Car - Live Detection Feed</h1>
<div class="stats">
<p><strong>Status:</strong> <span id="status">Loading...</span></p>
<p><strong>Markings Detected:</strong> <span id="markings">0</span></p>
<p><strong>Detection FPS:</strong> <span id="fps">0.0</span></p>
</div>
<img src="stream.mjpg" width="640" height="480" class="video">
<div class="stats">
<p><strong>Controls:</strong> Refresh page to restart | Press Ctrl+C in terminal to stop</p>
<p><strong>Detection Info:</strong> Green boxes = high confidence markings, Yellow boxes = low confidence</p>
</div>
</div>
<script>
// Auto-refresh status every 2 seconds
setInterval(function() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('status').textContent = data.status;
            document.getElementById('markings').textContent = data.markings;
            document.getElementById('fps').textContent = data.fps.toFixed(1);
        })
        .catch(err => {
            document.getElementById('status').textContent = 'Connection Error';
        });
}, 2000);
</script>
</body>
</html>
"""

class DetectionStreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

        # Detection state
        self.detector = MarkingDetector(debug=False)  # No debug windows for streaming
        self.detection_thread = None
        self.detection_running = False
        self.current_markings = []
        self.detection_fps = 0.0
        self.frame_count = 0
        self.last_stats_time = time.time()

        # Raw frame buffer for processing
        self.raw_frame = None
        self.raw_condition = Condition()

        print("✓ Detection system initialized")

    def start_detection(self):
        """Start the detection processing thread"""
        self.detection_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        print("✓ Detection thread started")

    def stop_detection(self):
        """Stop the detection processing"""
        self.detection_running = False
        if self.detection_thread:
            self.detection_thread.join()
        print("✓ Detection thread stopped")

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

                # Run detection
                start_time = time.time()
                markings = self.detector.detect_markings(frame)
                detection_time = time.time() - start_time

                # Update stats
                detection_times.append(detection_time)
                if len(detection_times) > 30:  # Keep last 30 samples
                    detection_times.pop(0)

                self.detection_fps = 1.0 / np.mean(detection_times) if detection_times else 0.0
                self.current_markings = markings

                # Create annotated frame
                annotated_frame = self._create_annotated_frame(frame, markings)

                # Convert to JPEG
                _, jpeg_data = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Update streaming buffer
                with self.condition:
                    self.frame = jpeg_data.tobytes()
                    self.condition.notify_all()

                # Small delay to prevent overwhelming the system
                time.sleep(0.05)  # ~20 FPS max detection rate

            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(0.1)

    def _create_annotated_frame(self, frame, markings):
        """Create frame with marking detection overlays"""
        # Correct camera orientation (upside down)
        annotated = self.detector.rotate_image_180(frame.copy())

        # Draw markings
        for i, marking in enumerate(markings):
            x, y, w, h = marking.bbox

            # Choose color based on confidence
            if marking.confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
                thickness = 2
            elif marking.confidence > 0.4:
                color = (0, 255, 255)  # Yellow for medium confidence
                thickness = 2
            else:
                color = (0, 128, 255)  # Orange for low confidence
                thickness = 1

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

            # Draw center point
            center = (int(marking.x), int(marking.y))
            cv2.circle(annotated, center, 3, (255, 0, 0), -1)

            # Add marking info
            text = f"M{i}: {marking.confidence:.2f}"
            cv2.putText(annotated, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Convert to car coordinates and display
            cam_x, cam_y = self.detector.pixel_to_camera_relative_mm(marking.x, marking.y)
            car_x, car_y = self.detector.camera_relative_to_car_center_mm(cam_x, cam_y)
            coord_text = f"({car_x:.0f},{car_y:.0f}mm)"
            cv2.putText(annotated, coord_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add performance overlay
        fps_text = f"Detection: {self.detection_fps:.1f} FPS | Markings: {len(markings)}"
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(annotated, timestamp, (10, annotated.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def write(self, buf):
        """Handle incoming camera data"""
        if buf.startswith(b'\xff\xd8'):
            # New JPEG frame - decode for processing
            try:
                # Decode JPEG to numpy array for detection
                nparr = np.frombuffer(self.buffer.getvalue(), np.uint8)
                if len(nparr) > 0:
                    decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if decoded_frame is not None:
                        with self.raw_condition:
                            self.raw_frame = decoded_frame
                            self.raw_condition.notify()
            except Exception as e:
                print(f"Frame decode error: {e}")

            # Reset buffer for new frame
            self.buffer.truncate()
            self.buffer.seek(0)

        return self.buffer.write(buf)

class DetectionStreamingHandler(server.BaseHTTPRequestHandler):
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
            # JSON status for web interface
            status_data = {
                'status': 'Running' if output.detection_running else 'Stopped',
                'markings': len(output.current_markings),
                'fps': output.detection_fps
            }
            import json
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
        else:
            self.send_error(404)
            self.end_headers()

class DetectionStreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def main():
    global output, picam2

    print("🔧 Initializing Whiteboard Eraser Detection Stream...")

    try:
        # Initialize camera
        print("📷 Starting camera...")
        picam2 = Picamera2(camera_num=0)
        picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))

        # Initialize detection output
        output = DetectionStreamingOutput()

        # Start camera recording
        picam2.start_recording(MJPEGEncoder(), FileOutput(output))
        print("✓ Camera recording started")

        # Start detection processing
        output.start_detection()

        # Start web server
        address = ('', 8000)
        server = DetectionStreamingServer(address, DetectionStreamingHandler)

        print("🌐 Detection stream server starting...")
        print(f"📺 Open your browser and go to: http://localhost:8000")
        print("🎯 Real-time marking detection overlay active")
        print("⚡ Press Ctrl+C to stop")
        print()

        server.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if 'output' in globals():
            output.stop_detection()
        if 'picam2' in globals():
            picam2.stop_recording()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()