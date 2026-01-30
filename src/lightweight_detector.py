#!/usr/bin/env python3
"""
Lightweight Marking Detector - No Streaming
Minimal power consumption version for battery operation
Runs CV loop and logs to terminal/file only
"""

import time
import cv2
import numpy as np
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from picamera2 import Picamera2
from dataclasses import dataclass
from simple_marking_detector import SimpleMarkingDetector

# ULTRA LOW POWER CONFIGURATION
CAMERA_WIDTH = 640    # Further reduced for battery operation
CAMERA_HEIGHT = 480   # Further reduced for battery operation
PROCESSING_WIDTH = 640   # Keep original processing resolution for better accuracy
PROCESSING_HEIGHT = 480  # Keep original processing resolution for better accuracy
TARGET_FPS = 5        # Lower FPS to reduce power draw
SAVE_FRAMES = False   # Set True to save annotated frames periodically
SAVE_INTERVAL = 30    # Save every N frames if enabled

# Image streaming mode
STREAM_IMAGE = "image" in sys.argv
STARTUP_IMAGE_PATH = "startup_detection.jpg"

@dataclass
class MarkingStats:
    """Simple stats tracking"""
    total_detections: int = 0
    frame_count: int = 0
    start_time: float = 0

    def fps(self):
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0

class StaticImageHandler(SimpleHTTPRequestHandler):
    """Minimal HTTP handler to serve one static image"""
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            # Serve simple HTML page with the image
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Startup Detection</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
            font-family: Arial, sans-serif;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            border: 2px solid #333;
            border-radius: 5px;
            background: white;
        }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Startup Detection Image</h1>
    <p>Single frame captured at startup with detection overlay</p>
    <img src="/{STARTUP_IMAGE_PATH}" alt="Detection">
</body>
</html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(html))
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == f'/{STARTUP_IMAGE_PATH}':
            # Serve the static image
            try:
                with open(STARTUP_IMAGE_PATH, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_error(404, "Image not found")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress HTTP server logs to keep console clean
        pass

def start_image_server():
    """start minimal HTTP server in background thread"""
    server = HTTPServer(('', 8000), StaticImageHandler)
    print(f"Image server started at http://localhost:8000")
    print(f"Serving startup image: {STARTUP_IMAGE_PATH}\n")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server

def main():
    print("=" * 60)
    print("LIGHTWEIGHT MARKING DETECTOR - Battery Optimized")
    print("=" * 60)
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Processing: {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Power mode: ULTRA LOW")
    if STREAM_IMAGE:
        print(f"Image mode: ENABLED - will serve startup image on port 8000")
    print("=" * 60)

    stats = MarkingStats(start_time=time.time())

    try:
        # Initialize camera with minimal power settings
        print("\nInitializing camera...")
        picam2 = Picamera2(camera_num=0)

        # Video configuration to match simple_camera_stream.py
        config = picam2.create_video_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
        )
        picam2.configure(config)

        # Camera controls matching simple_camera_stream.py
        controls = {
            "AeEnable": False,          # Disable auto-exposure
            "AwbEnable": True,          # Enable auto white balance for color consistency
            "ExposureTime": 8000,       # Reduced exposure time to save power
            "AnalogueGain": 2.5,        # Reduced gain to save power
            "Brightness": 0.725,        # High brightness (0.0 to 1.0)
            "Contrast": 2,              # Higher contrast for better marking distinction
        }
        picam2.set_controls(controls)
        print(f"Camera configured: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

        # Initialize detector
        print("\nInitializing detector...")
        detector = SimpleMarkingDetector(
            camera_height_mm=75.0,
            camera_angle_deg=20.5,
            image_width=PROCESSING_WIDTH,
            image_height=PROCESSING_HEIGHT,
            debug=False  # Disable debug to save processing
        )
        print("Detector initialized")

        # Start camera
        picam2.start()
        time.sleep(0.5)  # Camera warmup
        print("Camera started\n")

        # Capture and serve startup image if requested
        http_server = None
        if STREAM_IMAGE:
            print("Capturing startup image...")
            startup_frame = picam2.capture_array()

            # Convert to BGR
            if len(startup_frame.shape) == 3 and startup_frame.shape[2] == 4:
                startup_frame = cv2.cvtColor(startup_frame, cv2.COLOR_RGBA2BGR)
            elif len(startup_frame.shape) == 3 and startup_frame.shape[2] == 3:
                startup_frame = cv2.cvtColor(startup_frame, cv2.COLOR_RGB2BGR)

            # Resize to processing resolution
            if startup_frame.shape[1] != PROCESSING_WIDTH or startup_frame.shape[0] != PROCESSING_HEIGHT:
                startup_frame = cv2.resize(startup_frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))

            # Run detection on startup frame
            startup_markings = detector.detect_markings(startup_frame)

            # Create annotated frame
            annotated_startup = detector.visualize_detections(startup_frame, startup_markings)

            # Save annotated image
            cv2.imwrite(STARTUP_IMAGE_PATH, annotated_startup)
            print(f"Saved startup image: {STARTUP_IMAGE_PATH} ({len(startup_markings)} markings)")

            # Start HTTP server
            http_server = start_image_server()

        print("=" * 60)
        print("DETECTION LOOP RUNNING")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")

        frame_times = []
        detection_times = []

        while True:
            loop_start = time.time()

            # Capture frame
            frame = picam2.capture_array()

            # Convert from RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Resize to processing resolution
            if frame.shape[1] != PROCESSING_WIDTH or frame.shape[0] != PROCESSING_HEIGHT:
                frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))

            # Run detection
            det_start = time.time()
            markings = detector.detect_markings(frame)
            detection_time = time.time() - det_start
            detection_times.append(detection_time)

            # Update stats
            stats.frame_count += 1
            stats.total_detections += len(markings)

            # Track frame time
            frame_time = time.time() - loop_start
            frame_times.append(frame_time)

            # Keep only recent times for averaging
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(detection_times) > 30:
                detection_times.pop(0)

            # Print stats every 10 frames
            if stats.frame_count % 10 == 0:
                avg_det_time = np.mean(detection_times) * 1000  # ms
                avg_frame_time = np.mean(frame_times) * 1000    # ms
                current_fps = stats.fps()

                print(f"Frame {stats.frame_count:4d} | "
                      f"Markings: {len(markings):2d} | "
                      f"Det: {avg_det_time:5.1f}ms | "
                      f"Total: {avg_frame_time:5.1f}ms | "
                      f"FPS: {current_fps:4.1f}")

            # Optional: Save annotated frames periodically
            if SAVE_FRAMES and stats.frame_count % SAVE_INTERVAL == 0:
                annotated = detector.visualize_detections(frame, markings)
                filename = f"detection_{stats.frame_count:06d}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"  Saved {filename}")

            # Rate limiting to target FPS
            elapsed = time.time() - loop_start
            target_time = 1.0 / TARGET_FPS
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("SHUTTING DOWN")
        print("=" * 60)

        # Final stats
        runtime = time.time() - stats.start_time
        avg_fps = stats.fps()
        avg_detections_per_frame = stats.total_detections / stats.frame_count if stats.frame_count > 0 else 0

        print(f"\nFinal Statistics:")
        print(f"  Runtime:        {runtime:.1f}s")
        print(f"  Total frames:   {stats.frame_count}")
        print(f"  Average FPS:    {avg_fps:.2f}")
        print(f"  Total markings: {stats.total_detections}")
        print(f"  Avg per frame:  {avg_detections_per_frame:.2f}")

        if frame_times:
            print(f"\nPerformance:")
            print(f"  Avg frame time:     {np.mean(frame_times)*1000:.1f}ms")
            print(f"  Avg detection time: {np.mean(detection_times)*1000:.1f}ms")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nCleaning up...")
        if 'picam2' in locals():
            picam2.stop()
        print("Complete")

if __name__ == "__main__":
    main()
