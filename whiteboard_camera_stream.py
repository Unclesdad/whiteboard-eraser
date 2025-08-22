#!/usr/bin/env python3

import io
import socketserver
import time
from threading import Condition
from http import server
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import cv2
import numpy as np

# Import the whiteboard detector
from whiteboard_tracker4 import WhiteboardDetector

PAGE = """\
<html>
<head>
<title>Whiteboard Detection Stream</title>
</head>
<body>
<center><h1>Whiteboard Detection Live Stream</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
<center><p>Detecting whiteboard edges (red/green lines) and markings (cyan overlay)</p></center>
</body>
</html>
"""

class ProcessedStreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.detector = WhiteboardDetector(debug=False)
        self.last_process_time = 0
        self.min_frame_interval = 0.1  # 10 FPS max (100ms minimum between frames)
        
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame detected
            current_time = time.time()
            
            # Rate limiting - only process if enough time has passed
            if current_time - self.last_process_time < self.min_frame_interval:
                # Skip this frame, but still write the buffer for streaming continuity
                self.buffer.truncate()
                self.buffer.seek(0)
                return self.buffer.write(buf)
            
            # Get the image data from buffer before processing
            self.buffer.seek(0)
            current_buffer = self.buffer.getvalue()
            
            # Process the previous complete frame if we have one
            if current_buffer:
                try:
                    # Decode JPEG data to OpenCV format
                    nparr = np.frombuffer(current_buffer, np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame_bgr is not None:
                        # Process frame with whiteboard detection
                        processed_frame = self.process_frame(frame_bgr)
                        
                        # Encode processed frame back to JPEG
                        _, encoded_frame = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        processed_buffer = encoded_frame.tobytes()
                        
                        # Update the frame for streaming
                        with self.condition:
                            self.frame = processed_buffer
                            self.condition.notify_all()
                        
                        self.last_process_time = current_time
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    # Fall back to original frame
                    with self.condition:
                        self.frame = current_buffer
                        self.condition.notify_all()
            
            # Reset buffer for next frame
            self.buffer.truncate()
            self.buffer.seek(0)
            
        return self.buffer.write(buf)
    
    def process_frame(self, frame_bgr):
        """Process frame with whiteboard detection and overlay results"""
        try:
            # Run whiteboard detection
            detection_result = self.detector.detect_whiteboard_edges(frame_bgr)
            
            # Create overlay on the original frame
            result_frame = frame_bgr.copy()
            
            if detection_result:
                edge_lines, markings_mask, marking_regions = detection_result
                
                # Draw edge lines
                if edge_lines:
                    h, w = frame_bgr.shape[:2]
                    colors_bgr = [(0, 0, 255), (0, 255, 0)]  # Red, Green
                    
                    for i, (rho, theta) in enumerate(edge_lines):
                        color = colors_bgr[i % len(colors_bgr)]
                        
                        # Calculate line endpoints
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
                                intersections.append((w, int(y_right)))
                        
                        # Top edge (y = 0)
                        if abs(a) > 0.001:
                            x_top = rho / a
                            if 0 <= x_top <= w:
                                intersections.append((int(x_top), 0))
                        
                        # Bottom edge (y = h)
                        if abs(a) > 0.001:
                            x_bottom = (rho - h * b) / a
                            if 0 <= x_bottom <= w:
                                intersections.append((int(x_bottom), h))
                        
                        # Draw line if we have enough intersections
                        if len(intersections) >= 2:
                            cv2.line(result_frame, intersections[0], intersections[1], color, 3)
                
                # Draw markings overlay
                if markings_mask is not None and np.count_nonzero(markings_mask) > 0:
                    # Create cyan overlay for markings
                    marking_overlay = np.zeros_like(frame_bgr)
                    marking_overlay[markings_mask > 0] = [255, 255, 0]  # Cyan (BGR format)
                    result_frame = cv2.addWeighted(result_frame, 0.8, marking_overlay, 0.2, 0)
                
                # Draw marking regions as small circles
                if marking_regions:
                    for x, y in marking_regions:
                        cv2.circle(result_frame, (x, y), 4, (0, 255, 255), -1)  # Yellow circles
            
            return result_frame
            
        except Exception as e:
            print(f"Detection processing error: {e}")
            return frame_bgr  # Return original frame on error

class StreamingHandler(server.BaseHTTPRequestHandler):
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

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Initialize camera
print("Initializing camera and whiteboard detection...")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = ProcessedStreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

print("Camera and detection initialized!")

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Whiteboard detection camera stream starting...")
    print("Open your browser and go to: http://your_pi_ip:8000")
    print("Features:")
    print("- Real-time whiteboard edge detection (red/green lines)")
    print("- Dry-erase marker detection (cyan overlay)")
    print("- Marking regions (yellow circles)")
    print("- Max 10 FPS processing rate")
    print("Press Ctrl+C to stop")
    server.serve_forever()
finally:
    picam2.stop_recording()