#!/usr/bin/env python3
"""
Lightweight Marking Detector - No Streaming
Minimal power consumption version for battery operation
Runs CV loop and logs to terminal/file only
"""

import time
import cv2
import numpy as np
from picamera2 import Picamera2
from dataclasses import dataclass
from simple_marking_detector import SimpleMarkingDetector

# ULTRA LOW POWER CONFIGURATION
CAMERA_WIDTH = 640    # Further reduced for battery operation
CAMERA_HEIGHT = 480   # Further reduced for battery operation
PROCESSING_WIDTH = 320   # Minimal processing resolution
PROCESSING_HEIGHT = 240  # Minimal processing resolution
TARGET_FPS = 5        # Lower FPS to reduce power draw
SAVE_FRAMES = False   # Set True to save annotated frames periodically
SAVE_INTERVAL = 30    # Save every N frames if enabled

@dataclass
class MarkingStats:
    """Simple stats tracking"""
    total_detections: int = 0
    frame_count: int = 0
    start_time: float = 0

    def fps(self):
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0

def main():
    print("=" * 60)
    print("🔋 LIGHTWEIGHT MARKING DETECTOR - Battery Optimized")
    print("=" * 60)
    print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Processing: {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Power mode: ULTRA LOW")
    print("=" * 60)

    stats = MarkingStats(start_time=time.time())

    try:
        # Initialize camera with minimal power settings
        print("\n📷 Initializing camera...")
        picam2 = Picamera2(camera_num=0)

        # Minimal configuration
        config = picam2.create_still_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
            buffer_count=2  # Minimal buffering
        )
        picam2.configure(config)

        # Conservative camera controls to minimize power
        controls = {
            "AeEnable": False,       # Disable auto-exposure
            "AwbEnable": False,      # Disable auto white balance (saves power)
            "ExposureTime": 10000,   # Fixed exposure
            "AnalogueGain": 2.0,     # Fixed gain
            "Brightness": 0.6,       # Moderate brightness
        }
        picam2.set_controls(controls)
        print(f"✓ Camera configured with low-power settings")

        # Initialize detector
        print("\n🔍 Initializing detector...")
        detector = SimpleMarkingDetector(
            camera_height_mm=75.0,
            camera_angle_deg=20.5,
            image_width=PROCESSING_WIDTH,
            image_height=PROCESSING_HEIGHT,
            debug=False  # Disable debug to save processing
        )
        print("✓ Detector initialized")

        # Start camera
        picam2.start()
        time.sleep(0.5)  # Camera warmup
        print("✓ Camera started\n")

        print("=" * 60)
        print("📊 DETECTION LOOP RUNNING")
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
                print(f"  💾 Saved {filename}")

            # Rate limiting to target FPS
            elapsed = time.time() - loop_start
            target_time = 1.0 / TARGET_FPS
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 SHUTTING DOWN")
        print("=" * 60)

        # Final stats
        runtime = time.time() - stats.start_time
        avg_fps = stats.fps()
        avg_detections_per_frame = stats.total_detections / stats.frame_count if stats.frame_count > 0 else 0

        print(f"\n📊 Final Statistics:")
        print(f"  Runtime:        {runtime:.1f}s")
        print(f"  Total frames:   {stats.frame_count}")
        print(f"  Average FPS:    {avg_fps:.2f}")
        print(f"  Total markings: {stats.total_detections}")
        print(f"  Avg per frame:  {avg_detections_per_frame:.2f}")

        if frame_times:
            print(f"\n⚡ Performance:")
            print(f"  Avg frame time:     {np.mean(frame_times)*1000:.1f}ms")
            print(f"  Avg detection time: {np.mean(detection_times)*1000:.1f}ms")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n🔌 Cleaning up...")
        if 'picam2' in locals():
            picam2.stop()
        print("✓ Complete")

if __name__ == "__main__":
    main()
