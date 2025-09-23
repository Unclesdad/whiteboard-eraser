#!/usr/bin/env python3
"""
Simple Marking Detection - Back to Basics
Focus on actual pen markings with minimal filtering
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Marking:
    """Represents a detected marking in camera coordinates"""
    x: float  # pixels from left
    y: float  # pixels from top
    area: float  # pixels squared
    confidence: float  # 0.0 to 1.0
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)

class SimpleMarkingDetector:
    """
    Simple marking detector - back to basics approach
    Finds the whiteboard top edge and detects markings below it
    """

    def __init__(self,
                 camera_height_mm: float = 135.0,
                 camera_angle_deg: float = 20.7,
                 fov_vertical_deg: float = 41.4,
                 fov_horizontal_deg: float = 62.0,
                 image_width: int = 640,
                 image_height: int = 480,
                 debug: bool = False):

        self.camera_height_mm = camera_height_mm
        self.camera_angle_rad = np.radians(camera_angle_deg)
        self.fov_vertical_rad = np.radians(fov_vertical_deg)
        self.fov_horizontal_rad = np.radians(fov_horizontal_deg)
        self.image_width = image_width
        self.image_height = image_height
        self.debug = debug

        # Simple detection parameters
        self.min_marking_area = 10  # Very permissive for small pen marks
        self.max_marking_area = 1000
        self.marking_threshold = 75  # Lower threshold to catch lighter pen marks
        self.gaussian_blur_size = 3

        # Whiteboard detection - much simpler
        self.whiteboard_threshold = 140  # Lower threshold for whiteboard detection
        self.whiteboard_area_threshold = 1000  # Minimum area for valid whiteboard surface
        self.top_edge_search_height = 200  # Only look in top 200 pixels for edge

        # Morphological operations
        self.erode_kernel = np.ones((2, 2), np.uint8)
        self.dilate_kernel = np.ones((3, 3), np.uint8)

        # Performance tracking
        self.processing_times = []

        # Debug info
        self.last_whiteboard_mask = None

        # Calculate pixel-to-mm conversion factors
        self._calculate_pixel_to_mm_factors()

        print(f"SimpleMarkingDetector initialized:")
        print(f"  Very permissive thresholds for maximum detection")

    def _calculate_pixel_to_mm_factors(self):
        """Calculate conversion factors from pixels to millimeters"""
        # Simplified version from original
        fov_half_vertical = self.fov_vertical_rad / 2

        angle_far = self.camera_angle_rad - fov_half_vertical
        if abs(angle_far) < 0.01:
            angle_far = 0.01
        far_distance = self.camera_height_mm / np.tan(angle_far)

        angle_near = self.camera_angle_rad + fov_half_vertical
        if abs(angle_near) < 0.01:
            angle_near = 0.01
        near_distance = self.camera_height_mm / np.tan(angle_near)

        total_vertical_mm = far_distance - near_distance
        horizontal_coverage_mm = 2 * self.camera_height_mm * np.tan(self.fov_horizontal_rad / 2)

        self.mm_per_pixel_x = horizontal_coverage_mm / self.image_width
        self.mm_per_pixel_y = total_vertical_mm / self.image_height
        self.near_distance_mm = near_distance
        self.far_distance_mm = far_distance

    def rotate_image_180(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 180 degrees to correct upside-down camera mounting"""
        return cv2.rotate(image, cv2.ROTATE_180)

    def find_white_surface(self, image: np.ndarray) -> np.ndarray:
        """
        Find the whiteboard surface using bottom-up detection
        Camera is ON the whiteboard, so surface appears at bottom of frame

        Returns:
            Binary mask where white pixels represent the whiteboard surface
        """
        height, width = image.shape[:2]

        # Convert to HSV for better white detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # More restrictive white detection - only very white areas
        lower_white = np.array([0, 0, 210])      # Very high brightness
        upper_white = np.array([179, 20, 255])   # Very low saturation

        # Create initial white mask
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Start from bottom edge and scan upward to find whiteboard surface
        surface_mask = np.zeros_like(white_mask, dtype=np.uint8)

        # For each column, find how far up the white surface extends from bottom
        for x in range(width):
            # Start from bottom row and go up
            for y in range(height - 1, -1, -1):
                if white_mask[y, x] > 0:
                    # Found white pixel - mark it as part of surface
                    surface_mask[y, x] = 255
                else:
                    # Hit non-white area - stop scanning up this column
                    break

        # Clean up the surface mask slightly
        kernel = np.ones((3, 3), np.uint8)
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, kernel)

        # Ensure the surface connects to the bottom edge
        # Fill any small gaps near the bottom
        bottom_rows = 20  # Look at bottom 20 pixels
        for y in range(height - bottom_rows, height):
            for x in range(width):
                if white_mask[y, x] > 0:
                    surface_mask[y, x] = 255

        return surface_mask

    def detect_markings(self, image: np.ndarray) -> List[Marking]:
        """
        Detect markings using white surface detection approach

        1. Find white drawing surface
        2. Look for dark spots only within that surface
        3. Minimal filtering for speed
        """
        start_time = time.time()

        # Correct camera orientation
        corrected = self.rotate_image_180(image)

        # Find white surface directly from color image
        white_surface_mask = self.find_white_surface(corrected)
        self.last_whiteboard_mask = white_surface_mask

        # Convert to grayscale for marking detection
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

        # Apply minimal blur for speed
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

        # Simple thresholding for dark markings (optimized threshold)
        _, binary = cv2.threshold(blurred, self.marking_threshold, 255, cv2.THRESH_BINARY_INV)

        # Apply white surface mask - only look within detected white area
        binary = cv2.bitwise_and(binary, white_surface_mask)

        # Light morphological cleanup
        cleaned = cv2.erode(binary, self.erode_kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, self.dilate_kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markings = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Very permissive area filtering
            if self.min_marking_area <= area <= self.max_marking_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center
                center_x = x + w / 2
                center_y = y + h / 2

                # Simple confidence based on area and position within white surface
                # Check if marking is well within the white surface
                if white_surface_mask[int(center_y), int(center_x)] > 0:
                    position_score = 1.0  # Inside white surface
                else:
                    position_score = 0.3  # Outside or on edge

                area_score = min(1.0, area / 100.0)  # Bigger = more confident
                confidence = (position_score * 0.7 + area_score * 0.3)

                marking = Marking(
                    x=center_x,
                    y=center_y,
                    area=area,
                    confidence=max(0.1, min(1.0, confidence)),  # Ensure some confidence
                    bbox=(x, y, w, h)
                )
                markings.append(marking)

        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        if self.debug:
            avg_time = np.mean(self.processing_times)
            print(f"  Detected {len(markings)} markings in {processing_time*1000:.1f}ms")
            white_area = np.sum(white_surface_mask) / 255.0  # Convert to pixel count
            print(f"  White surface area: {white_area:.0f} pixels")

        return markings

    def pixel_to_camera_relative_mm(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to millimeters relative to camera position"""
        center_x_px = self.image_width / 2
        center_y_px = self.image_height / 2

        dx_px = pixel_x - center_x_px
        dy_px = pixel_y - center_y_px

        x_mm = dx_px * self.mm_per_pixel_x

        y_fraction = pixel_y / self.image_height
        distance_from_camera = self.near_distance_mm + y_fraction * (self.far_distance_mm - self.near_distance_mm)
        y_mm = distance_from_camera

        return x_mm, y_mm

    def camera_relative_to_car_center_mm(self, camera_x_mm: float, camera_y_mm: float) -> Tuple[float, float]:
        """Convert camera-relative coordinates to car center coordinates"""
        car_center_offset_mm = 110
        car_x_mm = camera_x_mm
        car_y_mm = camera_y_mm - car_center_offset_mm
        return car_x_mm, car_y_mm

    def detect_and_convert_to_car_coordinates(self, image: np.ndarray) -> List[Tuple[float, float, float]]:
        """Detect markings and convert to car-relative coordinates"""
        markings = self.detect_markings(image)
        car_markings = []

        for marking in markings:
            cam_x, cam_y = self.pixel_to_camera_relative_mm(marking.x, marking.y)
            car_x, car_y = self.camera_relative_to_car_center_mm(cam_x, cam_y)
            car_markings.append((car_x, car_y, marking.confidence))

        return car_markings

    def visualize_detections(self, image: np.ndarray, markings: List[Marking]) -> np.ndarray:
        """Create debug visualization"""
        if not self.debug:
            return image

        # Correct image orientation
        vis_image = self.rotate_image_180(image.copy())

        # Draw white surface boundary
        if self.last_whiteboard_mask is not None:
            # Find contours of the white surface
            contours, _ = cv2.findContours(self.last_whiteboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Draw the boundary of the largest white surface
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(vis_image, [largest_contour], -1, (0, 255, 255), 2)  # Yellow boundary

            # Add text
            white_area = np.sum(self.last_whiteboard_mask) / 255.0
            cv2.putText(vis_image, f"White surface: {white_area:.0f}px",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw markings with simple color coding
        for i, marking in enumerate(markings):
            x, y, w, h = marking.bbox

            # Color based on confidence
            if marking.confidence > 0.6:
                color = (0, 255, 0)  # Green - high confidence
            elif marking.confidence > 0.3:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 128, 255)  # Orange - low confidence

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw center point
            center = (int(marking.x), int(marking.y))
            cv2.circle(vis_image, center, 3, (255, 0, 0), -1)

            # Add marking info
            text = f"M{i}: {marking.confidence:.2f}"
            cv2.putText(vis_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Convert to car coordinates
            cam_x, cam_y = self.pixel_to_camera_relative_mm(marking.x, marking.y)
            car_x, car_y = self.camera_relative_to_car_center_mm(cam_x, cam_y)
            coord_text = f"({car_x:.0f},{car_y:.0f}mm)"
            cv2.putText(vis_image, coord_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add performance info
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            perf_text = f"Simple: {avg_time*1000:.1f}ms ({fps:.1f} FPS)"
            cv2.putText(vis_image, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return vis_image

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {"avg_time_ms": 0, "fps": 0, "samples": 0}

        avg_time = np.mean(self.processing_times)
        return {
            "avg_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "samples": len(self.processing_times)
        }