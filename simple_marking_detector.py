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

        # Simple detection parameters - very permissive for debugging
        self.min_marking_area = 5   # Even smaller minimum
        self.max_marking_area = 2000 # Larger maximum
        self.marking_threshold = 120  # Much higher threshold - pen marks are darker than this
        self.gaussian_blur_size = 3

        # Edge exclusion parameters
        self.edge_exclusion_pixels = 15  # Exclude markings within 15 pixels of whiteboard edge

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
        Find the whiteboard surface using simple brightness thresholding
        1. Create white mask from brightness
        2. Find blob that touches bottom edge
        3. Return that blob as surface

        Returns:
            Binary mask where white pixels represent the whiteboard surface
        """
        height, width = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple brightness thresholding - pixels above threshold = white
        brightness_threshold = 180  # Adjust this value as needed
        _, white_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Clean up the mask slightly
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Use flood fill from bottom edge to find connected whiteboard area
        # This preserves holes (markings) unlike connected components
        bottom_row = height - 1
        surface_mask = np.zeros_like(white_mask, dtype=np.uint8)

        # Create a slightly larger image for flood fill (opencv requirement)
        h, w = white_mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        # Find all white pixels on the bottom edge
        bottom_white_pixels = []
        for x in range(width):
            if white_mask[bottom_row, x] > 0:
                bottom_white_pixels.append((x, bottom_row))

        if bottom_white_pixels:
            # Start flood fill from the first bottom white pixel
            start_x, start_y = bottom_white_pixels[0]

            # Flood fill to find all connected white areas
            # This preserves holes because we're working with the original thresholded mask
            cv2.floodFill(white_mask, flood_mask, (start_x, start_y), 128)  # Use gray value to mark flooded area

            # Create surface mask from flooded area, but preserve original brightness threshold
            # Areas marked as 128 are the connected whiteboard region
            flooded_area = (white_mask == 128)

            # Apply the flooded area to the ORIGINAL brightness threshold to preserve holes
            original_threshold = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
            surface_mask = np.where(flooded_area, original_threshold, 0).astype(np.uint8)

            # Restore the white_mask for consistency
            white_mask[white_mask == 128] = 255

        else:
            # Fallback: use bottom portion of image
            print(f"  No bottom-connected blob found, using fallback")
            fallback_start = int(height * 0.5)
            surface_mask[fallback_start:, :] = 255

        return surface_mask

    def _create_edge_exclusion_mask(self, white_surface_mask: np.ndarray) -> np.ndarray:
        """
        Create a mask that excludes only areas near the whiteboard boundary,
        preserving the full interior for marking detection.
        """
        # Start with the full white surface
        detection_mask = white_surface_mask.copy()

        # Find the contour of the whiteboard boundary
        contours, _ = cv2.findContours(white_surface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (main whiteboard boundary)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a mask for the boundary exclusion zone
            boundary_mask = np.zeros_like(white_surface_mask)

            # Draw the contour with thick line to create exclusion zone
            cv2.drawContours(boundary_mask, [largest_contour], -1, 255, thickness=self.edge_exclusion_pixels * 2)

            # Remove the boundary zone from the detection mask
            detection_mask = cv2.bitwise_and(detection_mask, cv2.bitwise_not(boundary_mask))

        return detection_mask

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

        # Debug: Print actual image dimensions being processed
        if self.debug:
            height, width = corrected.shape[:2]
            print(f"  Processing frame: {width}x{height}")

        # Find white surface directly from color image
        white_surface_mask = self.find_white_surface(corrected)
        self.last_whiteboard_mask = white_surface_mask

        # Create edge exclusion mask - only exclude areas near whiteboard boundary
        detection_mask = self._create_edge_exclusion_mask(white_surface_mask)

        # MULTI-SCALE APPROACH: Use different kernel sizes for different image regions
        # Bottom = close markings (large kernel), Middle = medium kernel, Top = distant markings (small kernel)
        height = detection_mask.shape[0]

        # Define regions
        top_boundary = int(height * 0.33)      # Top 33% of image
        bottom_boundary = int(height * 0.67)   # Bottom 33% of image

        # Create masks for each region using the edge-excluded detection mask
        top_mask = np.zeros_like(detection_mask)
        middle_mask = np.zeros_like(detection_mask)
        bottom_mask = np.zeros_like(detection_mask)

        top_mask[:top_boundary, :] = detection_mask[:top_boundary, :]
        middle_mask[top_boundary:bottom_boundary, :] = detection_mask[top_boundary:bottom_boundary, :]
        bottom_mask[bottom_boundary:, :] = detection_mask[bottom_boundary:, :]

        # Define kernels for each region
        small_kernel = np.ones((5, 5), np.uint8)    # Small kernel for distant markings (top)
        medium_kernel = np.ones((7, 7), np.uint8)   # Medium kernel for middle
        large_kernel = np.ones((13, 13), np.uint8)  # Large kernel for close markings (bottom)

        # Process each region separately
        holes_combined = np.zeros_like(detection_mask)

        # Top region (distant markings - small kernel)
        if np.any(top_mask):
            ideal_top = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, small_kernel)
            holes_top = cv2.subtract(ideal_top, top_mask)
            holes_combined[:top_boundary, :] = holes_top[:top_boundary, :]

        # Middle region (medium kernel)
        if np.any(middle_mask):
            ideal_middle = cv2.morphologyEx(middle_mask, cv2.MORPH_CLOSE, medium_kernel)
            holes_middle = cv2.subtract(ideal_middle, middle_mask)
            holes_combined[top_boundary:bottom_boundary, :] = holes_middle[top_boundary:bottom_boundary, :]

        # Bottom region (close markings - large kernel)
        if np.any(bottom_mask):
            ideal_bottom = cv2.morphologyEx(bottom_mask, cv2.MORPH_CLOSE, large_kernel)
            holes_bottom = cv2.subtract(ideal_bottom, bottom_mask)
            holes_combined[bottom_boundary:, :] = holes_bottom[bottom_boundary:, :]

        # Clean up combined hole detection
        cleanup_kernel = np.ones((3, 3), np.uint8)
        holes_cleaned = cv2.morphologyEx(holes_combined, cv2.MORPH_OPEN, cleanup_kernel)

        # Find contours of the holes (these are our markings!)
        contours, _ = cv2.findContours(holes_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markings = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Get bounding box to determine which region this marking is in
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h / 2

            # Distance-aware area thresholds based on image region
            if center_y < top_boundary:
                # Top region - distant markings, smaller area thresholds
                min_area = 2
                max_area = 300
            elif center_y < bottom_boundary:
                # Middle region - medium area thresholds
                min_area = 4
                max_area = 800
            else:
                # Bottom region - close markings, larger area thresholds
                min_area = 6
                max_area = 2000

            # Region-appropriate area filtering
            if min_area <= area <= max_area:
                # Calculate center
                center_x = x + w / 2
                center_y = y + h / 2

                # Simple confidence based on area and position within white surface
                # Check if marking is well within the inner (edge-excluded) detection area
                if detection_mask[int(center_y), int(center_x)] > 0:
                    position_score = 1.0  # Inside detection area (away from edges)
                else:
                    position_score = 0.0  # Too close to edge or outside - skip this marking
                    continue  # Skip markings too close to whiteboard edges

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
            white_area = np.sum(white_surface_mask) / 255.0
            holes_area = np.sum(holes_combined) / 255.0
            contour_count = len(contours)
            print(f"  Multi-scale detection: Top(5x5) Mid(7x7) Bot(13x13)")
            print(f"  White surface: {white_area:.0f}px, Holes: {holes_area:.0f}px, Contours: {contour_count}")

            # Debug individual contours by region
            top_detections = 0
            middle_detections = 0
            bottom_detections = 0
            total_filtered = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                center_y = y + h / 2

                # Determine region and thresholds
                if center_y < top_boundary:
                    min_area, max_area = 2, 300
                    if min_area <= area <= max_area:
                        top_detections += 1
                    else:
                        total_filtered += 1
                elif center_y < bottom_boundary:
                    min_area, max_area = 4, 800
                    if min_area <= area <= max_area:
                        middle_detections += 1
                    else:
                        total_filtered += 1
                else:
                    min_area, max_area = 6, 2000
                    if min_area <= area <= max_area:
                        bottom_detections += 1
                    else:
                        total_filtered += 1

            print(f"  Detections by region: Top:{top_detections} Mid:{middle_detections} Bot:{bottom_detections} Filtered:{total_filtered}")

            # Show actual areas of first few contours for debugging
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours[:5]]  # First 5 contours
                print(f"  First 5 hole areas: {areas}")

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

    def visualize_detections(self, image: np.ndarray, markings: List[Marking], whiteboard_mask: np.ndarray = None) -> np.ndarray:
        """Create debug visualization"""
        if not self.debug:
            return image

        # Correct image orientation
        vis_image = self.rotate_image_180(image.copy())

        # Draw white surface boundary - use provided mask or scale up the stored one
        mask_to_use = whiteboard_mask if whiteboard_mask is not None else self.last_whiteboard_mask

        if mask_to_use is not None:
            # If mask resolution doesn't match image, scale it up
            if mask_to_use.shape != vis_image.shape[:2]:
                mask_to_use = cv2.resize(mask_to_use, (vis_image.shape[1], vis_image.shape[0]))

            # Find contours of the white surface
            contours, _ = cv2.findContours(mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Draw the boundary of the largest white surface
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(vis_image, [largest_contour], -1, (0, 255, 255), 2)  # Yellow boundary

            # Add text
            white_area = np.sum(mask_to_use) / 255.0
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