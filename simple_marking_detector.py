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

            # Draw the contour with much thinner line for exclusion zone
            cv2.drawContours(boundary_mask, [largest_contour], -1, 255, thickness=5)  # Fixed 5-pixel thickness

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

        # Create edge exclusion mask to avoid detecting boundary artifacts as markings
        detection_mask = self._create_edge_exclusion_mask(white_surface_mask)

        # SIMPLE HOLE DETECTION APPROACH:
        # Find black holes directly within the white surface mask
        # The white surface already has holes where markings are - detect them directly

        # Method: Invert the white surface mask to find holes, then find contours
        # This preserves the exact holes that exist in the surface detection

        # Create inverse of white surface mask - holes become white blobs
        holes_mask = cv2.bitwise_not(white_surface_mask)

        # Only keep holes that are completely surrounded by white surface
        # Use very light morphological operations to preserve small markings
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel to preserve small holes
        holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise

        # Further filter: only keep holes that are INSIDE the detection mask (away from edges)
        # This ensures we don't detect edge artifacts as markings
        holes_mask = cv2.bitwise_and(holes_mask, detection_mask)

        if self.debug:
            white_area = np.sum(white_surface_mask) / 255.0
            detection_area = np.sum(detection_mask) / 255.0
            excluded_area = white_area - detection_area
            holes_found = np.sum(holes_mask) / 255.0
            print(f"  White surface: {white_area:.0f}px")
            print(f"  Detection area: {detection_area:.0f}px")
            print(f"  Excluded by edge: {excluded_area:.0f}px ({excluded_area/white_area*100:.1f}%)")
            print(f"  Hole pixels found: {holes_found:.0f}px")

        # Use the cleaned holes mask directly
        holes_cleaned = holes_mask

        # Find contours of the holes (these are our markings!)
        contours, _ = cv2.findContours(holes_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            holes_after_cleanup = np.sum(holes_cleaned) / 255.0
            print(f"  Holes after cleanup: {holes_after_cleanup:.0f}px")
            print(f"  Found {len(contours)} contours before filtering")
            if len(contours) == 0:
                # Check if the issue is in hole detection
                raw_holes = np.sum(holes_mask) / 255.0
                print(f"  Raw holes before cleanup: {raw_holes:.0f}px")
                white_pixels = np.sum(white_surface_mask) / 255.0
                print(f"  White surface pixels: {white_pixels:.0f}px")

                # Check specific conditions
                detection_pixels = np.sum(detection_mask > 0)
                surface_zero_pixels = np.sum(white_surface_mask == 0)
                overlap_pixels = np.sum((detection_mask > 0) & (white_surface_mask == 0))
                print(f"  Detection mask pixels: {detection_pixels}")
                print(f"  Surface zero pixels: {surface_zero_pixels}")
                print(f"  Overlap (should be holes): {overlap_pixels}")

        markings = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Simple area filtering - detect holes of reasonable size
            min_area = 1    # Very low minimum to catch small markings
            max_area = 5000 # Maximum area to avoid detecting large edge artifacts

            if self.debug:
                print(f"    Contour {len(markings)}: area={area:.1f}, range={min_area}-{max_area}")

            if min_area <= area <= max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w / 2
                center_y = y + h / 2

                # Check if marking center is within detection area (not too close to edges)
                if detection_mask[int(center_y), int(center_x)] > 0:
                    # Simple confidence based on area - larger holes are more likely to be actual markings
                    area_score = min(1.0, area / 200.0)  # Normalize to 0-1 scale
                    confidence = max(0.3, min(1.0, area_score))  # Ensure minimum confidence

                    marking = Marking(
                        x=center_x,
                        y=center_y,
                        area=area,
                        confidence=confidence,
                        bbox=(x, y, w, h)
                    )
                    markings.append(marking)
                elif self.debug:
                    print(f"    Skipped contour at ({center_x:.0f},{center_y:.0f}) - too close to edge")

        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        if self.debug:
            avg_time = np.mean(self.processing_times)
            print(f"  Detected {len(markings)} markings in {processing_time*1000:.1f}ms")
            white_area = np.sum(white_surface_mask) / 255.0
            holes_area = np.sum(holes_cleaned) / 255.0
            contour_count = len(contours)
            print(f"  Simple hole detection: White={white_area:.0f}px, Holes={holes_area:.0f}px, Contours={contour_count}")

            # Show actual areas of first few contours for debugging
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours[:5]]  # First 5 contours
                print(f"  First 5 hole areas: {areas}")

            # Show how many were filtered by edge detection
            edge_filtered = contour_count - len(markings)
            print(f"  Markings detected: {len(markings)}, Edge-filtered: {edge_filtered}")

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