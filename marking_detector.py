#!/usr/bin/env python3
"""
Efficient Marking Detection Module for Whiteboard Eraser Car
Optimized for Raspberry Pi 5 with upside-down camera handling
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

class MarkingDetector:
    """
    Efficient marking detector optimized for RPi5 performance
    Handles upside-down camera mounting and converts to real-world coordinates
    """

    def __init__(self,
                 camera_height_mm: float = 135.0,  # 13.5cm above whiteboard
                 camera_angle_deg: float = 20.7,   # Camera tilt angle
                 fov_vertical_deg: float = 41.4,   # Camera vertical FOV
                 fov_horizontal_deg: float = 62.0, # Camera horizontal FOV
                 image_width: int = 640,
                 image_height: int = 480,
                 debug: bool = False):
        """
        Initialize marking detector

        Args:
            camera_height_mm: Distance from camera to whiteboard surface
            camera_angle_deg: Camera tilt angle (down from horizontal)
            fov_vertical_deg: Camera vertical field of view
            fov_horizontal_deg: Camera horizontal field of view
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
            debug: Enable debug visualization
        """
        self.camera_height_mm = camera_height_mm
        self.camera_angle_rad = np.radians(camera_angle_deg)
        self.fov_vertical_rad = np.radians(fov_vertical_deg)
        self.fov_horizontal_rad = np.radians(fov_horizontal_deg)
        self.image_width = image_width
        self.image_height = image_height
        self.debug = debug

        # Detection parameters optimized for speed
        self.min_marking_area = 20  # Minimum area in pixels
        self.max_marking_area = 500  # Maximum area in pixels
        self.marking_threshold = 80  # Darkness threshold for markings
        self.gaussian_blur_size = 3  # Small blur for noise reduction

        # Whiteboard surface detection parameters
        self.whiteboard_brightness_threshold = 150  # Lowered minimum brightness for whiteboard surface
        self.whiteboard_area_threshold = 1000  # Minimum whiteboard area in pixels
        self.edge_exclusion_pixels = 8  # Reduced exclusion distance from whiteboard edges
        self.contrast_threshold = 20  # Lowered minimum local contrast against white background

        # Edge detection parameters for whiteboard boundary
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.hough_threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 10

        # Morphological operations kernels
        self.erode_kernel = np.ones((2, 2), np.uint8)
        self.dilate_kernel = np.ones((3, 3), np.uint8)
        self.whiteboard_erode_kernel = np.ones((3, 3), np.uint8)
        self.whiteboard_dilate_kernel = np.ones((5, 5), np.uint8)

        # Performance tracking
        self.processing_times = []

        # Debug information (stored for visualization)
        self.last_detected_edges = []
        self.last_whiteboard_mask = None

        # Calculate pixel-to-mm conversion factors
        self._calculate_pixel_to_mm_factors()

        print(f"MarkingDetector initialized:")
        print(f"  Camera: {image_width}x{image_height}, {camera_height_mm}mm height")
        print(f"  FOV: {fov_horizontal_deg:.1f}° x {fov_vertical_deg:.1f}°")
        print(f"  Detection thresholds: area {self.min_marking_area}-{self.max_marking_area}px")

    def _calculate_pixel_to_mm_factors(self):
        """Calculate conversion factors from pixels to real-world millimeters"""
        # Calculate the ground plane coverage at camera height
        # Using camera tilt angle and FOV to determine real-world dimensions

        # Vertical coverage on ground (in direction camera is pointing)
        fov_half_vertical = self.fov_vertical_rad / 2

        # Distance from directly below camera to far edge of view
        angle_far = self.camera_angle_rad - fov_half_vertical
        if abs(angle_far) < 0.01:  # Avoid divide by zero
            angle_far = 0.01
        far_distance = self.camera_height_mm / np.tan(angle_far)

        # Distance from directly below camera to near edge of view
        angle_near = self.camera_angle_rad + fov_half_vertical
        if abs(angle_near) < 0.01:  # Avoid divide by zero
            angle_near = 0.01
        near_distance = self.camera_height_mm / np.tan(angle_near)

        # Total vertical coverage
        total_vertical_mm = far_distance - near_distance

        # Horizontal coverage at camera height
        horizontal_coverage_mm = 2 * self.camera_height_mm * np.tan(self.fov_horizontal_rad / 2)

        # Conversion factors
        self.mm_per_pixel_x = horizontal_coverage_mm / self.image_width
        self.mm_per_pixel_y = total_vertical_mm / self.image_height

        # Store distances for coordinate conversion
        self.near_distance_mm = near_distance
        self.far_distance_mm = far_distance

        if self.debug:
            print(f"  Pixel conversion: {self.mm_per_pixel_x:.2f} mm/px (x), {self.mm_per_pixel_y:.2f} mm/px (y)")
            print(f"  Ground coverage: {horizontal_coverage_mm:.1f}mm x {total_vertical_mm:.1f}mm")

    def rotate_image_180(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 180 degrees to correct upside-down camera mounting"""
        return cv2.rotate(image, cv2.ROTATE_180)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Efficient preprocessing for marking detection

        Args:
            image: Input BGR image from camera

        Returns:
            Preprocessed grayscale image ready for detection
        """
        # Correct camera orientation first
        corrected = self.rotate_image_180(image)

        # Convert to grayscale
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

        # Apply small Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)

        return blurred

    def _detect_whiteboard_edges(self, gray_image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect whiteboard edges/boundaries to define the whiteboard surface area

        Args:
            gray_image: Preprocessed grayscale image

        Returns:
            Tuple of (whiteboard_mask, detected_lines) where mask represents area below edges
        """
        # Apply edge detection
        edges = cv2.Canny(gray_image, self.canny_low_threshold, self.canny_high_threshold)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                               threshold=self.hough_threshold,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)

        # Create mask for whiteboard surface
        mask = np.ones_like(gray_image, dtype=np.uint8) * 255
        detected_lines = []

        if lines is not None:
            # Process detected lines to find whiteboard boundaries
            valid_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                # Filter lines that could be whiteboard edges
                # Accept lines with reasonable angles (not too vertical/horizontal)
                if abs(angle) > 10 and abs(angle) < 80:
                    valid_lines.append((x1, y1, x2, y2))

            # If we found valid edge lines, create mask below them
            if valid_lines:
                detected_lines = valid_lines

                # Create mask that includes area below the detected edges
                mask = np.zeros_like(gray_image, dtype=np.uint8)

                # For each valid line, determine the "below" area
                for x1, y1, x2, y2 in valid_lines:
                    # Create a line mask
                    line_mask = np.zeros_like(gray_image, dtype=np.uint8)
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, self.edge_exclusion_pixels)

                    # Fill area below the line (whiteboard surface)
                    # Find the bottom-most y coordinate of the line
                    line_y = max(y1, y2)

                    # Fill rectangle below this line
                    cv2.rectangle(mask, (0, line_y), (gray_image.shape[1], gray_image.shape[0]), 255, -1)

                    # Subtract the line itself to avoid detecting it as a marking
                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(line_mask))

        # If no edges found or mask is empty, fall back to brightness-based detection
        if np.sum(mask) < self.whiteboard_area_threshold:
            # Fallback: use brightness threshold
            _, mask = cv2.threshold(gray_image, self.whiteboard_brightness_threshold, 255, cv2.THRESH_BINARY)

            # Clean up the mask
            mask = cv2.erode(mask, self.whiteboard_erode_kernel, iterations=1)
            mask = cv2.dilate(mask, self.whiteboard_dilate_kernel, iterations=1)

        return mask, detected_lines

    def _calculate_local_contrast(self, gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """
        Calculate local contrast around a detection to ensure it's actually a marking on white surface

        Args:
            gray_image: Grayscale image
            x, y, w, h: Bounding box of the detection

        Returns:
            Local contrast value (higher = better marking candidate)
        """
        # Expand region to get surrounding context
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray_image.shape[1], x + w + pad)
        y2 = min(gray_image.shape[0], y + h + pad)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Get the region around the detection
        region = gray_image[y1:y2, x1:x2]

        # Get the actual detection area
        det_x1 = x - x1
        det_y1 = y - y1
        det_x2 = det_x1 + w
        det_y2 = det_y1 + h

        if det_x2 <= det_x1 or det_y2 <= det_y1:
            return 0.0

        detection_area = region[det_y1:det_y2, det_x1:det_x2]

        # Calculate average brightness of detection vs surrounding area
        if detection_area.size == 0:
            return 0.0

        detection_brightness = np.mean(detection_area)

        # Create mask for surrounding area (exclude the detection itself)
        surrounding_mask = np.ones_like(region, dtype=bool)
        surrounding_mask[det_y1:det_y2, det_x1:det_x2] = False

        if np.sum(surrounding_mask) == 0:
            return 0.0

        surrounding_brightness = np.mean(region[surrounding_mask])

        # Calculate contrast (surrounding should be brighter than detection for markings on white)
        contrast = surrounding_brightness - detection_brightness
        return max(0.0, contrast)

    def detect_markings(self, image: np.ndarray) -> List[Marking]:
        """
        Detect dark markings on white background using two-stage approach

        Args:
            image: Input BGR image from camera

        Returns:
            List of detected markings on whiteboard surface
        """
        start_time = time.time()

        # Stage 1: Preprocess image and detect whiteboard edges/surface
        gray = self.preprocess_image(image)
        whiteboard_mask, detected_edges = self._detect_whiteboard_edges(gray)

        # Store debug information
        self.last_detected_edges = detected_edges
        self.last_whiteboard_mask = whiteboard_mask

        # Stage 2: Detect markings only within whiteboard area
        # Create binary mask for dark markings
        # Invert so markings become white on black background
        _, binary = cv2.threshold(gray, self.marking_threshold, 255, cv2.THRESH_BINARY_INV)

        # Apply whiteboard mask - only look for markings on whiteboard surface
        binary = cv2.bitwise_and(binary, whiteboard_mask)

        # Clean up with morphological operations
        cleaned = cv2.erode(binary, self.erode_kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, self.dilate_kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markings = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if self.min_marking_area <= area <= self.max_marking_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate local contrast to ensure it's actually a marking on white surface
                local_contrast = self._calculate_local_contrast(gray, x, y, w, h)

                # Skip if contrast is too low (not a real marking on white surface)
                if local_contrast < self.contrast_threshold:
                    continue

                # Calculate center
                center_x = x + w / 2
                center_y = y + h / 2

                # Calculate enhanced confidence with whiteboard context
                confidence = self._calculate_confidence_with_context(contour, area, local_contrast, whiteboard_mask, x, y, w, h)

                marking = Marking(
                    x=center_x,
                    y=center_y,
                    area=area,
                    confidence=confidence,
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
            print(f"  Detected {len(markings)} markings in {processing_time*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")

        return markings

    def _calculate_confidence(self, contour: np.ndarray, area: float) -> float:
        """Calculate confidence score for a detected marking (legacy method)"""
        # Base confidence on area (prefer medium-sized markings)
        ideal_area = (self.min_marking_area + self.max_marking_area) / 2
        area_score = 1.0 - abs(area - ideal_area) / ideal_area

        # Factor in shape compactness (prefer more circular shapes)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            shape_score = min(compactness, 1.0)
        else:
            shape_score = 0.0

        # Combine scores
        confidence = (area_score * 0.7 + shape_score * 0.3)
        return max(0.0, min(1.0, confidence))

    def _calculate_confidence_with_context(self, contour: np.ndarray, area: float, local_contrast: float,
                                         whiteboard_mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """
        Calculate enhanced confidence score considering whiteboard context

        Args:
            contour: Detection contour
            area: Detection area in pixels
            local_contrast: Local contrast value
            whiteboard_mask: Whiteboard surface mask
            x, y, w, h: Bounding box coordinates

        Returns:
            Enhanced confidence score (0.0 to 1.0)
        """
        # Start with basic shape and area confidence
        base_confidence = self._calculate_confidence(contour, area)

        # Contrast score (normalize to 0-1, with optimal contrast around 60-80)
        optimal_contrast = 70.0
        contrast_score = 1.0 - abs(local_contrast - optimal_contrast) / optimal_contrast
        contrast_score = max(0.0, min(1.0, contrast_score))

        # Distance from whiteboard edge score
        # Check how far the detection is from the edge of the whiteboard mask
        center_x = x + w // 2
        center_y = y + h // 2

        # Create a kernel to check distance from edges
        distance_kernel = np.ones((self.edge_exclusion_pixels * 2, self.edge_exclusion_pixels * 2), np.uint8)
        eroded_mask = cv2.erode(whiteboard_mask, distance_kernel, iterations=1)

        edge_distance_score = 1.0
        if center_y < whiteboard_mask.shape[0] and center_x < whiteboard_mask.shape[1]:
            if eroded_mask[center_y, center_x] == 0:
                # Too close to edge, reduce confidence
                edge_distance_score = 0.3
            elif whiteboard_mask[center_y, center_x] == 0:
                # Outside whiteboard entirely
                edge_distance_score = 0.0

        # Size consistency score (markings should be reasonably sized relative to image)
        image_area = whiteboard_mask.shape[0] * whiteboard_mask.shape[1]
        relative_size = area / image_area
        if relative_size > 0.01:  # Too large (more than 1% of image)
            size_consistency_score = 0.2
        elif relative_size < 0.0001:  # Too small
            size_consistency_score = 0.5
        else:
            size_consistency_score = 1.0

        # Combine all scores with weights
        enhanced_confidence = (
            base_confidence * 0.3 +           # Basic shape/area
            contrast_score * 0.4 +            # Local contrast (most important)
            edge_distance_score * 0.2 +       # Distance from whiteboard edges
            size_consistency_score * 0.1      # Size reasonableness
        )

        return max(0.0, min(1.0, enhanced_confidence))

    def pixel_to_camera_relative_mm(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to millimeters relative to camera position

        Args:
            pixel_x: X coordinate in pixels (0 = left)
            pixel_y: Y coordinate in pixels (0 = top, after rotation correction)

        Returns:
            (x_mm, y_mm) relative to camera position
            x_mm: positive = right of camera
            y_mm: positive = forward of camera (toward far edge of view)
        """
        # Convert to camera-centered coordinates
        center_x_px = self.image_width / 2
        center_y_px = self.image_height / 2

        # Offset from center in pixels
        dx_px = pixel_x - center_x_px
        dy_px = pixel_y - center_y_px

        # Convert to mm using perspective correction
        # X direction is straightforward (horizontal)
        x_mm = dx_px * self.mm_per_pixel_x

        # Y direction needs perspective correction
        # Map pixel Y to distance along ground
        y_fraction = pixel_y / self.image_height  # 0 = near, 1 = far
        distance_from_camera = self.near_distance_mm + y_fraction * (self.far_distance_mm - self.near_distance_mm)

        # Y coordinate relative to directly below camera
        y_mm = distance_from_camera

        return x_mm, y_mm

    def camera_relative_to_car_center_mm(self, camera_x_mm: float, camera_y_mm: float) -> Tuple[float, float]:
        """
        Convert camera-relative coordinates to car center coordinates
        Camera is mounted 7.5cm above and directly over the front axle

        Args:
            camera_x_mm: X coordinate relative to camera (right positive)
            camera_y_mm: Y coordinate relative to camera (forward positive)

        Returns:
            (x_mm, y_mm) relative to car center (between front and back axles)
            x_mm: positive = right of car
            y_mm: positive = forward of car
        """
        # Camera is directly above front axle, and car center is 11cm behind front axle
        car_center_offset_mm = 110  # 11cm

        # Transform coordinates
        car_x_mm = camera_x_mm  # X direction unchanged
        car_y_mm = camera_y_mm - car_center_offset_mm  # Forward from car center

        return car_x_mm, car_y_mm

    def detect_and_convert_to_car_coordinates(self, image: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Detect markings and convert to car-relative coordinates

        Args:
            image: Input BGR image from camera

        Returns:
            List of (x_mm, y_mm, confidence) tuples relative to car center
        """
        markings = self.detect_markings(image)
        car_markings = []

        for marking in markings:
            # Convert pixel to camera-relative coordinates
            cam_x, cam_y = self.pixel_to_camera_relative_mm(marking.x, marking.y)

            # Convert to car center coordinates
            car_x, car_y = self.camera_relative_to_car_center_mm(cam_x, cam_y)

            car_markings.append((car_x, car_y, marking.confidence))

        return car_markings

    def visualize_detections(self, image: np.ndarray, markings: List[Marking]) -> np.ndarray:
        """
        Create debug visualization of detected markings

        Args:
            image: Original BGR image
            markings: List of detected markings

        Returns:
            Annotated image
        """
        if not self.debug:
            return image

        # Correct image orientation
        vis_image = self.rotate_image_180(image.copy())

        # Draw detected whiteboard edges/boundaries (if available)
        if self.debug and self.last_detected_edges:
            for x1, y1, x2, y2 in self.last_detected_edges:
                cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Magenta lines

            # Add edge detection info
            edge_text = f"Edges: {len(self.last_detected_edges)}"
            cv2.putText(vis_image, edge_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Draw whiteboard mask area (semi-transparent overlay)
        if self.debug and self.last_whiteboard_mask is not None:
            # Create colored overlay for whiteboard area
            mask_overlay = np.zeros_like(vis_image)
            mask_overlay[:, :, 1] = self.last_whiteboard_mask  # Green channel
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 0.9, mask_overlay, 0.1, 0)

        # Draw markings
        for i, marking in enumerate(markings):
            x, y, w, h = marking.bbox

            # Draw bounding box
            color = (0, 255, 0) if marking.confidence > 0.5 else (0, 255, 255)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            # Draw center point
            center = (int(marking.x), int(marking.y))
            cv2.circle(vis_image, center, 3, (255, 0, 0), -1)

            # Add text info
            text = f"{i}: {marking.confidence:.2f}"
            cv2.putText(vis_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Convert to car coordinates and display
            cam_x, cam_y = self.pixel_to_camera_relative_mm(marking.x, marking.y)
            car_x, car_y = self.camera_relative_to_car_center_mm(cam_x, cam_y)
            coord_text = f"({car_x:.0f},{car_y:.0f}mm)"
            cv2.putText(vis_image, coord_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add performance info
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            perf_text = f"Avg: {avg_time*1000:.1f}ms ({fps:.1f} FPS)"
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


# Test function for standalone testing
def test_marking_detector():
    """Test the marking detector with sample images"""
    import glob

    print("Testing MarkingDetector...")
    detector = MarkingDetector(debug=True)

    # Look for test images
    test_images = glob.glob("*.jpg") + glob.glob("*.JPG") + glob.glob("*.png")

    if not test_images:
        print("No test images found. Place some images in the current directory.")
        return

    for img_path in test_images[:3]:  # Test first 3 images
        print(f"\nProcessing {img_path}...")

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Resize if too large (for testing)
        if image.shape[1] > 640:
            scale = 640 / image.shape[1]
            new_width = 640
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))

        # Detect markings
        markings = detector.detect_markings(image)
        print(f"Found {len(markings)} markings")

        # Get car coordinates
        car_coords = detector.detect_and_convert_to_car_coordinates(image)
        for i, (x, y, conf) in enumerate(car_coords):
            print(f"  Marking {i}: ({x:.1f}, {y:.1f}) mm, confidence: {conf:.2f}")

        # Show visualization
        vis_image = detector.visualize_detections(image, markings)
        cv2.imshow(f"Detections - {img_path}", vis_image)
        cv2.waitKey(1000)  # Show for 1 second

    cv2.destroyAllWindows()

    # Print performance stats
    stats = detector.get_performance_stats()
    print(f"\nPerformance: {stats['avg_time_ms']:.1f}ms avg ({stats['fps']:.1f} FPS)")


if __name__ == "__main__":
    test_marking_detector()