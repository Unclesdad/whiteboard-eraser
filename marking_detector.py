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

        # Morphological operations kernels
        self.erode_kernel = np.ones((2, 2), np.uint8)
        self.dilate_kernel = np.ones((3, 3), np.uint8)

        # Performance tracking
        self.processing_times = []

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
        far_distance = self.camera_height_mm / np.tan(self.camera_angle_rad - fov_half_vertical)
        # Distance from directly below camera to near edge of view
        near_distance = self.camera_height_mm / np.tan(self.camera_angle_rad + fov_half_vertical)

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

    def detect_markings(self, image: np.ndarray) -> List[Marking]:
        """
        Detect dark markings on white background

        Args:
            image: Input BGR image from camera

        Returns:
            List of detected markings
        """
        start_time = time.time()

        # Preprocess image
        gray = self.preprocess_image(image)

        # Create binary mask for dark markings
        # Invert so markings become white on black background
        _, binary = cv2.threshold(gray, self.marking_threshold, 255, cv2.THRESH_BINARY_INV)

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

                # Calculate center
                center_x = x + w / 2
                center_y = y + h / 2

                # Calculate confidence based on area and shape
                confidence = self._calculate_confidence(contour, area)

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
        """Calculate confidence score for a detected marking"""
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