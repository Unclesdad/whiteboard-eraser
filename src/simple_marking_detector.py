#!/usr/bin/env python3
import cv2
import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Marking:
    x: float
    y: float
    area: float
    confidence: float
    bbox: Tuple[int, int, int, int]

class SimpleMarkingDetector:
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

        self.min_marking_area = 2
        self.max_marking_area = 100
        self.marking_threshold = 120
        self.gaussian_blur_size = 3
        self.max_solidity = 0.85
        self.edge_exclusion_pixels = 15
        self.whiteboard_threshold = 140
        self.whiteboard_area_threshold = 1000
        self.top_edge_search_height = 200

        self.erode_kernel = np.ones((2, 2), np.uint8)
        self.dilate_kernel = np.ones((3, 3), np.uint8)

        self.processing_times = []
        self.last_whiteboard_mask = None

        self._calculate_pixel_to_mm_factors()

        if self.debug:
            print(f"SimpleMarkingDetector initialized")
            print(f"  permissive thresholds for max detection")

    def _calculate_pixel_to_mm_factors(self):
        center_distance = self.camera_height_mm / np.tan(self.camera_angle_rad)
        horizontal_coverage_mm = 2 * center_distance * np.tan(self.fov_horizontal_rad / 2)
        self.mm_per_pixel_x = horizontal_coverage_mm / self.image_width

        if self.debug:
            print(f"  center dist={center_distance:.0f}mm, H width={horizontal_coverage_mm:.0f}mm")
            print(f"  {self.mm_per_pixel_x:.2f} mm/px horiz")

    def rotate_image_180(self, image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_180)

    def find_white_surface(self, image: np.ndarray) -> np.ndarray:
        """find whiteboard by thresholding brightness, then flood-fill from bottom edge
        to get the connected region (preserves marking holes unlike connected components)"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness_threshold = 180
        _, white_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        bottom_row = height - 1
        surface_mask = np.zeros_like(white_mask, dtype=np.uint8)

        h, w = white_mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        bottom_white_pixels = []
        for x in range(width):
            if white_mask[bottom_row, x] > 0:
                bottom_white_pixels.append((x, bottom_row))

        if bottom_white_pixels:
            start_x, start_y = bottom_white_pixels[0]
            cv2.floodFill(white_mask, flood_mask, (start_x, start_y), 128)

            flooded_area = (white_mask == 128)
            # re-apply original threshold to preserve holes
            original_threshold = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)[1]
            surface_mask = np.where(flooded_area, original_threshold, 0).astype(np.uint8)

            white_mask[white_mask == 128] = 255
        else:
            print(f"  no bottom blob, using fallback")
            fallback_start = int(height * 0.5)
            surface_mask[fallback_start:, :] = 255

        return surface_mask

    def _analyze_blob_shape(self, contour: np.ndarray) -> dict:
        """get shape metrics to filter out reflections (solid blobs) vs markings (irregular)"""
        area = cv2.contourArea(contour)

        # solidity = area / convex hull area
        # reflections ~1.0, markings lower
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

        perimeter = cv2.arcLength(contour, True)
        perimeter_area_ratio = perimeter / area if area > 0 else 0.0

        return {
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'perimeter_area_ratio': perimeter_area_ratio,
            'area': area
        }

    def _is_likely_marking(self, shape_metrics: dict) -> tuple[bool, str]:
        """check if blob passes filters for being a marking"""
        area = shape_metrics['area']
        solidity = shape_metrics['solidity']

        if area < self.min_marking_area:
            return False, f"too_small ({area:.1f}px < {self.min_marking_area})"

        if area > self.max_marking_area:
            return False, f"too_large ({area:.1f}px > {self.max_marking_area})"

        if solidity > self.max_solidity:
            return False, f"too_solid (solidity={solidity:.2f})"

        return True, "valid_marking"

    def _create_edge_exclusion_mask(self, white_surface_mask: np.ndarray) -> np.ndarray:
        """exclude areas near whiteboard edges to avoid false detections"""
        detection_mask = white_surface_mask.copy()
        contours, _ = cv2.findContours(white_surface_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            boundary_mask = np.zeros_like(white_surface_mask)
            cv2.drawContours(boundary_mask, [largest_contour], -1, 255, thickness=5)
            detection_mask = cv2.bitwise_and(detection_mask, cv2.bitwise_not(boundary_mask))

        return detection_mask

    def detect_markings(self, image: np.ndarray) -> List[Marking]:
        """find markings by detecting holes in the white surface"""
        start_time = time.time()
        corrected = self.rotate_image_180(image)

        if self.debug:
            height, width = corrected.shape[:2]
            print(f"  Processing frame: {width}x{height}")

        white_surface_mask = self.find_white_surface(corrected)
        self.last_whiteboard_mask = white_surface_mask

        detection_mask = self._create_edge_exclusion_mask(white_surface_mask)

        # invert surface mask to find holes (markings show up as holes in white surface)
        holes_mask = cv2.bitwise_not(white_surface_mask)

        kernel = np.ones((2, 2), np.uint8)
        holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, kernel)

        # only keep holes away from edges
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
        rejected_count = {'too_small': 0, 'too_large': 0, 'too_solid': 0, 'near_edge': 0}

        for contour in contours:
            shape_metrics = self._analyze_blob_shape(contour)
            is_marking, reason = self._is_likely_marking(shape_metrics)

            if self.debug and len(markings) < 10:
                area = shape_metrics['area']
                solidity = shape_metrics['solidity']
                print(f"    Contour {len(markings)}: area={area:.1f}, solidity={solidity:.2f}, {reason}")

            if not is_marking:
                if 'too_small' in reason:
                    rejected_count['too_small'] += 1
                elif 'too_large' in reason:
                    rejected_count['too_large'] += 1
                elif 'too_solid' in reason:
                    rejected_count['too_solid'] += 1
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2
            center_y = y + h / 2

            if detection_mask[int(center_y), int(center_x)] > 0:
                area = shape_metrics['area']
                solidity = shape_metrics['solidity']

                # confidence: smaller + more irregular = better
                area_score = 1.0 - ((area - self.min_marking_area) /
                                   (self.max_marking_area - self.min_marking_area))
                area_score = max(0.0, min(1.0, area_score))
                solidity_score = 1.0 - solidity

                confidence = (area_score * 0.4 + solidity_score * 0.6)
                confidence = max(0.2, min(1.0, confidence))

                marking = Marking(
                    x=center_x,
                    y=center_y,
                    area=area,
                    confidence=confidence,
                    bbox=(x, y, w, h)
                )
                markings.append(marking)
            else:
                rejected_count['near_edge'] += 1
                if self.debug and len(markings) < 5:
                    print(f"    Skipped contour at ({center_x:.0f},{center_y:.0f}) - too close to edge")

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        if self.debug:
            print(f"  Detected {len(markings)} markings in {processing_time*1000:.1f}ms")
            white_area = np.sum(white_surface_mask) / 255.0
            holes_area = np.sum(holes_cleaned) / 255.0
            contour_count = len(contours)
            print(f"  Blob detection: White={white_area:.0f}px, Holes={holes_area:.0f}px, Contours={contour_count}")

            total_rejected = sum(rejected_count.values())
            print(f"  Rejections: {total_rejected} total")
            print(f"    - Too small (<{self.min_marking_area}px): {rejected_count['too_small']}")
            print(f"    - Too large (>{self.max_marking_area}px): {rejected_count['too_large']}")
            print(f"    - Too solid (>{self.max_solidity}): {rejected_count['too_solid']}")
            print(f"    - Near edge: {rejected_count['near_edge']}")
            print(f"  Accepted markings: {len(markings)}")

        return markings

    def pixel_to_camera_relative_mm(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """convert pixel coords to mm relative to camera using angular projection"""
        center_x_px = self.image_width / 2

        dx_px = pixel_x - center_x_px
        x_mm = dx_px * self.mm_per_pixel_x

        # map pixel y to viewing angle, then to ground distance
        y_normalized = (pixel_y / self.image_height) - 0.5
        angle_from_center = y_normalized * self.fov_vertical_rad
        viewing_angle = self.camera_angle_rad + angle_from_center
        viewing_angle = np.clip(viewing_angle, 0.087, 1.57)

        distance_forward = self.camera_height_mm / np.tan(viewing_angle)
        y_mm = distance_forward

        return x_mm, y_mm

    def camera_relative_to_car_center_mm(self, camera_x_mm: float, camera_y_mm: float) -> Tuple[float, float]:
        """convert from camera coords to car center coords"""
        car_center_offset_mm = 110
        car_x_mm = camera_x_mm
        car_y_mm = camera_y_mm - car_center_offset_mm
        return car_x_mm, car_y_mm

    def detect_and_convert_to_car_coordinates(self, image: np.ndarray) -> List[Tuple[float, float, float]]:
        """detect markings and return as car-relative coordinates"""
        markings = self.detect_markings(image)
        car_markings = []

        for marking in markings:
            cam_x, cam_y = self.pixel_to_camera_relative_mm(marking.x, marking.y)
            car_x, car_y = self.camera_relative_to_car_center_mm(cam_x, cam_y)
            car_markings.append((car_x, car_y, marking.confidence))

        return car_markings

    def visualize_detections(self, image: np.ndarray, markings: List[Marking], whiteboard_mask: np.ndarray = None) -> np.ndarray:
        """draw detected markings on image for debugging"""
        if not self.debug:
            return image

        vis_image = self.rotate_image_180(image.copy())
        mask_to_use = whiteboard_mask if whiteboard_mask is not None else self.last_whiteboard_mask

        if mask_to_use is not None:
            if mask_to_use.shape != vis_image.shape[:2]:
                mask_to_use = cv2.resize(mask_to_use, (vis_image.shape[1], vis_image.shape[0]))

            contours, _ = cv2.findContours(mask_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(vis_image, [largest_contour], -1, (0, 255, 255), 2)

            white_area = np.sum(mask_to_use) / 255.0
            cv2.putText(vis_image, f"White surface: {white_area:.0f}px",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for i, marking in enumerate(markings):
            x, y, w, h = marking.bbox

            if marking.confidence > 0.6:
                color = (0, 255, 0)
            elif marking.confidence > 0.3:
                color = (0, 255, 255)
            else:
                color = (0, 128, 255)

            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            center = (int(marking.x), int(marking.y))
            cv2.circle(vis_image, center, 3, (255, 0, 0), -1)

            text = f"M{i}: {marking.confidence:.2f}"
            cv2.putText(vis_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cam_x, cam_y = self.pixel_to_camera_relative_mm(marking.x, marking.y)
            car_x, car_y = self.camera_relative_to_car_center_mm(cam_x, cam_y)
            coord_text = f"({car_x:.0f},{car_y:.0f}mm)"
            cv2.putText(vis_image, coord_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            perf_text = f"Simple: {avg_time*1000:.1f}ms ({fps:.1f} FPS)"
            cv2.putText(vis_image, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return vis_image

    def get_performance_stats(self) -> dict:
        if not self.processing_times:
            return {"avg_time_ms": 0, "fps": 0, "samples": 0}

        avg_time = np.mean(self.processing_times)
        return {
            "avg_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "samples": len(self.processing_times)
        }