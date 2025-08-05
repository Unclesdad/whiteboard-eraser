#!/usr/bin/env python3
"""
Whiteboard Edge Detection and Persistent Marking Tracker - Horizontal Camera View
For Raspberry Pi Camera Module 3 with Pi 5
Camera mounted on whiteboard, looking parallel to surface
Includes marking persistence and erasure based on object proximity
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Set
import math
import uuid

@dataclass
class WhiteboardConfig:
    """Configuration for whiteboard dimensions and camera setup"""
    width_mm: float  # Physical width of whiteboard in mm
    height_mm: float  # Physical height of whiteboard in mm
    distance_mm: float  # Distance from camera to whiteboard surface in mm (camera height)
    camera_x_mm: float  # Camera's X position on whiteboard (from left edge)
    camera_y_mm: float  # Camera's Y position on whiteboard (from top edge)
    camera_placement_mm: float = 100.0  # Distance from camera to object center (behind camera)
    erasure_radius_mm: float = 50.0  # Radius within which markings are erased
    camera_fov_h: float = 62.0  # Horizontal field of view in degrees
    camera_fov_v: float = 48.8  # Vertical field of view in degrees

@dataclass
class Marking:
    """Represents a detected marking on the whiteboard"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID
    center_x_mm: float = 0.0  # X position on whiteboard in mm
    center_y_mm: float = 0.0  # Y position on whiteboard in mm
    width_mm: float = 0.0     # Width of marking in mm
    height_mm: float = 0.0    # Height of marking in mm
    color: str = ""          # Color category ('red', 'blue', 'green', 'black')
    confidence: float = 0.0   # Detection confidence (0-1)
    last_seen: float = 0.0    # Timestamp when last detected
    detection_count: int = 1  # Number of times detected (for stability)

@dataclass
class CameraPosition:
    """Current camera position relative to whiteboard"""
    x_mm: float  # X position from left edge
    y_mm: float  # Y position from top edge
    confidence: float  # Position confidence based on visible features

@dataclass
class ObjectPosition:
    """Position of the object center (behind camera)"""
    x_mm: float  # X position on whiteboard
    y_mm: float  # Y position on whiteboard

@dataclass
class DetectedFeatures:
    """Information about detected features in the camera view"""
    left_edge_px: Optional[int] = None
    right_edge_px: Optional[int] = None
    red_markers: List[int] = field(default_factory=list)
    whiteboard_bottom_px: Optional[int] = None
    markings: List[Marking] = field(default_factory=list)

class WhiteboardTracker:
    def __init__(self, config: WhiteboardConfig):
        self.config = config
        self.camera = None
        self.frame_width = 1920
        self.frame_height = 1080
        
        # Persistence logic
        self.persistent_markings: Dict[str, Marking] = {}
        self.marking_merge_threshold_mm = 15.0  # Reduced for line points
        self.line_point_spacing_mm = 10.0  # Distance between points on lines
        
        # Calculate field of view parameters
        self.calculate_fov_parameters()
        
        # Color ranges for whiteboard detection
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 40, 255])
        
        # Color ranges for different markers in HSV
        self.color_ranges = {
            'red': [
                (np.array([0, 70, 70]), np.array([10, 255, 255])),
                (np.array([170, 70, 70]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 50, 50]), np.array([130, 255, 255]))
            ],
            'green': [
                (np.array([40, 50, 50]), np.array([80, 255, 255]))
            ],
            'black': [
                (np.array([0, 0, 0]), np.array([180, 255, 50]))
            ]
        }
        
        self.min_marking_area = 100  # Minimum pixel area for a marking
        
    def calculate_fov_parameters(self):
        """Calculate what the camera can see at the whiteboard plane"""
        self.fov_width_at_board = 2 * self.config.distance_mm * math.tan(math.radians(self.config.camera_fov_h / 2))
        self.fov_height_at_board = 2 * self.config.distance_mm * math.tan(math.radians(self.config.camera_fov_v / 2))
        
        self.pixels_per_mm_horizontal = self.frame_width / self.fov_width_at_board
        self.pixels_per_mm_vertical = self.frame_height / self.fov_height_at_board
        
        self.visible_left_mm = self.config.camera_x_mm - self.fov_width_at_board / 2
        self.visible_right_mm = self.config.camera_x_mm + self.fov_width_at_board / 2
        self.visible_top_mm = self.config.camera_y_mm - self.fov_height_at_board / 2
        self.visible_bottom_mm = self.config.camera_y_mm + self.fov_height_at_board / 2
        
    def initialize_camera(self):
        """Initialize the Raspberry Pi Camera"""
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (self.frame_width, self.frame_height)},
            lores={"size": (640, 480)},
            display="lores"
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)
        
    def get_camera_position(self, features: DetectedFeatures) -> CameraPosition:
        """Calculate camera position based on detected features"""
        confidence = 0.0
        estimated_x = self.config.camera_x_mm
        estimated_y = self.config.camera_y_mm
        
        if features.left_edge_px is not None:
            left_edge_mm = self.pixel_to_whiteboard_position_x(features.left_edge_px)
            estimated_x = -left_edge_mm
            confidence += 0.5
            
        if features.right_edge_px is not None:
            right_edge_mm = self.pixel_to_whiteboard_position_x(features.right_edge_px)
            if features.left_edge_px is None:
                estimated_x = self.config.width_mm - right_edge_mm
                confidence += 0.5
            else:
                confidence += 0.3
                
        if confidence == 0:
            confidence = 0.1
            
        return CameraPosition(estimated_x, estimated_y, confidence)
    
    def get_object_center_position(self, camera_pos: CameraPosition) -> ObjectPosition:
        """Calculate the position of the object center (behind the camera)"""
        # For now, assuming camera faces along positive X direction
        # The object center is CAMERA_PLACEMENT mm behind the camera
        object_x = camera_pos.x_mm - self.config.camera_placement_mm
        object_y = camera_pos.y_mm
        
        return ObjectPosition(object_x, object_y)
    
    def update_persistent_markings(self, detected_markings: List[Marking]):
        """Update persistent marking storage with newly detected markings"""
        current_time = time.time()
        
        # Match detected markings with existing ones
        unmatched_detected = set(range(len(detected_markings)))
        
        for detected_idx, detected in enumerate(detected_markings):
            best_match_id = None
            best_distance = float('inf')
            
            # Find closest existing marking
            for marking_id, existing in self.persistent_markings.items():
                distance = math.sqrt(
                    (detected.center_x_mm - existing.center_x_mm) ** 2 +
                    (detected.center_y_mm - existing.center_y_mm) ** 2
                )
                
                if distance < self.marking_merge_threshold_mm and distance < best_distance:
                    best_distance = distance
                    best_match_id = marking_id
            
            if best_match_id:
                # Update existing marking
                existing = self.persistent_markings[best_match_id]
                # Weighted average for position (favor new detection)
                weight = 0.7
                existing.center_x_mm = weight * detected.center_x_mm + (1-weight) * existing.center_x_mm
                existing.center_y_mm = weight * detected.center_y_mm + (1-weight) * existing.center_y_mm
                existing.width_mm = detected.width_mm
                existing.height_mm = detected.height_mm
                existing.last_seen = current_time
                existing.detection_count += 1
                existing.confidence = min(1.0, existing.detection_count * 0.1)
                unmatched_detected.discard(detected_idx)
            
        # Add new markings
        for idx in unmatched_detected:
            detected = detected_markings[idx]
            detected.last_seen = current_time
            self.persistent_markings[detected.id] = detected
    
    def erase_markings_near_object(self, object_pos: ObjectPosition) -> List[str]:
        """Remove markings within erasure radius of object center"""
        erased_ids = []
        
        for marking_id, marking in list(self.persistent_markings.items()):
            distance = math.sqrt(
                (marking.center_x_mm - object_pos.x_mm) ** 2 +
                (marking.center_y_mm - object_pos.y_mm) ** 2
            )
            
            if distance <= self.config.erasure_radius_mm:
                del self.persistent_markings[marking_id]
                erased_ids.append(marking_id)
                
        return erased_ids
    
    def detect_markings(self, frame, hsv) -> List[Marking]:
        """Detect colored markings on the whiteboard, including lines as point sequences"""
        markings = []
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        for color_name, color_ranges in self.color_ranges.items():
            color_mask = np.zeros_like(white_mask)
            for lower, upper in color_ranges:
                color_mask |= cv2.inRange(hsv, lower, upper)
            
            color_mask = cv2.bitwise_and(color_mask, white_mask)
            
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_marking_area:
                    continue
                
                # Check if this is a line-like shape
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(min(w, h), 1)
                
                if aspect_ratio > 3:  # Likely a line
                    # Extract skeleton/centerline of the shape
                    line_markings = self._extract_line_markings(contour, color_mask[y:y+h, x:x+w], 
                                                               x, y, color_name)
                    markings.extend(line_markings)
                else:
                    # Regular blob-like marking
                    center_x_px = x + w // 2
                    center_y_px = y + h // 2
                    
                    center_x_mm = self.pixel_to_whiteboard_position_x(center_x_px)
                    center_y_mm = self.pixel_to_whiteboard_position_y(center_y_px)
                    width_mm = w / self.pixels_per_mm_horizontal
                    height_mm = h / self.pixels_per_mm_vertical
                    
                    confidence = min(1.0, area / 1000)
                    
                    marking = Marking(
                        center_x_mm=center_x_mm,
                        center_y_mm=center_y_mm,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        color=color_name,
                        confidence=confidence
                    )
                    markings.append(marking)
                    
        return markings
    
    def _extract_line_markings(self, contour, mask_roi, x_offset, y_offset, color_name) -> List[Marking]:
        """Extract a series of point markings along a line"""
        markings = []
        
        # For line detection, we'll use a simpler approach without cv2.ximgproc
        # Find the approximate centerline by fitting a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_poly = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we have at least 2 points, we can create a line
        if len(approx_poly) >= 2:
            # Convert to simple point list
            points = approx_poly.reshape(-1, 2)
            
            # For each segment of the polyline
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                
                # Calculate number of intermediate points based on line length
                length = np.linalg.norm(end - start)
                num_points = max(2, int(length / (self.line_point_spacing_mm * self.pixels_per_mm_horizontal)))
                
                # Interpolate points along the segment
                for j in range(num_points):
                    t = j / (num_points - 1) if num_points > 1 else 0
                    point = start + t * (end - start)
                    
                    x_px = int(point[0]) + x_offset
                    y_px = int(point[1]) + y_offset
                    
                    center_x_mm = self.pixel_to_whiteboard_position_x(x_px)
                    center_y_mm = self.pixel_to_whiteboard_position_y(y_px)
                    
                    # Create a small marking for this point on the line
                    marking = Marking(
                        center_x_mm=center_x_mm,
                        center_y_mm=center_y_mm,
                        width_mm=8.0,   # Small fixed width
                        height_mm=8.0,  # Small fixed height
                        color=color_name,
                        confidence=0.8
                    )
                    markings.append(marking)
        
        # If the shape is more complex, also sample along the contour
        elif len(contour) > 10:
            # Sample points directly from the contour
            perimeter = cv2.arcLength(contour, True)
            num_samples = max(2, int(perimeter / (self.line_point_spacing_mm * self.pixels_per_mm_horizontal)))
            
            for i in range(num_samples):
                idx = int(i * len(contour) / num_samples)
                point = contour[idx][0]
                
                x_px = point[0] + x_offset
                y_px = point[1] + y_offset
                
                center_x_mm = self.pixel_to_whiteboard_position_x(x_px)
                center_y_mm = self.pixel_to_whiteboard_position_y(y_px)
                
                marking = Marking(
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    width_mm=8.0,
                    height_mm=8.0,
                    color=color_name,
                    confidence=0.8
                )
                markings.append(marking)
                
        return markings
    
    def detect_features(self, frame) -> DetectedFeatures:
        """Detect whiteboard edges and markings"""
        features = DetectedFeatures()
        features.red_markers = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        features.left_edge_px = self._find_left_edge(white_mask)
        features.right_edge_px = self._find_right_edge(white_mask)
        
        red_mask = np.zeros_like(white_mask)
        for lower, upper in self.color_ranges['red']:
            red_mask |= cv2.inRange(hsv, lower, upper)
        features.red_markers = self._find_vertical_red_lines(red_mask)
        
        features.whiteboard_bottom_px = self._find_bottom_edge(white_mask)
        features.markings = self.detect_markings(frame, hsv)
        
        return features, white_mask
    
    def _find_left_edge(self, mask):
        """Find leftmost vertical edge of whiteboard"""
        h, w = mask.shape
        vertical_sum = np.sum(mask, axis=0)
        
        for x in range(w):
            if vertical_sum[x] > h * 0.3:
                if x > 0 and vertical_sum[x-1] < h * 0.1:
                    return x
        return None
    
    def _find_right_edge(self, mask):
        """Find rightmost vertical edge of whiteboard"""
        h, w = mask.shape
        vertical_sum = np.sum(mask, axis=0)
        
        for x in range(w-1, -1, -1):
            if vertical_sum[x] > h * 0.3:
                if x < w-1 and vertical_sum[x+1] < h * 0.1:
                    return x
        return None
    
    def _find_vertical_red_lines(self, red_mask):
        """Find vertical red lines that act as edge markers"""
        h, w = red_mask.shape
        vertical_sum = np.sum(red_mask, axis=0)
        
        red_lines = []
        min_line_height = h * 0.2
        
        in_line = False
        line_start = 0
        
        for x in range(w):
            if vertical_sum[x] > min_line_height:
                if not in_line:
                    in_line = True
                    line_start = x
            else:
                if in_line:
                    line_center = (line_start + x) // 2
                    red_lines.append(line_center)
                    in_line = False
                    
        return red_lines
    
    def _find_bottom_edge(self, mask):
        """Find horizontal bottom edge of whiteboard"""
        h, w = mask.shape
        horizontal_sum = np.sum(mask, axis=1)
        
        for y in range(h-1, -1, -1):
            if horizontal_sum[y] > w * 0.3:
                if y < h-1 and horizontal_sum[y+1] < w * 0.1:
                    return y
        return None
    
    def pixel_to_whiteboard_position_x(self, pixel_x: int) -> float:
        """Convert horizontal pixel position to X position on whiteboard in mm"""
        pixel_offset = pixel_x - (self.frame_width / 2)
        angle_offset = (pixel_offset / self.frame_width) * self.config.camera_fov_h
        distance_from_camera_center = self.config.distance_mm * math.tan(math.radians(angle_offset))
        whiteboard_x = self.config.camera_x_mm + distance_from_camera_center
        return whiteboard_x
    
    def pixel_to_whiteboard_position_y(self, pixel_y: int) -> float:
        """Convert vertical pixel position to Y position on whiteboard in mm"""
        pixel_offset = pixel_y - (self.frame_height / 2)
        angle_offset = (pixel_offset / self.frame_height) * self.config.camera_fov_v
        distance_from_camera_center = self.config.distance_mm * math.tan(math.radians(angle_offset))
        whiteboard_y = self.config.camera_y_mm + distance_from_camera_center
        return whiteboard_y
    
    def is_position_visible(self, x_mm: float, y_mm: float) -> bool:
        """Check if a position is within current camera view"""
        return (self.visible_left_mm <= x_mm <= self.visible_right_mm and
                self.visible_top_mm <= y_mm <= self.visible_bottom_mm)
    
    def draw_overlay(self, frame, features: DetectedFeatures, camera_pos: CameraPosition, 
                     object_pos: ObjectPosition, erased_ids: List[str]):
        """Draw detection overlay on frame"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw detected edges
        if features.left_edge_px is not None:
            cv2.line(overlay, (features.left_edge_px, 0), 
                    (features.left_edge_px, h), (0, 255, 0), 2)
            
        if features.right_edge_px is not None:
            cv2.line(overlay, (features.right_edge_px, 0), 
                    (features.right_edge_px, h), (0, 255, 0), 2)
        
        # Group markings by proximity for line visualization
        marking_groups = self._group_line_markings(list(self.persistent_markings.values()))
        
        # Draw all persistent markings that are visible
        for marking in self.persistent_markings.values():
            if self.is_position_visible(marking.center_x_mm, marking.center_y_mm):
                # Convert to pixels
                center_x_px = int((marking.center_x_mm - self.config.camera_x_mm) * self.pixels_per_mm_horizontal + self.frame_width / 2)
                center_y_px = int((marking.center_y_mm - self.config.camera_y_mm) * self.pixels_per_mm_vertical + self.frame_height / 2)
                width_px = int(marking.width_mm * self.pixels_per_mm_horizontal)
                height_px = int(marking.height_mm * self.pixels_per_mm_vertical)
                
                colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'black': (128, 128, 128)}
                color = colors.get(marking.color, (255, 255, 255))
                
                # Fade based on time since last seen
                time_since_seen = time.time() - marking.last_seen
                alpha = max(0.3, 1.0 - time_since_seen / 10.0)
                
                # For line points (small markings), draw as circles
                if marking.width_mm <= 10 and marking.height_mm <= 10:
                    radius = max(2, int(marking.width_mm * self.pixels_per_mm_horizontal / 2))
                    cv2.circle(overlay, (center_x_px, center_y_px), radius, color, -1)
                else:
                    # Regular rectangular marking
                    x1 = center_x_px - width_px // 2
                    y1 = center_y_px - height_px // 2
                    x2 = center_x_px + width_px // 2
                    y2 = center_y_px + height_px // 2
                    
                    thickness = 2 if time_since_seen < 1 else 1
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                    
                    # Label only for larger markings
                    label = f"{marking.color[0].upper()}: ({marking.center_x_mm:.0f}, {marking.center_y_mm:.0f})"
                    cv2.putText(overlay, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw erasure circle around object center
        if self.is_position_visible(object_pos.x_mm, object_pos.y_mm):
            center_x_px = int((object_pos.x_mm - self.config.camera_x_mm) * self.pixels_per_mm_horizontal + self.frame_width / 2)
            center_y_px = int((object_pos.y_mm - self.config.camera_y_mm) * self.pixels_per_mm_vertical + self.frame_height / 2)
            radius_px = int(self.config.erasure_radius_mm * self.pixels_per_mm_horizontal)
            
            # Draw erasure circle
            cv2.circle(overlay, (center_x_px, center_y_px), radius_px, (255, 0, 255), 2)
            cv2.circle(overlay, (center_x_px, center_y_px), 5, (255, 0, 255), -1)
            cv2.putText(overlay, "Eraser", (center_x_px - 30, center_y_px - radius_px - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Info text
        info_text = [
            f"Camera: ({camera_pos.x_mm:.0f}, {camera_pos.y_mm:.0f}) mm",
            f"Object Center: ({object_pos.x_mm:.0f}, {object_pos.y_mm:.0f}) mm",
            f"Total Markings: {len(self.persistent_markings)}",
            f"Erased: {len(erased_ids)} this frame"
        ]
        
        y_pos = h - 120
        for text in info_text:
            cv2.putText(overlay, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
            
        return overlay
    
    def _group_line_markings(self, markings: List[Marking]) -> List[List[Marking]]:
        """Group markings that likely belong to the same line"""
        # Simple proximity-based grouping
        groups = []
        used = set()
        
        for i, marking in enumerate(markings):
            if i in used or marking.width_mm > 10:  # Skip if already grouped or not a line point
                continue
                
            group = [marking]
            used.add(i)
            
            # Find nearby markings of the same color
            for j, other in enumerate(markings):
                if j in used or other.color != marking.color or other.width_mm > 10:
                    continue
                    
                # Check if close to any marking in the group
                for g_marking in group:
                    dist = math.sqrt(
                        (other.center_x_mm - g_marking.center_x_mm) ** 2 +
                        (other.center_y_mm - g_marking.center_y_mm) ** 2
                    )
                    if dist < self.line_point_spacing_mm * 2:
                        group.append(other)
                        used.add(j)
                        break
                        
            if len(group) > 1:
                groups.append(group)
                
        return groups
    
    def get_current_state(self, frame) -> Tuple[CameraPosition, ObjectPosition, List[Marking], List[str]]:
        """Main method to get current state and update persistent markings"""
        # Detect features in current frame
        features, _ = self.detect_features(frame)
        
        # Get camera position
        camera_pos = self.get_camera_position(features)
        
        # Calculate object center position
        object_pos = self.get_object_center_position(camera_pos)
        
        # Update persistent markings with newly detected ones
        self.update_persistent_markings(features.markings)
        
        # Erase markings near object center
        erased_ids = self.erase_markings_near_object(object_pos)
        
        # Return current state
        current_markings = list(self.persistent_markings.values())
        return camera_pos, object_pos, current_markings, erased_ids
    
    def run(self):
        """Main loop for whiteboard tracking"""
        self.initialize_camera()
        
        print("Starting whiteboard tracking with persistent markings...")
        print(f"Whiteboard: {self.config.width_mm}mm x {self.config.height_mm}mm")
        print(f"Camera placement: {self.config.camera_placement_mm}mm behind camera")
        print(f"Erasure radius: {self.config.erasure_radius_mm}mm")
        print("\nPress 'q' to quit, 's' to save state, 'c' to clear all markings")
        
        cv2.namedWindow("Whiteboard Tracker", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                frame = self.camera.capture_array()
                
                # Get current state
                camera_pos, object_pos, markings, erased_ids = self.get_current_state(frame)
                
                # For visualization
                features, _ = self.detect_features(frame)
                display_frame = self.draw_overlay(frame, features, camera_pos, object_pos, erased_ids)
                
                cv2.imshow("Whiteboard Tracker", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print(f"\nCurrent State:")
                    print(f"Camera: ({camera_pos.x_mm:.1f}, {camera_pos.y_mm:.1f}) mm")
                    print(f"Object Center: ({object_pos.x_mm:.1f}, {object_pos.y_mm:.1f}) mm")
                    print(f"Total Persistent Markings: {len(markings)}")
                    for i, marking in enumerate(markings):
                        age = time.time() - marking.last_seen
                        print(f"  {i+1}. {marking.color} at ({marking.center_x_mm:.1f}, {marking.center_y_mm:.1f}) mm, "
                              f"age: {age:.1f}s")
                elif key == ord('c'):
                    self.persistent_markings.clear()
                    print("Cleared all markings")
                    
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

def main():
    # Example configuration
    config = WhiteboardConfig(
        width_mm=2400.0,          # 2.4 meters wide
        height_mm=1200.0,         # 1.2 meters tall
        distance_mm=50.0,         # Camera 50mm above whiteboard
        camera_x_mm=1200.0,       # Camera centered horizontally
        camera_y_mm=600.0,        # Camera centered vertically
        camera_placement_mm=100.0, # Object center 100mm behind camera
        erasure_radius_mm=50.0    # Erase within 50mm radius
    )
    
    tracker = WhiteboardTracker(config)
    
    # For visualization
    tracker.run()
    
    # For integration:
    # tracker.initialize_camera()
    # while True:
    #     frame = tracker.camera.capture_array()
    #     camera_pos, object_pos, markings, erased = tracker.get_current_state(frame)
    #     # Use the state information...

if __name__ == "__main__":
    main()