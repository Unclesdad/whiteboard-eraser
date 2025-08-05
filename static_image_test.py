#!/usr/bin/env python3
"""
Test script for whiteboard vision system using static images
"""

import cv2
import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import uuid
import json

# Define the classes we need locally to avoid importing from whiteboard_tracker
@dataclass
class WhiteboardConfig:
    """Configuration for whiteboard dimensions and camera setup"""
    width_mm: float  # Physical width of whiteboard in mm
    height_mm: float  # Physical height of whiteboard in mm
    distance_mm: float  # Distance from camera to whiteboard surface in mm
    camera_x_mm: float  # Camera's X position on whiteboard (from left edge)
    camera_y_mm: float  # Camera's Y position on whiteboard (from top edge)
    camera_placement_mm: float = 100.0  # Distance from camera to object center (behind camera)
    erasure_radius_mm: float = 50.0  # Radius within which markings are erased
    camera_fov_h: float = 62.0  # Horizontal field of view in degrees
    camera_fov_v: float = 48.8  # Vertical field of view in degrees

@dataclass
class Marking:
    """Represents a detected marking on the whiteboard"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    width_mm: float = 0.0
    height_mm: float = 0.0
    color: str = ""
    confidence: float = 0.0
    last_seen: float = 0.0
    detection_count: int = 1

@dataclass
class CameraPosition:
    """Current camera position relative to whiteboard"""
    x_mm: float
    y_mm: float
    confidence: float

@dataclass
class ObjectPosition:
    """Position of the object center (behind camera)"""
    x_mm: float
    y_mm: float

# Simplified version of the tracker for testing with images
class TestWhiteboardTracker:
    def __init__(self, config: WhiteboardConfig):
        self.config = config
        
        # Calculate field of view parameters
        self.calculate_fov_parameters()
        
        # Color ranges for whiteboard detection - more permissive
        self.white_lower = np.array([0, 0, 150])  # Lower threshold for whiteboards
        self.white_upper = np.array([180, 60, 255])  # Allow more saturation
        
        # Gray edge detection (for whiteboard borders)
        self.gray_lower = np.array([0, 0, 50])
        self.gray_upper = np.array([180, 40, 150])
        
        # Color ranges for different markers in HSV
        # Added purple/violet detection
        self.color_ranges = {
            'purple': [
                (np.array([120, 50, 50]), np.array([150, 255, 255]))  # Purple/violet range
            ],
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
        
        self.min_marking_area = 50  # Smaller threshold for testing
        
    def calculate_fov_parameters(self):
        """Calculate what the camera can see at the whiteboard plane"""
        # Typical phone camera FOV
        self.config.camera_fov_h = 70.0  # Approximate for phone camera
        self.config.camera_fov_v = 55.0
        
        self.fov_width_at_board = 2 * self.config.distance_mm * np.tan(np.radians(self.config.camera_fov_h / 2))
        self.fov_height_at_board = 2 * self.config.distance_mm * np.tan(np.radians(self.config.camera_fov_v / 2))
        
        print(f"Field of view at whiteboard: {self.fov_width_at_board:.1f}mm x {self.fov_height_at_board:.1f}mm")
        
    def detect_edges_and_position(self, image):
        """Detect whiteboard edges and estimate camera position"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # Debug: show what we're working with
        print(f"\nDebug - Image analysis:")
        print(f"  Image dimensions: {w}x{h}")
        
        # Detect white areas
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        white_pixels = np.sum(white_mask > 0)
        print(f"  White pixels: {white_pixels} ({100*white_pixels/(w*h):.1f}% of image)")
        
        # Detect gray edges
        gray_mask = cv2.inRange(hsv, self.gray_lower, self.gray_upper)
        gray_pixels = np.sum(gray_mask > 0)
        print(f"  Gray pixels: {gray_pixels} ({100*gray_pixels/(w*h):.1f}% of image)")
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Use Hough Line detection to find straight edges
        edges = cv2.Canny(white_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        # Find edges
        left_edge = None
        right_edge = None
        top_edge = None
        bottom_edge = None
        
        # Also find corner if two edges meet
        corner_x = None
        corner_y = None
        angle = None
        
        # Scan for edges with improved detection
        vertical_sum = np.sum(white_mask, axis=0)
        horizontal_sum = np.sum(white_mask, axis=1)
        
        # Debug: print some statistics
        print(f"  Vertical sum max: {np.max(vertical_sum)}, mean: {np.mean(vertical_sum):.1f}")
        print(f"  Horizontal sum max: {np.max(horizontal_sum)}, mean: {np.mean(horizontal_sum):.1f}")
        
        # More flexible edge detection
        v_threshold = np.max(vertical_sum) * 0.1  # 10% of max
        h_threshold = np.max(horizontal_sum) * 0.1
        
        # Left edge - look for first significant white column (prioritize bottom half)
        for x in range(w//3):
            # Check bottom half first
            bottom_sum = np.sum(white_mask[h//2:, x])
            if bottom_sum > (h//2) * 0.3:  # At least 30% white in bottom half
                left_edge = x
                print(f"  Found left edge at x={x}")
                break
                    
        # Right edge - look from right side (prioritize bottom half)
        for x in range(w-1, 2*w//3, -1):
            bottom_sum = np.sum(white_mask[h//2:, x])
            if bottom_sum > (h//2) * 0.3:
                right_edge = x
                print(f"  Found right edge at x={x}")
                break
                    
        # Top edge - this is the top of the whiteboard, search from bottom up
        for y in range(h-1, h//3, -1):  # Start from bottom, go up
            if horizontal_sum[y] > h_threshold:
                # Look for where whiteboard starts (transition from non-white to white)
                if y < h-1 and horizontal_sum[y+1] < horizontal_sum[y] * 0.5:
                    top_edge = y
                    print(f"  Found top edge at y={y}")
                    break
                    
        # Bottom edge - usually at the very bottom of frame
        # Start from bottom and look for last row with significant white
        for y in range(h-1, 2*h//3, -1):
            if horizontal_sum[y] > h_threshold:
                bottom_edge = y
                print(f"  Found bottom edge at y={y}")
                break
        
        # Try to detect corner and angle from Hough lines
        # Focus on lines in the bottom 2/3 of the image
        edges_bottom = edges.copy()
        edges_bottom[:h//3, :] = 0  # Mask out top third
        
        lines = cv2.HoughLinesP(edges_bottom, 1, np.pi/180, threshold=80, minLineLength=80, maxLineGap=20)
        
        if lines is not None:
            # Find the two most prominent perpendicular lines
            vertical_lines = []
            horizontal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                
                # Classify as vertical or horizontal
                if abs(angle_deg) > 45 or abs(angle_deg) < -45:
                    vertical_lines.append((x1, y1, x2, y2, angle_rad))
                else:
                    horizontal_lines.append((x1, y1, x2, y2, angle_rad))
            
            # If we have both vertical and horizontal lines, find intersection
            if vertical_lines and horizontal_lines:
                # Take the longest lines
                v_line = max(vertical_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2)
                h_line = max(horizontal_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2)
                
                # Find intersection point (simplified - assumes lines are nearly perpendicular)
                corner_x = (v_line[0] + v_line[2]) // 2
                corner_y = (h_line[1] + h_line[3]) // 2
                
                # Calculate angle based on which edges we see
                if left_edge is not None and top_edge is not None:
                    angle = 45  # Looking at top-left corner
                elif right_edge is not None and top_edge is not None:
                    angle = 315  # Looking at top-right corner
                elif left_edge is not None and bottom_edge is not None:
                    angle = 135  # Looking at bottom-left corner
                elif right_edge is not None and bottom_edge is not None:
                    angle = 225  # Looking at bottom-right corner
                    
        # Estimate camera position based on edges
        camera_x_estimate = None
        camera_y_estimate = None
        confidence = 0.0
        
        pixels_per_mm_h = w / self.fov_width_at_board
        pixels_per_mm_v = h / self.fov_height_at_board
        
        if left_edge is not None and right_edge is not None:
            # Both edges visible - camera is between them
            visible_width_px = right_edge - left_edge
            visible_width_mm = visible_width_px / pixels_per_mm_h
            
            # Distance from left edge in pixels
            center_from_left_px = (w / 2) - left_edge
            center_from_left_mm = center_from_left_px / pixels_per_mm_h
            
            camera_x_estimate = center_from_left_mm
            confidence += 0.8
            print(f"Both horizontal edges visible: camera estimated at {camera_x_estimate:.1f}mm from left edge")
            
        elif left_edge is not None:
            # Only left edge visible
            edge_from_center_px = left_edge - (w / 2)
            edge_from_center_mm = edge_from_center_px / pixels_per_mm_h
            camera_x_estimate = -edge_from_center_mm
            confidence += 0.4
            print(f"Left edge visible: camera estimated at {camera_x_estimate:.1f}mm from left edge")
            
        elif right_edge is not None:
            # Only right edge visible
            edge_from_center_px = right_edge - (w / 2)
            edge_from_center_mm = edge_from_center_px / pixels_per_mm_h
            camera_x_estimate = self.config.width_mm - (-edge_from_center_mm)
            confidence += 0.4
            print(f"Right edge visible: camera estimated at {camera_x_estimate:.1f}mm from left edge")
        
        # Similar for vertical position
        if top_edge is not None and bottom_edge is not None:
            center_from_top_px = (h / 2) - top_edge
            center_from_top_mm = center_from_top_px / pixels_per_mm_v
            camera_y_estimate = center_from_top_mm
            confidence += 0.2
            
        # Draw edges on visualization with debug info
        edge_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        # Also show gray mask
        edge_vis[:,:,1] = gray_mask // 2  # Show gray areas in green channel
        
        # Draw Hough lines if found
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edge_vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        # Draw vertical sum graph at bottom
        graph_height = 100
        graph = np.zeros((graph_height, w, 3), dtype=np.uint8)
        if np.max(vertical_sum) > 0:
            v_sum_normalized = (vertical_sum * graph_height / np.max(vertical_sum)).astype(int)
            for x in range(w):
                cv2.line(graph, (x, graph_height), (x, graph_height - v_sum_normalized[x]), (0, 255, 0), 1)
        edge_vis[-graph_height:, :] = graph
        
        if left_edge is not None:
            cv2.line(edge_vis, (left_edge, 0), (left_edge, h), (0, 255, 0), 2)
            cv2.putText(edge_vis, "L", (left_edge + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if right_edge is not None:
            cv2.line(edge_vis, (right_edge, 0), (right_edge, h), (0, 255, 0), 2)
            cv2.putText(edge_vis, "R", (right_edge - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if top_edge is not None:
            cv2.line(edge_vis, (0, top_edge), (w, top_edge), (255, 0, 0), 2)
            cv2.putText(edge_vis, "T", (30, top_edge + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if bottom_edge is not None:
            cv2.line(edge_vis, (0, bottom_edge), (w, bottom_edge), (255, 0, 0), 2)
            cv2.putText(edge_vis, "B", (30, bottom_edge - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        # Draw corner if found
        if corner_x is not None and corner_y is not None:
            cv2.circle(edge_vis, (corner_x, corner_y), 10, (0, 0, 255), -1)
            cv2.putText(edge_vis, f"Corner {angle}°", (corner_x + 15, corner_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Return all edge information
        return (camera_x_estimate, camera_y_estimate, confidence, edge_vis, 
                left_edge, right_edge, top_edge, bottom_edge, angle)
        
        # Estimate camera position based on edges
        camera_x_estimate = None
        camera_y_estimate = None
        confidence = 0.0
        
        pixels_per_mm_h = w / self.fov_width_at_board
        pixels_per_mm_v = h / self.fov_height_at_board
        
        if left_edge is not None and right_edge is not None:
            # Both edges visible - camera is between them
            visible_width_px = right_edge - left_edge
            visible_width_mm = visible_width_px / pixels_per_mm_h
            
            # Distance from left edge in pixels
            center_from_left_px = (w / 2) - left_edge
            center_from_left_mm = center_from_left_px / pixels_per_mm_h
            
            camera_x_estimate = center_from_left_mm
            confidence += 0.8
            print(f"Both horizontal edges visible: camera estimated at {camera_x_estimate:.1f}mm from left edge")
            
        elif left_edge is not None:
            # Only left edge visible
            edge_from_center_px = left_edge - (w / 2)
            edge_from_center_mm = edge_from_center_px / pixels_per_mm_h
            camera_x_estimate = -edge_from_center_mm
            confidence += 0.4
            print(f"Left edge visible: camera estimated at {camera_x_estimate:.1f}mm from left edge")
            
        elif right_edge is not None:
            # Only right edge visible
            edge_from_center_px = right_edge - (w / 2)
            edge_from_center_mm = edge_from_center_px / pixels_per_mm_h
            camera_x_estimate = self.config.width_mm - (-edge_from_center_mm)
            confidence += 0.4
            print(f"Right edge visible: camera estimated at {camera_x_estimate:.1f}mm from left edge")
        
        # Similar for vertical position
        if top_edge is not None and bottom_edge is not None:
            center_from_top_px = (h / 2) - top_edge
            center_from_top_mm = center_from_top_px / pixels_per_mm_v
            camera_y_estimate = center_from_top_mm
            confidence += 0.2
            
        # Draw edges on visualization with debug info
        edge_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        # Also show gray mask
        edge_vis[:,:,1] = gray_mask // 2  # Show gray areas in green channel
        
        # Draw vertical sum graph at bottom
        graph_height = 100
        graph = np.zeros((graph_height, w, 3), dtype=np.uint8)
        v_sum_normalized = (vertical_sum * graph_height / np.max(vertical_sum)).astype(int)
        for x in range(w):
            cv2.line(graph, (x, graph_height), (x, graph_height - v_sum_normalized[x]), (0, 255, 0), 1)
        edge_vis[-graph_height:, :] = graph
        
        if left_edge is not None:
            cv2.line(edge_vis, (left_edge, 0), (left_edge, h), (0, 255, 0), 2)
            cv2.putText(edge_vis, "L", (left_edge + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if right_edge is not None:
            cv2.line(edge_vis, (right_edge, 0), (right_edge, h), (0, 255, 0), 2)
            cv2.putText(edge_vis, "R", (right_edge - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if top_edge is not None:
            cv2.line(edge_vis, (0, top_edge), (w, top_edge), (255, 0, 0), 2)
            cv2.putText(edge_vis, "T", (30, top_edge + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if bottom_edge is not None:
            cv2.line(edge_vis, (0, bottom_edge), (w, bottom_edge), (255, 0, 0), 2)
            cv2.putText(edge_vis, "B", (30, bottom_edge - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        return camera_x_estimate, camera_y_estimate, confidence, edge_vis
        
    def detect_markings(self, image) -> List[Marking]:
        """Detect colored markings in the image"""
        markings = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # Create visualization image
        vis_image = image.copy()
        
        # Detect white areas (whiteboard)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Process each color
        for color_name, color_ranges in self.color_ranges.items():
            color_mask = np.zeros_like(white_mask)
            
            for lower, upper in color_ranges:
                color_mask |= cv2.inRange(hsv, lower, upper)
            
            # Only look for markings on the whiteboard
            color_mask = cv2.bitwise_and(color_mask, white_mask)
            
            # Noise reduction - lighter touch to preserve thin lines
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            # Skip erosion to preserve thin lines
            
            # Focus on bottom portion of image where whiteboard is
            # Give more weight to markings in the bottom 2/3 of the image
            weight_mask = np.ones_like(color_mask)
            weight_mask[:h//3, :] = 0.5  # Reduce weight for top third
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            print(f"\n{color_name.upper()} detection: Found {len(contours)} contours")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                print(f"  Contour {i}: area = {area}")
                
                if area < self.min_marking_area:
                    continue
                    
                # Get bounding box
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Draw on visualization
                color_bgr = {
                    'purple': (255, 0, 255),
                    'red': (0, 0, 255),
                    'blue': (255, 0, 0),
                    'green': (0, 255, 0),
                    'black': (128, 128, 128)
                }
                
                cv2.rectangle(vis_image, (x, y), (x + w_box, y + h_box), 
                             color_bgr.get(color_name, (255, 255, 255)), 2)
                cv2.putText(vis_image, f"{color_name} ({area:.0f})", 
                           (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           color_bgr.get(color_name, (255, 255, 255)), 2)
                
                # Calculate position in mm (assuming camera is centered on visible area)
                pixels_per_mm_h = w / self.fov_width_at_board
                pixels_per_mm_v = h / self.fov_height_at_board
                
                center_x_px = x + w_box // 2
                center_y_px = y + h_box // 2
                
                # Convert to mm coordinates (relative to camera position)
                center_x_mm = self.config.camera_x_mm + (center_x_px - w/2) / pixels_per_mm_h
                center_y_mm = self.config.camera_y_mm + (center_y_px - h/2) / pixels_per_mm_v
                
                marking = Marking(
                    center_x_mm=center_x_mm,
                    center_y_mm=center_y_mm,
                    width_mm=w_box / pixels_per_mm_h,
                    height_mm=h_box / pixels_per_mm_v,
                    color=color_name,
                    confidence=min(1.0, area / 1000)
                )
                markings.append(marking)
                
                print(f"  -> Added marking at ({center_x_mm:.1f}, {center_y_mm:.1f}) mm, "
                      f"size: {marking.width_mm:.1f} x {marking.height_mm:.1f} mm")
        
        return markings, vis_image, white_mask

def test_image(image_path: str, output_dir: str = "test_output"):
    """Test the whiteboard vision on a single image"""
    print(f"\nTesting image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return ord('q')  # Return 'q' to skip
        
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure for test (camera at 135mm = 13.5cm height)
    config = WhiteboardConfig(
        width_mm=2000.0,    # Approximate whiteboard size
        height_mm=1000.0,   
        distance_mm=135.0,  # 13.5cm as specified
        camera_x_mm=0.0,    # Initial guess - will be determined by edge detection
        camera_y_mm=0.0,    # Initial guess - will be determined by edge detection
        camera_placement_mm=0.0,  # No RC car for this test
        erasure_radius_mm=50.0
    )
    
    # Create tracker
    tracker = TestWhiteboardTracker(config)
    
    # First detect edges and estimate camera position
    result = tracker.detect_edges_and_position(image)
    camera_x, camera_y, position_confidence, edge_vis = result[:4]
    left_edge, right_edge, top_edge, bottom_edge = result[4:8]
    angle = result[8] if len(result) > 8 else None
    print(f"\nCamera position estimation:")
    if camera_x is not None:
        print(f"  X position: {camera_x:.1f}mm from left edge")
        # Update config with estimated position
        tracker.config.camera_x_mm = camera_x
    else:
        print(f"  X position: Could not determine (no edges visible)")
        # Use center as fallback
        tracker.config.camera_x_mm = config.width_mm / 2
        
    if camera_y is not None:
        print(f"  Y position: {camera_y:.1f}mm from top edge")
        tracker.config.camera_y_mm = camera_y
    else:
        print(f"  Y position: Could not determine (no horizontal edges visible)")
        tracker.config.camera_y_mm = config.height_mm / 2
        
    print(f"  Confidence: {position_confidence:.2f}")
    
    # Now detect markings with known camera position
    markings, vis_image, white_mask = tracker.detect_markings(image)
    
    # Print results
    print(f"\nDetected {len(markings)} markings:")
    for i, marking in enumerate(markings):
        print(f"{i+1}. {marking.color} marking at ({marking.center_x_mm:.1f}, {marking.center_y_mm:.1f}) mm")
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save visualization
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_detected.jpg"), vis_image)
    print(f"Saved visualization to {output_dir}/{base_name}_detected.jpg")
    
    # Save white mask
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_white_mask.jpg"), white_mask)
    
    # Save edge detection
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_edges.jpg"), edge_vis)
    
    # Create color masks visualization
    color_masks = np.zeros((h, w, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Show each color detection
    for color_name, color_ranges in tracker.color_ranges.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in color_ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        
        if color_name == 'purple':
            color_masks[:, :, 2] = mask  # Red channel
        elif color_name == 'blue':
            color_masks[:, :, 0] = mask  # Blue channel
        elif color_name == 'green':
            color_masks[:, :, 1] = mask  # Green channel
            
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_color_masks.jpg"), color_masks)
    
    # Save camera position to JSON
    camera_data = {
        "image": os.path.basename(image_path),
        "camera_position": {
            "x_mm": tracker.config.camera_x_mm,
            "y_mm": tracker.config.camera_y_mm,
            "angle_degrees": angle if angle is not None else 0,
            "confidence": position_confidence
        },
        "edges_detected": {
            "left": left_edge is not None,
            "right": right_edge is not None,
            "top": top_edge is not None,
            "bottom": bottom_edge is not None
        },
        "edge_positions_px": {
            "left": left_edge,
            "right": right_edge,
            "top": top_edge,
            "bottom": bottom_edge
        },
        "markings": [
            {
                "color": m.color,
                "x_mm": m.center_x_mm,
                "y_mm": m.center_y_mm,
                "width_mm": m.width_mm,
                "height_mm": m.height_mm
            } for m in markings
        ]
    }
    
    json_path = os.path.join(output_dir, f"{base_name}_camera_position.json")
    with open(json_path, 'w') as f:
        json.dump(camera_data, f, indent=2)
    print(f"Saved camera position to {json_path}")
    
    # Show results
    windows = [
        ("Original", cv2.resize(image, (800, 600))),
        ("Edge Detection", cv2.resize(edge_vis, (800, 600))),
        ("Detected Markings", cv2.resize(vis_image, (800, 600))),
        ("White Mask", cv2.resize(white_mask, (800, 600))),
        ("Color Masks (R=purple, G=green, B=blue)", cv2.resize(color_masks, (800, 600)))
    ]
    
    for name, img in windows:
        cv2.imshow(name, img)
    
    print("\nPress SPACE for next image, 'q' to quit, any other key to continue...")
    key = cv2.waitKey(0) & 0xFF
    
    # Close windows after key press
    cv2.destroyAllWindows()
    
    # Return key for main loop to handle
    return key

def main():
    """Main test function"""
    print("Whiteboard Vision Test Script")
    print("=============================")
    print("Controls: SPACE = next image, 'q' = quit\n")
    
    # Test with command line arguments or default test images
    if len(sys.argv) > 1:
        # Test specific images
        image_paths = sys.argv[1:]
    else:
        # Look for test images in current directory
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            import glob
            image_paths.extend(glob.glob(ext))
        
        if not image_paths:
            print("Usage: python test_whiteboard_vision.py [image1.jpg] [image2.jpg] ...")
            print("Or place images in the current directory")
            return
    
    print(f"Found {len(image_paths)} images to test")
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing {image_path}")
        key = test_image(image_path)
        
        if key == ord('q'):
            print("\nQuitting...")
            break
    
    cv2.destroyAllWindows()
    print("\nTest complete!")

if __name__ == "__main__":
    main()