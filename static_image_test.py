#!/usr/bin/env python3
"""
Test script for whiteboard vision system using static images
"""

import cv2
import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple

# Import the whiteboard tracker components we need
# (Assuming whiteboard_tracker.py is in the same directory)
from whiteboard_tracker import WhiteboardConfig, Marking, CameraPosition, ObjectPosition

# Simplified version of the tracker for testing with images
class TestWhiteboardTracker:
    def __init__(self, config: WhiteboardConfig):
        self.config = config
        
        # Calculate field of view parameters
        self.calculate_fov_parameters()
        
        # Color ranges for whiteboard detection
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 40, 255])
        
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
        
        # Detect white areas
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find edges
        left_edge = None
        right_edge = None
        top_edge = None
        bottom_edge = None
        
        # Scan for edges
        vertical_sum = np.sum(white_mask, axis=0)
        horizontal_sum = np.sum(white_mask, axis=1)
        
        # Left edge
        for x in range(w // 4):
            if vertical_sum[x] > h * 0.3:
                if x > 0 and vertical_sum[x-1] < h * 0.1:
                    left_edge = x
                    break
                    
        # Right edge
        for x in range(w-1, 3*w//4, -1):
            if vertical_sum[x] > h * 0.3:
                if x < w-1 and vertical_sum[x+1] < h * 0.1:
                    right_edge = x
                    break
                    
        # Top edge
        for y in range(h // 4):
            if horizontal_sum[y] > w * 0.3:
                if y > 0 and horizontal_sum[y-1] < w * 0.1:
                    top_edge = y
                    break
                    
        # Bottom edge
        for y in range(h-1, 3*h//4, -1):
            if horizontal_sum[y] > w * 0.3:
                if y < h-1 and horizontal_sum[y+1] < w * 0.1:
                    bottom_edge = y
                    break
        
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
            
        # Draw edges on visualization
        edge_vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        if left_edge is not None:
            cv2.line(edge_vis, (left_edge, 0), (left_edge, h), (0, 255, 0), 2)
        if right_edge is not None:
            cv2.line(edge_vis, (right_edge, 0), (right_edge, h), (0, 255, 0), 2)
        if top_edge is not None:
            cv2.line(edge_vis, (0, top_edge), (w, top_edge), (255, 0, 0), 2)
        if bottom_edge is not None:
            cv2.line(edge_vis, (0, bottom_edge), (w, bottom_edge), (255, 0, 0), 2)
            
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
            
            # Noise reduction
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
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
        return
        
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
    camera_x, camera_y, position_confidence, edge_vis = tracker.detect_edges_and_position(image)
    
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
    
    # Show results
    cv2.imshow("Original", cv2.resize(image, (800, 600)))
    cv2.imshow("Edge Detection", cv2.resize(edge_vis, (800, 600)))
    cv2.imshow("Detected Markings", cv2.resize(vis_image, (800, 600)))
    cv2.imshow("White Mask", cv2.resize(white_mask, (800, 600)))
    cv2.imshow("Color Masks (R=purple, G=green, B=blue)", cv2.resize(color_masks, (800, 600)))
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)

def main():
    """Main test function"""
    print("Whiteboard Vision Test Script")
    print("=============================")
    
    # Test with command line arguments or default test images
    if len(sys.argv) > 1:
        # Test specific images
        for image_path in sys.argv[1:]:
            test_image(image_path)
    else:
        # Look for test images in current directory
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            import glob
            test_images.extend(glob.glob(ext))
        
        if test_images:
            print(f"Found {len(test_images)} images to test")
            for image_path in test_images:
                test_image(image_path)
        else:
            print("Usage: python test_whiteboard_vision.py [image1.jpg] [image2.jpg] ...")
            print("Or place images in the current directory")
    
    cv2.destroyAllWindows()
    print("\nTest complete!")

if __name__ == "__main__":
    main()