import cv2
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
import time

class WhiteboardTracker5:
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Optimized kernels for RPi5
        self.kernel_3x3 = np.ones((3, 3), np.uint8)
        self.kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Optimize OpenCV for RPi5
        cv2.setUseOptimized(True)
        cv2.setNumThreads(2)  # Leave 2 cores for system/other processes
        
        # Performance tracking
        self.processing_times = []
        
        # Scene analysis for adaptive processing
        self.is_low_contrast_scene = False
    
    def analyze_background_contrast(self, image: np.ndarray) -> bool:
        """Analyze if this is a low contrast scene (whiteboard vs light background)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Sample different regions to assess overall contrast
        bottom_region = gray[int(h * 0.4):, :]  # Whiteboard area
        top_region = gray[:int(h * 0.4), :]     # Background/wall area
        
        # Calculate statistics for each region
        bottom_mean = bottom_region.mean()
        bottom_std = bottom_region.std()
        top_mean = top_region.mean()
        top_std = top_region.std()
        
        # Calculate contrast metrics
        intensity_difference = abs(bottom_mean - top_mean)
        combined_std = (bottom_std + top_std) / 2
        overall_range = gray.max() - gray.min()
        
        # More conservative low contrast detection to avoid false classification
        # Calculate additional metrics for better classification
        whiteboard_std = bottom_region.std()
        background_std = top_region.std()
        local_contrast_ratio = intensity_difference / max(combined_std, 1)
        
        # True low contrast scenarios have:
        # 1. Very small intensity difference AND high brightness
        # 2. Low local contrast variation
        # 3. Both regions very bright with minimal texture
        
        is_very_low_diff = intensity_difference < 25  # Stricter threshold
        is_very_bright = (bottom_mean + top_mean) / 2 > 180  # Higher brightness requirement
        is_low_texture = whiteboard_std < 15 and background_std < 15  # Smooth regions
        is_poor_local_contrast = local_contrast_ratio < 2.0  # Poor contrast ratio
        
        # Require multiple conditions for low contrast classification
        is_low_contrast = (is_very_low_diff and is_very_bright and 
                          (is_low_texture or is_poor_local_contrast))
        
        if self.debug:
            print(f"  Scene analysis: intensity_diff={intensity_difference:.1f}, "
                  f"brightness={(bottom_mean + top_mean) / 2:.1f}, "
                  f"contrast_ratio={local_contrast_ratio:.2f}, "
                  f"low_contrast={is_low_contrast}")
        
        return is_low_contrast
        
    def find_whiteboard_surface_improved(self, image: np.ndarray) -> np.ndarray:
        """Improved whiteboard surface detection focusing on actual whiteboard area"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Focus on bottom 60% where whiteboard surface appears (more focused)
        bottom_start = int(h * 0.4)  # Changed to 0.4 to focus on bottom 60%
        bottom_region = gray[bottom_start:, :]
        
        if self.debug:
            print(f"  Processing region: {w}x{h-bottom_start} (bottom 60%)")
        
        # Use adaptive thresholding for better whiteboard detection
        # Whiteboard should be consistently bright
        p75 = np.percentile(bottom_region, 75)
        p90 = np.percentile(bottom_region, 90)
        
        # Use adaptive threshold - try strict first, then fallback to more lenient
        strict_threshold = max(180, p90 * 0.9)
        lenient_threshold = max(160, p75 * 0.8)
        
        # Try strict threshold first
        surface_mask = ((bottom_region > strict_threshold) * 255).astype(np.uint8)
        
        # More aggressive cleanup to remove noise
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_OPEN, self.kernel_3x3)
        surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, self.kernel_5x5)
        
        # Keep only largest connected component that's reasonably sized
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(surface_mask, connectivity=8)
        
        largest_area = 0
        min_whiteboard_area = (h - bottom_start) * w * 0.1  # At least 10% of region
        
        if num_labels > 1:
            # Find largest component that's also reasonably sized for a whiteboard
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            largest_area = areas[largest_idx - 1]
            
            if largest_area > min_whiteboard_area:
                surface_mask = ((labels == largest_idx) * 255).astype(np.uint8)
            else:
                if self.debug:
                    print(f"    Strict threshold failed: {largest_area} < {min_whiteboard_area}")
                # Try lenient threshold as fallback
                surface_mask = ((bottom_region > lenient_threshold) * 255).astype(np.uint8)
                surface_mask = cv2.morphologyEx(surface_mask, cv2.MORPH_CLOSE, self.kernel_5x5)
                
                num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(surface_mask, connectivity=8)
                if num_labels2 > 1:
                    areas2 = stats2[1:, cv2.CC_STAT_AREA]
                    largest_idx2 = np.argmax(areas2) + 1
                    largest_area = areas2[largest_idx2 - 1]
                    
                    if largest_area > min_whiteboard_area:
                        surface_mask = ((labels2 == largest_idx2) * 255).astype(np.uint8)
                        if self.debug:
                            print(f"    Fallback threshold succeeded: {largest_area}")
                    else:
                        if self.debug:
                            print(f"    Both thresholds failed: {largest_area}")
                        return np.zeros_like(gray, dtype=np.uint8)
        
        # Create full-size mask
        full_mask = np.zeros_like(gray, dtype=np.uint8)
        full_mask[bottom_start:, :] = surface_mask
        
        surface_pixels = np.count_nonzero(full_mask)
        if self.debug:
            print(f"  Surface pixels: {surface_pixels} ({surface_pixels/(h*w)*100:.1f}% of image)")
        
        return full_mask
    
    def detect_whiteboard_boundary_edges(self, image: np.ndarray, surface_mask: np.ndarray) -> np.ndarray:
        """Detect edges specifically where whiteboard surface terminates (not frame edges)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # NEW APPROACH: Find edges of the surface mask itself (where surface terminates)
        # This detects surface boundaries, not general image edges that might be frame
        
        # Apply slight blur to surface mask to smooth boundaries
        surface_blurred = cv2.GaussianBlur(surface_mask.astype(np.float32), (3, 3), 1.0)
        surface_blurred = (surface_blurred > 127).astype(np.uint8) * 255
        
        # Find edges of the surface mask (where whiteboard surface ends)
        surface_edges = cv2.Canny(surface_blurred, 50, 150, apertureSize=3)
        
        # Create a narrow band around surface edges to validate against image content
        dilated_surface_edges = cv2.dilate(surface_edges, self.kernel_3x3, iterations=2)
        
        # Also do traditional edge detection but only in the dilated surface edge region
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        image_edges = cv2.Canny(blurred, 30, 90, apertureSize=3)
        
        # Combine: Surface boundaries where there's also supporting image evidence
        # This filters out noise while keeping real surface termination edges
        boundary_edges = cv2.bitwise_and(surface_edges, dilated_surface_edges)
        supporting_edges = cv2.bitwise_and(image_edges, dilated_surface_edges)
        
        # Final edges: Surface boundaries with image support, or strong surface boundaries
        boundary_edges = cv2.bitwise_or(boundary_edges, supporting_edges)
        
        # Focus on bottom portion where actual whiteboard edges should be
        # Zero out top 30% to avoid ceiling/wall edges
        boundary_edges[:int(h * 0.3), :] = 0
        
        # Filter edges by position - whiteboard edges should be in specific locations
        filtered_edges = self.filter_whiteboard_edges_by_geometry(boundary_edges, h, w)
        
        edge_pixels = np.count_nonzero(filtered_edges)
        if self.debug:
            print(f"  Boundary edge pixels: {edge_pixels}")
            
        return filtered_edges
    
    def filter_whiteboard_edges_by_geometry(self, edges: np.ndarray, h: int, w: int) -> np.ndarray:
        """Filter edges based on expected whiteboard geometry"""
        # Find connected components in edges
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        
        filtered_edges = np.zeros_like(edges)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 10:  # Too small to be a meaningful edge
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Relaxed filtering - focus on length and position
            min_dimension = min(width, height)
            max_dimension = max(width, height)
            aspect_ratio = max_dimension / max(min_dimension, 1)
            
            # More lenient position check (bottom 60% instead of 70%)
            is_in_likely_region = y > h * 0.4
            
            # More lenient size check 
            is_reasonably_long = max_dimension > min(w, h) * 0.08  # Reduced from 0.1
            
            # More lenient aspect ratio (edges can be less linear)
            is_somewhat_linear = aspect_ratio > 1.5  # Reduced from 2.0
            
            # Keep most edges that could plausibly be whiteboard boundaries
            # Don't be too restrictive about left/right positioning
            
            if (is_in_likely_region and is_reasonably_long and is_somewhat_linear):
                filtered_edges[labels == i] = 255
                
                if self.debug:
                    print(f"    Kept edge: area={area}, aspect={aspect_ratio:.1f}, pos=({x},{y})")
        
        return filtered_edges
    
    def detect_edges_gradient_based(self, image: np.ndarray, surface_mask: np.ndarray) -> np.ndarray:
        """Gradient-based edge detection for low contrast scenarios"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Create boundary region but use broader area for low contrast
        eroded = cv2.erode(surface_mask, self.kernel_3x3, iterations=1)
        dilated = cv2.dilate(surface_mask, self.kernel_5x5, iterations=3)  # Broader dilation
        boundary_region = cv2.subtract(dilated, eroded)
        
        # Use multiple gradient-based approaches
        # Method 1: Sobel gradients with lower threshold
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Use lower percentile threshold for low contrast
        sobel_thresh = np.percentile(sobel_magnitude, 85)  # Lower than usual 94%
        sobel_edges = ((sobel_magnitude > sobel_thresh) * 255).astype(np.uint8)
        
        # Method 2: Laplacian edge detection for fine details
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian_magnitude = np.abs(laplacian)
        laplacian_thresh = np.percentile(laplacian_magnitude, 90)
        laplacian_edges = ((laplacian_magnitude > laplacian_thresh) * 255).astype(np.uint8)
        
        # Method 3: Lower threshold Canny for subtle edges
        low_canny = cv2.Canny(gray, 20, 60, apertureSize=3)  # Much lower thresholds
        
        # Combine edge detection methods
        combined_edges = cv2.bitwise_or(sobel_edges, laplacian_edges)
        combined_edges = cv2.bitwise_or(combined_edges, low_canny)
        
        # Apply boundary region mask
        boundary_edges = cv2.bitwise_and(combined_edges, boundary_region)
        
        # Focus on bottom region where whiteboard edges should be
        boundary_edges[:int(h * 0.4), :] = 0
        
        # Light morphological cleanup
        boundary_edges = cv2.morphologyEx(boundary_edges, cv2.MORPH_CLOSE, self.kernel_3x3)
        
        # Apply geometric filtering but with more relaxed constraints for low contrast
        filtered_edges = self.filter_edges_low_contrast(boundary_edges, h, w)
        
        edge_pixels = np.count_nonzero(filtered_edges)
        if self.debug:
            print(f"  Low-contrast edge pixels: {edge_pixels}")
            
        return filtered_edges
    
    def filter_edges_low_contrast(self, edges: np.ndarray, h: int, w: int) -> np.ndarray:
        """More permissive edge filtering for low contrast scenarios"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        
        filtered_edges = np.zeros_like(edges)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 5:  # Even lower threshold for low contrast
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Very relaxed constraints for low contrast
            min_dimension = min(width, height)
            max_dimension = max(width, height)
            aspect_ratio = max_dimension / max(min_dimension, 1)
            
            # More permissive position and size checks
            is_in_likely_region = y > h * 0.1  # Very lenient position
            is_reasonably_long = max_dimension > min(w, h) * 0.05  # Smaller minimum
            is_somewhat_linear = aspect_ratio > 1.2  # Very relaxed linearity
            
            if (is_in_likely_region and is_reasonably_long and is_somewhat_linear):
                filtered_edges[labels == i] = 255
                
                if self.debug:
                    print(f"    Kept low-contrast edge: area={area}, aspect={aspect_ratio:.1f}, pos=({x},{y})")
        
        return filtered_edges
    
    def is_surface_edge(self, rho: float, theta: float, h: int, w: int) -> bool:
        """Determine if a line represents a surface edge vs a frame edge"""
        angle_deg = np.degrees(theta)
        a, b = np.cos(theta), np.sin(theta)
        
        # Surface edges should be at image boundaries or near them
        # Frame edges are typically internal to the image
        
        if abs(b) > 0.001:  # Non-vertical line (horizontal-ish)
            # For horizontal lines, check if they're in the lower portion
            # Surface edges extend to bottom, frame edges are typically higher up
            y_left = rho / b if abs(b) > 0.001 else 0
            y_right = (rho - w * a) / b if abs(b) > 0.001 else 0
            avg_y = (y_left + y_right) / 2
            
            # Surface edges should be in bottom 70% of image
            # Frame edges are typically in middle region
            if avg_y < h * 0.3:  # Too high up - likely frame edge
                return False
                
        if abs(a) > 0.001:  # Non-horizontal line (vertical-ish)
            # For vertical lines, check if they extend to image edges
            x_pos = rho / a if abs(a) > 0.001 else 0
            
            # Surface edges should be near left or right boundaries
            # Frame edges are typically more central
            edge_margin = w * 0.15  # Within 15% of edges
            is_near_left_edge = x_pos < edge_margin
            is_near_right_edge = x_pos > (w - edge_margin)
            
            if not (is_near_left_edge or is_near_right_edge):
                return False  # Too central - likely frame edge
        
        return True
    
    def validate_line_geometry(self, rho: float, theta: float, h: int, w: int) -> bool:
        """Validate if a line makes geometric sense for a whiteboard boundary"""
        angle_deg = np.degrees(theta)
        
        # More permissive orientation check for subtle edges
        is_reasonable_angle = (
            abs(angle_deg) < 35 or abs(angle_deg - 90) < 35 or  # Horizontal or vertical ±35°
            abs(angle_deg - 180) < 35 or  # Horizontal wraparound
            (25 < angle_deg < 65) or (115 < angle_deg < 155)  # Reasonable diagonals
        )
        
        if not is_reasonable_angle:
            return False
        
        # Position-based validation
        a, b = np.cos(theta), np.sin(theta)
        
        # Check if line position makes sense
        if abs(b) > 0.001:  # Non-vertical line
            # Calculate y-positions at left and right edges
            y_left = rho / b
            y_right = (rho - w * a) / b
            
            # Line should intersect image in reasonable region (bottom 60%)
            # Check if ANY part of the line is in valid region
            min_y = min(y_left, y_right)
            max_y = max(y_left, y_right)
            
            # Line should pass through the image bounds and be in reasonable region
            line_in_image = (min_y <= h and max_y >= 0)
            line_in_whiteboard_region = (max_y > h * 0.4)  # At least part in bottom 60%
            
            if not (line_in_image and line_in_whiteboard_region):
                return False
        
        if abs(a) > 0.001:  # Non-horizontal line  
            # For vertical-ish lines, check if line passes through reasonable x range
            # Calculate x positions at top and bottom of image
            x_top = (rho - 0 * b) / a if abs(b) < 0.001 else rho / a
            x_bottom = (rho - h * b) / a
            
            min_x = min(x_top, x_bottom)  
            max_x = max(x_top, x_bottom)
            
            # Line should intersect the image in x direction
            line_intersects_image = (min_x <= w and max_x >= 0)
            
            if not line_intersects_image:
                return False
        
        return True
    
    def detect_right_edge_simple(self, image: np.ndarray, surface_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Simple right edge detection for wall-mounted whiteboards"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Focus on right 40% where vertical edge should be
        right_start = int(w * 0.6)
        right_region = gray[:, right_start:]
        right_mask_region = surface_mask[:, right_start:]
        
        if np.count_nonzero(right_mask_region) < 50:
            return None
        
        # Look for vertical edges using simple gradient
        grad_x = cv2.Sobel(right_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_magnitude = np.abs(grad_x)
        
        # Find strongest vertical gradient column
        column_strengths = np.mean(grad_magnitude[int(h*0.3):, :], axis=0)  # Bottom 70% only
        
        if len(column_strengths) == 0:
            return None
        
        # Find peak column and convert to global coordinates
        peak_col = np.argmax(column_strengths)
        global_x = right_start + peak_col
        
        # Check if gradient is strong enough
        if column_strengths[peak_col] > np.percentile(column_strengths, 70):
            # Return vertical line at this x position
            rho = float(global_x)
            theta = np.pi / 2  # 90 degrees = vertical line
            return (rho, theta)
        
        return None
    
    def find_whiteboard_lines(self, edges: np.ndarray) -> List[Tuple[float, float]]:
        """Simplified line detection with slope-based filtering"""
        h, w = edges.shape
        
        if np.count_nonzero(edges) < 20:
            return []
        
        # More sensitive Hough threshold to catch weaker edges
        edge_pixel_count = np.count_nonzero(edges)
        threshold = max(10, min(20, edge_pixel_count // 5))
        
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=threshold)
        
        if self.debug:
            print(f"    Using Hough threshold: {threshold} (edge pixels: {edge_pixel_count})")
        
        if lines is None:
            return []
        
        if self.debug:
            print(f"  Raw Hough lines: {len(lines)}")
        
        valid_lines = []
        
        for line in lines[:10]:  # Limit processing to first 10 lines
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            
            # Simple slope classification
            is_diagonal = 15 <= abs(angle_deg) <= 75 or 105 <= abs(angle_deg) <= 165
            is_horizontal = abs(angle_deg) < 15 or abs(angle_deg - 180) < 15
            is_vertical = 75 < angle_deg < 105
            
            # Accept reasonable edge orientations
            if is_diagonal or is_horizontal or is_vertical:
                # Geometric validation first
                if not self.validate_line_geometry(rho, theta, h, w):
                    if self.debug:
                        print(f"    Rejected (bad geometry): angle={angle_deg:.1f}°, rho={rho:.1f}")
                    continue
                
                # Check if this is a surface edge vs frame edge
                if not self.is_surface_edge(rho, theta, h, w):
                    if self.debug:
                        print(f"    Rejected (frame edge, not surface edge): angle={angle_deg:.1f}°, rho={rho:.1f}")
                    continue
                
                # Quick line quality check
                support_score = self.score_line_fast(rho, theta, edges, h, w)
                
                # Lower support threshold to detect subtle edges
                min_support = 0.03 if self.is_low_contrast_scene else 0.05
                if support_score > min_support:
                    valid_lines.append((rho, theta, support_score))
                    if self.debug:
                        print(f"    Valid line: angle={angle_deg:.1f}°, rho={rho:.1f}, support={support_score:.3f}")
                elif self.debug:
                    print(f"    Rejected (low support): angle={angle_deg:.1f}°, support={support_score:.3f}")
        
        # Sort by support score and return best lines
        valid_lines.sort(key=lambda x: x[2], reverse=True)
        
        # Improved deduplication for low contrast scenarios
        final_lines = []
        if valid_lines:
            final_lines.append(valid_lines[0][:2])  # Best line
            
            # Look for additional lines with significantly different slopes
            used_angles = [np.degrees(valid_lines[0][1])]
            
            for rho, theta, score in valid_lines[1:]:
                angle = np.degrees(theta)
                
                # Check if this angle is significantly different from all used angles
                is_different = True
                for used_angle in used_angles:
                    angle_diff = min(abs(used_angle - angle), 
                                   abs(used_angle - angle + 180),
                                   abs(used_angle - angle - 180))
                    
                    # Use smaller angle difference for low contrast to allow more diverse lines
                    min_angle_diff = 15 if self.is_low_contrast_scene else 20
                    if angle_diff <= min_angle_diff:
                        is_different = False
                        break
                
                if is_different:
                    final_lines.append((rho, theta))
                    used_angles.append(angle)
                    
                    # For low contrast, try to find up to 3 lines instead of just 2
                    max_lines = 3 if self.is_low_contrast_scene else 2
                    if len(final_lines) >= max_lines:
                        break
        
        if self.debug:
            print(f"  Final lines: {len(final_lines)}")
        
        return final_lines
    
    def score_line_fast(self, rho: float, theta: float, edges: np.ndarray, h: int, w: int) -> float:
        """Fast line scoring using sampling"""
        a, b = np.cos(theta), np.sin(theta)
        hits = 0
        samples = 0
        
        # Sample every 3rd pixel for speed
        sample_step = 3
        
        if abs(b) > abs(a):  # More horizontal line
            for x in range(0, w, sample_step):
                y = (rho - x * a) / b
                if 0 <= y < h:
                    y_int = int(y + 0.5)
                    if 0 <= y_int < h:
                        samples += 1
                        if edges[y_int, x] > 0:
                            hits += 1
        else:  # More vertical line  
            for y in range(0, h, sample_step):
                x = (rho - y * b) / a
                if 0 <= x < w:
                    x_int = int(x + 0.5)
                    if 0 <= x_int < w:
                        samples += 1
                        if edges[y, x_int] > 0:
                            hits += 1
        
        return hits / max(samples, 1)
    
    def detect_markings_20x20(self, image: np.ndarray, surface_mask: np.ndarray, 
                             edge_lines: List[Tuple[float, float]] = None) -> List[Tuple[int, int]]:
        """Detect dry-erase markings using contrast-based detection within whiteboard boundaries"""
        markings_start_time = time.time()
        h, w = image.shape[:2]
        
        # Require detected edges to proceed - no edges means no valid whiteboard area
        if not edge_lines:
            if self.debug:
                print("  No edge lines provided - skipping marking detection")
            return []
        
        # Create whiteboard interior mask based on actual detected edge geometry
        interior_mask = self.create_whiteboard_interior_mask(image, edge_lines)
        
        if np.count_nonzero(interior_mask) < 100:
            if self.debug:
                print("  Insufficient whiteboard interior area detected")
            return []
        
        # Create exclusion zones around detected edge lines to avoid detecting edges as markings
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        buffer_distance = 25  # Pixels to exclude around edge lines
        
        for rho, theta in edge_lines:
            a, b = np.cos(theta), np.sin(theta)
            
            # Create thick line around the detected edge
            for offset in range(-buffer_distance, buffer_distance + 1):
                offset_rho = rho + offset
                
                # Draw line with offset
                for x in range(w):
                    if abs(b) > 0.001:
                        y_line = (offset_rho - x * a) / b
                        if 0 <= y_line < h:
                            exclusion_mask[int(y_line), x] = 255
                
                # Also handle nearly vertical lines
                if abs(a) > 0.001:
                    for y in range(h):
                        x_line = (offset_rho - y * b) / a
                        if 0 <= x_line < w:
                            exclusion_mask[y, int(x_line)] = 255
        
        # Apply morphological operations to create buffer zones
        kernel_buffer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_distance//2, buffer_distance//2))
        exclusion_mask = cv2.dilate(exclusion_mask, kernel_buffer, iterations=1)
        
        # Final search area: whiteboard interior minus exclusion zones
        search_mask = cv2.bitwise_and(interior_mask, cv2.bitwise_not(exclusion_mask))
        
        if self.debug:
            search_pixels = np.count_nonzero(search_mask)
            excluded_pixels = np.count_nonzero(exclusion_mask)
            interior_pixels = np.count_nonzero(interior_mask)
            print(f"  Whiteboard interior: {interior_pixels} pixels")
            print(f"  Excluded around edges: {excluded_pixels} pixels") 
            print(f"  Final search area: {search_pixels} pixels")
        
        # Now use contrast-based detection instead of HSV color detection
        marking_boxes = self.detect_contrast_markings(image, search_mask)
        
        markings_time = time.time() - markings_start_time
        if self.debug:
            print(f"  Total marking detection time: {markings_time:.4f}s")
        
        return marking_boxes
    
    def detect_contrast_markings(self, image: np.ndarray, search_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Detect markings using contrast-based analysis - color agnostic approach"""
        h, w = image.shape[:2]
        
        # Convert to grayscale for contrast analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        search_region = cv2.bitwise_and(gray, gray, mask=search_mask)
        
        # Calculate whiteboard surface statistics within search area
        if np.count_nonzero(search_mask) == 0:
            return []
        
        wb_pixels = search_region[search_mask > 0]
        wb_mean = np.mean(wb_pixels)
        wb_std = np.std(wb_pixels)
        
        if self.debug:
            print(f"  Whiteboard statistics: mean={wb_mean:.1f}, std={wb_std:.1f}")
        
        # Multi-level adaptive thresholding to catch both bold and faint markings
        all_markings = np.zeros((h, w), dtype=np.uint8)
        
        # Level 1: Bold markings (significant contrast)
        bold_threshold = wb_mean - max(30, wb_std * 2.0)  # Markings significantly darker than surface
        bold_mask = (search_region < bold_threshold) & (search_mask > 0)
        bold_markings = bold_mask.astype(np.uint8) * 255
        
        # Level 2: Medium markings (moderate contrast) 
        medium_threshold = wb_mean - max(20, wb_std * 1.5)
        medium_mask = (search_region < medium_threshold) & (search_mask > 0)
        medium_markings = medium_mask.astype(np.uint8) * 255
        
        # Level 3: Faint markings (subtle contrast) - for cases like the obvious purple marking
        faint_threshold = wb_mean - max(15, wb_std * 1.0)
        faint_mask = (search_region < faint_threshold) & (search_mask > 0)
        faint_markings = faint_mask.astype(np.uint8) * 255
        
        if self.debug:
            bold_pixels = np.count_nonzero(bold_markings)
            medium_pixels = np.count_nonzero(medium_markings)
            faint_pixels = np.count_nonzero(faint_markings)
            print(f"  Thresholds - Bold: {bold_threshold:.1f} ({bold_pixels} px), Medium: {medium_threshold:.1f} ({medium_pixels} px), Faint: {faint_threshold:.1f} ({faint_pixels} px)")
        
        # Process each level with appropriate validation
        for level, (marking_mask, level_name) in enumerate([
            (bold_markings, "bold"),
            (medium_markings, "medium"), 
            (faint_markings, "faint")
        ]):
            if np.count_nonzero(marking_mask) == 0:
                continue
                
            # Clean up noise with morphological operations
            if level == 0:  # Bold markings - minimal cleanup
                cleaned = cv2.morphologyEx(marking_mask, cv2.MORPH_OPEN, self.kernel_3x3)
            elif level == 1:  # Medium markings - moderate cleanup
                cleaned = cv2.morphologyEx(marking_mask, cv2.MORPH_OPEN, self.kernel_3x3)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_3x3)
            else:  # Faint markings - more aggressive cleanup
                cleaned = cv2.morphologyEx(marking_mask, cv2.MORPH_OPEN, self.kernel_3x3)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_5x5)
            
            # Connected component analysis and validation
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                
                # Size validation - adaptive based on level
                min_area = 8 if level == 2 else 12  # Lower threshold for faint markings
                max_area = 50000  # Prevent huge regions
                if area < min_area or area > max_area:
                    if self.debug:
                        print(f"    {level_name} component {i}: REJECTED size ({area} pixels)")
                    continue
                
                # Shape validation
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                if aspect_ratio > 15:  # Allow slightly more elongated shapes than color detection
                    if self.debug:
                        print(f"    {level_name} component {i}: REJECTED shape (ratio {aspect_ratio:.1f})")
                    continue
                
                # Position validation - ensure not too close to search area boundaries
                margin = 5  # Smaller margin since we're already within detected boundaries
                if (x < margin or y < margin or 
                    x + width > w - margin or y + height > h - margin):
                    continue
                
                # Add valid component to final markings
                component_mask = (labels == i).astype(np.uint8) * 255
                all_markings = cv2.bitwise_or(all_markings, component_mask)
                
                if self.debug:
                    print(f"    {level_name} component {i}: ACCEPTED ({area} pixels, {width}x{height})")
        
        # Extract marking centers and create 20x20 boxes
        marking_boxes = self.extract_marking_boxes_20x20(all_markings)
        
        if self.debug:
            total_pixels = np.count_nonzero(all_markings)
            print(f"  Final contrast markings: {total_pixels} pixels, {len(marking_boxes)} boxes")
        
        return marking_boxes
    
    def extract_marking_boxes_20x20(self, marking_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert marking pixels to 20x20 box centers using simple clustering"""
        if np.count_nonzero(marking_mask) == 0:
            return []
        
        # Find all marking pixels
        y_coords, x_coords = np.where(marking_mask > 0)
        marking_points = list(zip(x_coords, y_coords))
        
        if not marking_points:
            return []
        
        boxes = []
        used = set()
        box_size = 20
        half_box = box_size // 2
        
        for x, y in marking_points:
            if (x, y) in used:
                continue
            
            # This point becomes center of new 20x20 box
            boxes.append((x, y))
            
            # Mark all points within this box as used
            for px, py in marking_points:
                if (px, py) not in used:
                    if abs(px - x) <= half_box and abs(py - y) <= half_box:
                        used.add((px, py))
        
        return boxes
    
    def detect_whiteboard_edges(self, image: np.ndarray) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]]:
        """Main detection function optimized for RPi5 performance"""
        if image is None:
            return None
        
        start_time = time.time()
        
        try:
            # Step 0: Analyze background contrast to choose appropriate method
            self.is_low_contrast_scene = self.analyze_background_contrast(image)
            
            # Step 1: Improved surface detection
            surface_mask = self.find_whiteboard_surface_improved(image)
            
            if np.count_nonzero(surface_mask) < 1000:  # Higher threshold for better accuracy
                if self.debug:
                    print("  Insufficient whiteboard surface area detected")
                return None
            
            # Step 2: Hybrid edge detection strategy - try standard first, then supplement if needed
            if self.debug:
                print("  Trying standard edge detection first")
            
            # Always try standard detection first
            edges = self.detect_whiteboard_boundary_edges(image, surface_mask)
            edge_lines = self.find_whiteboard_lines(edges)
            
            # If insufficient lines found, supplement with gradient-based detection
            if len(edge_lines) < 2 and self.is_low_contrast_scene:
                if self.debug:
                    print(f"  Standard detection found {len(edge_lines)} lines, supplementing with gradient-based")
                
                # Try gradient-based detection 
                gradient_edges = self.detect_edges_gradient_based(image, surface_mask)
                gradient_lines = self.find_whiteboard_lines(gradient_edges)
                
                # Combine results, avoiding duplicates
                combined_lines = edge_lines.copy()
                for grad_rho, grad_theta in gradient_lines:
                    is_duplicate = False
                    grad_angle = np.degrees(grad_theta)
                    
                    for exist_rho, exist_theta in combined_lines:
                        exist_angle = np.degrees(exist_theta)
                        angle_diff = min(abs(grad_angle - exist_angle),
                                       abs(grad_angle - exist_angle + 180),
                                       abs(grad_angle - exist_angle - 180))
                        rho_diff = abs(grad_rho - exist_rho)
                        
                        if angle_diff < 10 and rho_diff < 20:  # Similar line
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        combined_lines.append((grad_rho, grad_theta))
                
                edge_lines = combined_lines[:3]  # Limit to max 3 lines
                
                if self.debug:
                    print(f"  Combined detection: {len(edge_lines)} total lines")
            else:
                if self.debug:
                    print(f"  Standard detection sufficient: {len(edge_lines)} lines")
            
            # Step 3: Targeted right edge detection for wall scenarios
            if len(edge_lines) == 1 and self.is_low_contrast_scene:
                if self.debug:
                    print("  Attempting targeted right edge detection for wall scenario")
                
                right_edge = self.detect_right_edge_simple(image, surface_mask)
                if right_edge is not None:
                    edge_lines.append(right_edge)
                    if self.debug:
                        print("  Added right edge candidate")
            
            # Step 4: Improved marking detection with 20x20 boxes
            marking_boxes = self.detect_markings_20x20(image, surface_mask, edge_lines)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if self.debug:
                print(f"  Processing time: {processing_time:.3f}s")
                rpi5_estimate = processing_time * 15  # Conservative 15x slower estimate
                rpi5_fps = 1.0 / rpi5_estimate if rpi5_estimate > 0 else 0
                print(f"  RPi5 estimate: {rpi5_estimate:.3f}s (~{rpi5_fps:.1f} FPS)")
                print(f"  Results: {len(edge_lines)} edge lines, {len(marking_boxes)} marking boxes")
            
            if not edge_lines:
                if self.debug:
                    print("  No valid whiteboard edges detected")
                return None
                
            return edge_lines, marking_boxes
            
        except Exception as e:
            if self.debug:
                print(f"  Error: {e}")
            return None
    
    def get_average_processing_time(self) -> float:
        """Return average processing time for performance monitoring"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def create_whiteboard_interior_mask(self, image: np.ndarray, edge_lines: List[Tuple[float, float]]) -> np.ndarray:
        """Create mask for whiteboard interior area based on detected edge geometry"""
        mask_start_time = time.time()
        h, w = image.shape[:2]
        interior_mask = np.zeros((h, w), dtype=np.uint8)
        
        if not edge_lines:
            return interior_mask
        
        # Find left and right edge boundaries
        left_boundary = None
        right_boundary = None
        top_boundary = 0  # Start with image top (whiteboard is below edges)
        
        for rho, theta in edge_lines:
            angle_deg = np.degrees(theta)
            
            # Classify edge orientation
            is_vertical_ish = 75 < abs(angle_deg) < 105 or 75 < abs(angle_deg - 180) < 105
            is_horizontal_ish = abs(angle_deg) < 15 or abs(angle_deg - 180) < 15
            
            if is_vertical_ish:
                # Vertical edge - could be left or right boundary
                a, b = np.cos(theta), np.sin(theta)
                if abs(a) > 0.001:
                    x_pos = rho / a  # X position of vertical line
                    if x_pos < w/2:  # Left side of image
                        if left_boundary is None or x_pos > left_boundary:
                            left_boundary = x_pos
                    else:  # Right side of image  
                        if right_boundary is None or x_pos < right_boundary:
                            right_boundary = x_pos
            
            elif is_horizontal_ish:
                # Horizontal edge - likely top boundary (whiteboard is below this)
                a, b = np.cos(theta), np.sin(theta)
                if abs(b) > 0.001:
                    # Calculate average Y position across image width
                    y_positions = []
                    for x in range(0, w, w//10):  # Sample 10 points
                        y_line = (rho - x * a) / b
                        if 0 <= y_line < h:
                            y_positions.append(y_line)
                    if y_positions:
                        avg_y = sum(y_positions) / len(y_positions)
                        if avg_y > top_boundary:  # Find the lowest horizontal edge
                            top_boundary = int(avg_y)
        
        # Vectorized approach: create coordinate grids once
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Start with all pixels as potential interior (will be filtered down)
        is_below_all_edges = np.ones((h, w), dtype=bool)
        
        # Check each edge line using vectorized operations
        for rho, theta in edge_lines:
            a, b = np.cos(theta), np.sin(theta)
            
            # For non-vertical lines, check if pixels are below the line
            if abs(b) > 0.001:  # Non-vertical line
                # Vectorized calculation: edge_y_at_x = (rho - x * a) / b
                edge_y_grid = (rho - x_coords * a) / b
                
                # Pixels must be below the edge line (with 10 pixel buffer)
                # Use logical AND to accumulate constraints from all edges
                is_below_all_edges &= (y_coords >= edge_y_grid + 10)
        
        # Convert boolean mask to uint8 and apply to interior_mask
        interior_mask[is_below_all_edges] = 255
        
        # Additional safety: zero out top 40% of image to focus on bottom 60%
        interior_mask[:int(h * 0.4), :] = 0
        
        mask_time = time.time() - mask_start_time
        
        if self.debug:
            interior_pixels = np.count_nonzero(interior_mask)
            print(f"  Whiteboard interior (below edges): {interior_pixels} pixels ({interior_pixels/(h*w)*100:.1f}% of image)")
            print(f"  Interior mask creation time: {mask_time:.4f}s")
        
        return interior_mask

    def save_debug_visualization(self, image: np.ndarray, marking_boxes: List[Tuple[int, int]], 
                               edge_lines: List[Tuple[float, float]], filename: str) -> None:
        """Save debug visualization with bounding boxes and detected edges"""
        debug_img = image.copy()
        h, w = image.shape[:2]
        
        # Draw detected edge lines in red and green (like original debug image)
        colors = [(0, 0, 255), (0, 255, 0)]  # Red, Green
        for i, (rho, theta) in enumerate(edge_lines):
            color = colors[i % len(colors)]
            a, b = np.cos(theta), np.sin(theta)
            
            # Draw line across the full image
            if abs(b) > 0.001:  # Not nearly vertical
                # Calculate line endpoints
                y1 = int(rho / b) if b != 0 else 0
                y2 = int((rho - w * a) / b) if b != 0 else h
                x1, x2 = 0, w
                
                # Clamp to image bounds
                if y1 < 0:
                    x1 = int(-rho / a) if a != 0 else 0
                    y1 = 0
                if y2 >= h:
                    x2 = int((rho - (h-1) * b) / a) if a != 0 else w
                    y2 = h - 1
                if y1 >= h:
                    x1 = int((rho - (h-1) * b) / a) if a != 0 else 0
                    y1 = h - 1
                if y2 < 0:
                    x2 = int(-rho / a) if a != 0 else w
                    y2 = 0
                    
                cv2.line(debug_img, (max(0, min(w-1, x1)), max(0, min(h-1, y1))), 
                        (max(0, min(w-1, x2)), max(0, min(h-1, y2))), color, 3)
            else:  # Nearly vertical line
                x_line = int(rho / a) if abs(a) > 0.001 else 0
                if 0 <= x_line < w:
                    cv2.line(debug_img, (x_line, 0), (x_line, h-1), color, 3)
        
        # Draw 20x20 marking boxes in green
        for x, y in marking_boxes:
            cv2.rectangle(debug_img, (x-10, y-10), (x+10, y+10), (0, 255, 0), 2)
            cv2.circle(debug_img, (x, y), 3, (0, 255, 255), -1)  # Center point in yellow
        
        # Add text overlay with detection stats
        cv2.putText(debug_img, f"Edges: {len(edge_lines)}, Markings: {len(marking_boxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_img, f"Edges: {len(edge_lines)}, Markings: {len(marking_boxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imwrite(filename, debug_img)
        if self.debug:
            print(f"  Debug visualization saved: {filename}")
    
    def detect_whiteboard_edges_rpi5(self, image: np.ndarray, target_width: int = 320) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]]:
        """RPi5-optimized version that processes at lower resolution for speed"""
        if image is None:
            return None
        
        original_h, original_w = image.shape[:2]
        
        # Downscale for processing (significant speed boost on RPi5)
        scale = target_width / original_w
        target_h = int(original_h * scale)
        
        if self.debug:
            print(f"  Downscaling: {original_w}x{original_h} -> {target_width}x{target_h} (scale={scale:.3f})")
        
        small_image = cv2.resize(image, (target_width, target_h))
        
        # Process on smaller image
        result = self.detect_whiteboard_edges(small_image)
        
        if result is None:
            return None
        
        edge_lines, marking_boxes = result
        
        # Scale results back to original image coordinates
        scaled_lines = []
        for rho, theta in edge_lines:
            # Scale rho back to original image size
            scaled_rho = rho / scale
            scaled_lines.append((scaled_rho, theta))  # theta remains the same
        
        scaled_boxes = []
        for x, y in marking_boxes:
            # Scale box centers back to original image coordinates  
            scaled_x = int(x / scale)
            scaled_y = int(y / scale)
            scaled_boxes.append((scaled_x, scaled_y))
        
        if self.debug:
            print(f"  Scaled results: {len(scaled_lines)} lines, {len(scaled_boxes)} marking boxes")
        
        return scaled_lines, scaled_boxes


def create_debug_visualization(image, edge_lines, marking_boxes, output_filename):
    """Create simple debug visualization"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Result with overlays
        result_img = image.copy()
        h, w = image.shape[:2]
        
        # Draw edge lines
        if edge_lines:
            colors = [(0, 0, 255), (0, 255, 0)]  # Red, Green
            for i, (rho, theta) in enumerate(edge_lines):
                color = colors[i % len(colors)]
                a, b = np.cos(theta), np.sin(theta)
                
                # Find line endpoints
                if abs(b) > 0.001:
                    y1 = int(rho / b) if rho / b >= 0 else 0
                    y2 = int((rho - w * a) / b) if (rho - w * a) / b <= h else h
                    cv2.line(result_img, (0, y1), (w, y2), color, 2)
                elif abs(a) > 0.001:
                    x1 = int(rho / a) if rho / a >= 0 else 0  
                    x2 = int((rho - h * b) / a) if (rho - h * b) / a <= w else w
                    cv2.line(result_img, (x1, 0), (x2, h), color, 2)
        
        # Draw 20x20 marking boxes
        if marking_boxes:
            for center_x, center_y in marking_boxes:
                # Draw 20x20 box centered at marking
                top_left = (center_x - 10, center_y - 10)
                bottom_right = (center_x + 10, center_y + 10)
                cv2.rectangle(result_img, top_left, bottom_right, (255, 255, 0), 2)
        
        ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Results: {len(edge_lines)} lines, {len(marking_boxes)} markings')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Debug visualization saved: {output_filename}")
        
    except Exception as e:
        print(f"  Visualization error: {e}")


def main():
    """Test function for development"""
    tracker = WhiteboardTracker5(debug=True)
    
    # Find test images
    image_files = []
    for ext in ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']:
        image_files.extend(glob.glob(f'*.{ext}'))
    
    if not image_files:
        print("No test images found")
        return
    
    print(f"Testing WhiteboardTracker5 on {len(image_files)} images...")
    
    successful_detections = 0
    
    for image_path in sorted(image_files):
        print(f"\nProcessing: {image_path}")
        print("-" * 50)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Could not load {image_path}")
            continue
        
        # Resize to target RPi5 resolution for realistic testing
        target_height = 400
        scale = target_height / image.shape[0]
        new_width = int(image.shape[1] * scale)
        image = cv2.resize(image, (new_width, target_height))
        
        print(f"  Resized to: {new_width}x{target_height} (RPi5 target resolution)")
        
        result = tracker.detect_whiteboard_edges(image)
        
        if result:
            edge_lines, marking_boxes = result
            successful_detections += 1
            
            # Create debug visualization
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_file = f"debug_{base_name}_tracker5.png"
            create_debug_visualization(image, edge_lines, marking_boxes, output_file)
        else:
            print("  No whiteboard edges detected")
    
    print(f"\nResults: {successful_detections}/{len(image_files)} successful detections")
    if tracker.processing_times:
        avg_time = tracker.get_average_processing_time()
        fps_estimate = 1.0 / avg_time if avg_time > 0 else 0
        print(f"Average processing time: {avg_time:.3f}s ({fps_estimate:.1f} FPS)")


if __name__ == "__main__":
    main()