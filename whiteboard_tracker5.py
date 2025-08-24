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
        
        # Low contrast indicators:
        # 1. Small difference between whiteboard and background
        # 2. High mean intensities (bright scene) 
        # 3. Both regions are relatively bright
        is_low_intensity_diff = intensity_difference < 40  # More sensitive
        is_bright_scene = (bottom_mean + top_mean) / 2 > 160  # Lower threshold
        is_both_bright = bottom_mean > 150 and top_mean > 150  # Both regions bright
        
        # Consider low contrast if any of these conditions are met
        is_low_contrast = is_low_intensity_diff or is_bright_scene or is_both_bright
        
        if self.debug:
            print(f"  Scene analysis: intensity_diff={intensity_difference:.1f}, "
                  f"bottom_mean={bottom_mean:.1f}, top_mean={top_mean:.1f}, "
                  f"bright_scene={is_bright_scene}, both_bright={is_both_bright}, "
                  f"low_contrast={is_low_contrast}")
        
        return is_low_contrast
        
    def find_whiteboard_surface_improved(self, image: np.ndarray) -> np.ndarray:
        """Improved whiteboard surface detection focusing on actual whiteboard area"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Focus on bottom 70% where whiteboard surface appears
        bottom_start = int(h * 0.3)
        bottom_region = gray[bottom_start:, :]
        
        if self.debug:
            print(f"  Processing region: {w}x{h-bottom_start} (bottom 70%)")
        
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
        """Detect edges specifically at whiteboard boundaries - optimized for RPi5"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Create boundary detection region - balanced approach
        eroded = cv2.erode(surface_mask, self.kernel_5x5, iterations=1)
        dilated = cv2.dilate(surface_mask, self.kernel_5x5, iterations=1)
        boundary_region = cv2.subtract(dilated, eroded)
        
        # Use simple Gaussian blur instead of bilateral filter (much faster on RPi5)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # More conservative Canny parameters for cleaner edges
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Only keep edges that are at whiteboard boundaries
        boundary_edges = cv2.bitwise_and(edges, boundary_region)
        
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
            
            # More lenient position check (bottom 80% instead of 70%)
            is_in_likely_region = y > h * 0.2
            
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
        boundary_edges[:int(h * 0.2), :] = 0
        
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
    
    def find_whiteboard_lines(self, edges: np.ndarray) -> List[Tuple[float, float]]:
        """Simplified line detection with slope-based filtering"""
        h, w = edges.shape
        
        if np.count_nonzero(edges) < 20:
            return []
        
        # Adaptive Hough threshold based on available edge pixels
        edge_pixel_count = np.count_nonzero(edges)
        threshold = max(15, min(30, edge_pixel_count // 3))
        
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
                # Quick line quality check
                support_score = self.score_line_fast(rho, theta, edges, h, w)
                
                # Lower support threshold for low contrast scenarios
                min_support = 0.05 if self.is_low_contrast_scene else 0.1
                if support_score > min_support:
                    valid_lines.append((rho, theta, support_score))
                    if self.debug:
                        print(f"    Line: angle={angle_deg:.1f}°, rho={rho:.1f}, support={support_score:.3f}")
        
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
        """Detect dry-erase markings and return 20x20 pixel box centers"""
        h, w = image.shape[:2]
        
        # Create search region (whiteboard surface below edge lines)
        search_mask = surface_mask.copy()
        
        if edge_lines:
            # Restrict search to area below detected edges
            below_edges_mask = np.ones((h, w), dtype=np.uint8) * 255
            
            for rho, theta in edge_lines:
                a, b = np.cos(theta), np.sin(theta)
                line_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Mark area below this line
                for x in range(w):
                    if abs(b) > 0.001:
                        y_line = (rho - x * a) / b
                        if 0 <= y_line <= h:
                            line_mask[int(y_line):, x] = 255
                
                below_edges_mask = cv2.bitwise_and(below_edges_mask, line_mask)
            
            search_mask = cv2.bitwise_and(search_mask, below_edges_mask)
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_region = cv2.bitwise_and(hsv, hsv, mask=search_mask)
        
        # Expanded marker color ranges with lower saturation thresholds
        color_ranges = [
            # Purple/violet range 1 (lighter purple)
            [(110, 20, 30), (140, 255, 255)],
            # Purple/violet range 2 (darker purple)  
            [(140, 20, 30), (170, 255, 255)],
            # Blue range (expanded)
            [(90, 30, 30), (130, 255, 255)],
            # Red range 1 (lower hue)
            [(0, 30, 30), (15, 255, 255)],
            # Red range 2 (higher hue - wraparound)
            [(165, 30, 30), (180, 255, 255)],
            # Green range (for completeness)
            [(40, 30, 30), (80, 255, 255)]
        ]
        
        all_markings = np.zeros((h, w), dtype=np.uint8)
        
        # Detect each color
        for lower, upper in color_ranges:
            color_mask = cv2.inRange(hsv_region, np.array(lower), np.array(upper))
            
            # Minimal cleanup
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, self.kernel_3x3)
            
            # Filter small noise with lower threshold for better detection
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 8:  # Lower threshold for marking pixels
                    component_mask = (labels == i).astype(np.uint8) * 255
                    all_markings = cv2.bitwise_or(all_markings, component_mask)
                    
                    if self.debug:
                        color_idx = color_ranges.index([lower, upper])
                        color_names = ["light_purple", "dark_purple", "blue", "red1", "red2", "green"]
                        print(f"    Found {color_names[color_idx]} marking: {stats[i, cv2.CC_STAT_AREA]} pixels")
        
        # Extract marking centers and create 20x20 boxes
        marking_boxes = self.extract_marking_boxes_20x20(all_markings)
        
        if self.debug:
            total_pixels = np.count_nonzero(all_markings)
            print(f"  Marking pixels: {total_pixels}, 20x20 boxes: {len(marking_boxes)}")
        
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
            
            # Step 2: Adaptive edge detection based on scene analysis
            if self.is_low_contrast_scene:
                if self.debug:
                    print("  Using gradient-based edge detection for low contrast scene")
                edges = self.detect_edges_gradient_based(image, surface_mask)
            else:
                if self.debug:
                    print("  Using standard edge detection for high contrast scene")
                edges = self.detect_whiteboard_boundary_edges(image, surface_mask)
            
            # Step 3: Simplified line detection with validation
            edge_lines = self.find_whiteboard_lines(edges)
            
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