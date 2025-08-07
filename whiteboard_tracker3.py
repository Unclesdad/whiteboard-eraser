import cv2
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Try to import Numba for JIT compilation
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - install with: pip install numba")
    # Fallback decorator
    def njit(func):
        return func
    def jit(func):
        return func

class WhiteboardEdgeDetector:
    def __init__(self, debug: bool = True, save_visualizations: bool = True):
        self.debug = debug
        self.save_visualizations = save_visualizations  # Save images instead of displaying
        # Pre-calculate reusable kernels
        self.kernel_7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_3x3 = np.ones((3, 3), np.uint8)
        self.kernel_25x25 = np.ones((25, 25), np.uint8)
        
    def find_whiteboard_surface_region(self, image: np.ndarray) -> np.ndarray:
        """Find the whiteboard surface using a focused approach - ORIGINAL LOGIC"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        h_half = h // 2
        
        print(f"  Image size: {w}x{h}")
        
        # Focus on lower half where whiteboard surface should be
        lower_half = gray[h_half:, :]
        lower_mean = lower_half.mean()
        print(f"  Lower half intensity: min={lower_half.min()}, max={lower_half.max()}, mean={lower_mean:.1f}")
        
        # Pre-calculate percentiles once for efficiency
        percentiles = np.percentile(lower_half, [60, 70, 75])
        p60, p70, p75 = percentiles
        
        # Create mask for whiteboard surface using multiple strategies
        masks = []
        
        # Strategy 1: Reasonably bright regions
        bright_thresh = max(150, p70)
        bright_mask = ((lower_half > bright_thresh) * 255).astype(np.uint8)
        masks.append(('bright', bright_mask))
        
        # Strategy 2: Top 40% brightest pixels
        top_mask = ((lower_half > p60) * 255).astype(np.uint8)
        masks.append(('top60', top_mask))
        
        # Strategy 3: Otsu thresholding
        _, otsu_mask = cv2.threshold(lower_half, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(('otsu', otsu_mask))
        
        # Choose best mask based on having a large connected component
        best_mask = None
        best_score = 0
        area_threshold = lower_half.size * 0.05
        
        for name, mask in masks:
            # Clean up mask
            clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_7x7)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.kernel_7x7)
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
            
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_area = areas.max()
                coverage = largest_area / lower_half.size
                
                print(f"    {name}: largest_area={largest_area}, coverage={coverage:.3f}")
                
                if coverage > 0.05 and largest_area > best_score:
                    best_score = largest_area
                    best_mask = clean_mask
                    print(f"    -> New best mask: {name}")
        
        if best_mask is None:
            print("  No suitable surface mask found, using fallback")
            best_mask = ((lower_half > p75) * 255).astype(np.uint8)
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, self.kernel_5x5)
        
        # Create full image mask
        full_mask = np.zeros_like(gray, dtype=np.uint8)
        full_mask[h_half:, :] = best_mask
        
        # Keep only largest component in full mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
        if num_labels > 1:
            largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            full_mask = ((labels == largest_idx) * 255).astype(np.uint8)
        
        final_area = np.count_nonzero(full_mask)
        print(f"  Final surface area: {final_area} pixels ({final_area/(h*w):.3f} of image)")
        
        return full_mask
    
    def create_edge_detection_regions(self, surface_mask: np.ndarray) -> np.ndarray:
        """Create focused boundary region for edge detection - ORIGINAL LOGIC"""
        eroded = cv2.erode(surface_mask, self.kernel_25x25, iterations=1)
        dilated = cv2.dilate(surface_mask, self.kernel_25x25, iterations=3)
        boundary_region = cv2.subtract(dilated, eroded)
        
        print(f"  Boundary region: {np.count_nonzero(boundary_region)} pixels")
        return boundary_region
    
    def filter_structural_edges(self, edges: np.ndarray, boundary_region: np.ndarray) -> np.ndarray:
        """Filter edges to focus on structural boundaries - ORIGINAL LOGIC PRESERVED"""
        h, w = edges.shape
        
        # Apply boundary mask
        boundary_edges = cv2.bitwise_and(edges, boundary_region)
        
        print(f"    Edges before boundary filtering: {np.count_nonzero(edges)}")
        print(f"    Edges after boundary filtering: {np.count_nonzero(boundary_edges)}")
        
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(boundary_edges, connectivity=8)
        
        if num_labels <= 1:
            return boundary_edges
        
        filtered_edges = np.zeros_like(boundary_edges)
        structural_components = 0
        
        # Pre-calculate boundary thresholds for efficiency
        w_025 = w * 0.25
        w_075 = w * 0.75
        h_025 = h * 0.25
        h_075 = h * 0.75
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 8:
                continue
                
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Keep components that could be structural edges
            min_dim = max(min(width, height), 1)
            aspect_ratio = max(width, height) / min_dim
            
            if area >= 8 and aspect_ratio > 1.5:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                
                near_boundary = (x < w_025 or x + width > w_075 or 
                               y < h_025 or y + height > h_075)
                
                if near_boundary or aspect_ratio > 3 or area > 30:
                    filtered_edges[labels == i] = 255
                    structural_components += 1
                    print(f"      Kept component: area={area}, aspect={aspect_ratio:.1f}, near_boundary={near_boundary}")
                else:
                    print(f"      Rejected component: area={area}, aspect={aspect_ratio:.1f}, not near boundary")
            else:
                print(f"      Rejected component: area={area}, aspect={aspect_ratio:.1f} (too small/round)")
        
        print(f"    Structural components kept: {structural_components}/{num_labels-1}")
        print(f"    Edges after content filtering: {np.count_nonzero(filtered_edges)}")
        
        # Morphological cleanup
        filtered_edges = cv2.morphologyEx(filtered_edges, cv2.MORPH_CLOSE, self.kernel_3x3)
        
        # Final cleanup: remove remaining curved or irregular patterns
        contours, _ = cv2.findContours(filtered_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_edges = np.zeros_like(filtered_edges)
        linear_contours = 0
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) <= 6:
                    cv2.drawContours(final_edges, [contour], -1, 255, -1)
                    linear_contours += 1
                    print(f"      Kept linear contour with {len(approx)} vertices")
                else:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        cv2.drawContours(final_edges, [contour], -1, 255, -1)
                        linear_contours += 1
                        print(f"      Kept large contour with {len(approx)} vertices, area={area}")
                    else:
                        print(f"      Rejected curved contour with {len(approx)} vertices, area={area}")
        
        print(f"    Linear contours kept: {linear_contours}/{len(contours)}")
        print(f"    Final filtered edges: {np.count_nonzero(final_edges)}")
        
        return final_edges
    
    def detect_clean_edges(self, image: np.ndarray, boundary_region: np.ndarray) -> List[np.ndarray]:
        """Detect edges with enhanced filtering - ORIGINAL METHODS"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pre-compute blurred versions once
        blur5 = cv2.GaussianBlur(gray, (5, 5), 1)
        bilateral = cv2.bilateralFilter(gray, 11, 100, 100)
        
        edge_results = []
        
        # Method 1: Standard Canny with moderate blur
        edges1 = cv2.Canny(blur5, 50, 150, apertureSize=3)
        filtered1 = self.filter_structural_edges(edges1, boundary_region)
        edge_results.append(('Canny_50_150_filtered', filtered1))
        
        # Method 2: Lower threshold Canny
        edges2 = cv2.Canny(blur5, 30, 100, apertureSize=3)
        filtered2 = self.filter_structural_edges(edges2, boundary_region)
        edge_results.append(('Canny_30_100_filtered', filtered2))
        
        # Method 3: Bilateral filter + Canny
        edges3 = cv2.Canny(bilateral, 40, 120, apertureSize=3)
        filtered3 = self.filter_structural_edges(edges3, boundary_region)
        edge_results.append(('Bilateral_Canny_filtered', filtered3))
        
        # Method 4: Sobel with high threshold
        sobel_x = cv2.Sobel(blur5, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur5, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.hypot(sobel_x, sobel_y)  # Faster than sqrt
        sobel_thresh = np.percentile(sobel_mag, 94)
        sobel_edges = ((sobel_mag > sobel_thresh) * 255).astype(np.uint8)
        filtered4 = self.filter_structural_edges(sobel_edges, boundary_region)
        edge_results.append(('Sobel_filtered', filtered4))
        
        # Report edge pixel counts
        for name, edges in edge_results:
            print(f"  {name}: {np.count_nonzero(edges)} edge pixels")
        
        return edge_results
    
    @staticmethod
    @njit
    def score_line_numba(rho, theta, edges, h, w):
        """Numba-optimized line scoring if available"""
        a = np.cos(theta)
        b = np.sin(theta)
        edge_hits = 0
        total_samples = 0
        
        if abs(b) > abs(a):  # More horizontal line
            for x in range(w):
                y = (rho - x * a) / b
                if 0 <= y < h:
                    y_int = int(y + 0.5)  # Round
                    if 0 <= y_int < h:
                        total_samples += 1
                        # Check 3x1 neighborhood
                        hit_found = False
                        for dy in range(max(0, y_int-1), min(h, y_int+2)):
                            if edges[dy, x] > 0:
                                hit_found = True
                                break
                        if hit_found:
                            edge_hits += 1
        else:  # More vertical line
            for y in range(h):
                x = (rho - y * b) / a
                if 0 <= x < w:
                    x_int = int(x + 0.5)  # Round
                    if 0 <= x_int < w:
                        total_samples += 1
                        # Check 1x3 neighborhood
                        hit_found = False
                        for dx in range(max(0, x_int-1), min(w, x_int+2)):
                            if edges[y, dx] > 0:
                                hit_found = True
                                break
                        if hit_found:
                            edge_hits += 1
        
        if total_samples == 0:
            return 0.0, 0
        
        return edge_hits / total_samples, edge_hits
    
    def score_line_against_edges(self, rho: float, theta: float, edges: np.ndarray) -> Tuple[float, int]:
        """Score how well a line matches the actual edge pixels"""
        h, w = edges.shape
        
        if NUMBA_AVAILABLE:
            return self.score_line_numba(rho, theta, edges, h, w)
        
        # Fallback to vectorized numpy version
        a = np.cos(theta)
        b = np.sin(theta)
        edge_hits = 0
        total_samples = 0
        
        if abs(b) > abs(a):  # More horizontal
            x_range = np.arange(0, w, 1)
            y_values = (rho - x_range * a) / b
            valid_mask = (y_values >= 0) & (y_values < h)
            valid_x = x_range[valid_mask]
            valid_y = np.round(y_values[valid_mask]).astype(int)
            
            total_samples = len(valid_x)
            for x, y in zip(valid_x, valid_y):
                # Check 3x1 neighborhood
                for dy in range(max(0, y-1), min(h, y+2)):
                    if edges[dy, x] > 0:
                        edge_hits += 1
                        break
        else:  # More vertical
            y_range = np.arange(0, h, 1)
            x_values = (rho - y_range * b) / a
            valid_mask = (x_values >= 0) & (x_values < w)
            valid_y = y_range[valid_mask]
            valid_x = np.round(x_values[valid_mask]).astype(int)
            
            total_samples = len(valid_y)
            for x, y in zip(valid_x, valid_y):
                # Check 1x3 neighborhood
                for dx in range(max(0, x-1), min(w, x+2)):
                    if edges[y, dx] > 0:
                        edge_hits += 1
                        break
        
        if total_samples == 0:
            return 0.0, 0
        
        return edge_hits / total_samples, edge_hits
    
    def deduplicate_lines(self, lines: List[Tuple[float, float, float, int, str]]) -> List[Tuple[float, float, float, int, str]]:
        """Remove duplicate lines - ORIGINAL LOGIC"""
        if len(lines) <= 1:
            return lines
        
        unique_lines = []
        
        for line in lines:
            rho, theta, support, hits, edge_type = line
            angle_deg = np.degrees(theta)
            
            is_duplicate = False
            for i, existing in enumerate(unique_lines):
                existing_rho, existing_theta = existing[:2]
                existing_angle = np.degrees(existing_theta)
                
                rho_diff = abs(rho - existing_rho)
                angle_diff = min(abs(angle_deg - existing_angle),
                               abs(angle_deg - existing_angle + 180),
                               abs(angle_deg - existing_angle - 180))
                
                if rho_diff < 10 and angle_diff < 2:
                    print(f"      Removing duplicate: rho_diff={rho_diff:.1f}, angle_diff={angle_diff:.1f}°")
                    if support > existing[2]:
                        unique_lines[i] = line
                        print(f"        Replaced with better support ({support:.3f} > {existing[2]:.3f})")
                    else:
                        print(f"        Kept existing with better support ({existing[2]:.3f} > {support:.3f})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        print(f"    Deduplication: {len(lines)} → {len(unique_lines)} lines")
        return unique_lines
    
    def validate_whiteboard_geometry(self, lines: List[Tuple[float, float, float, int, str]], 
                                   image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, int, str]]:
        """Sequential selection - ORIGINAL LOGIC PRESERVED"""
        if len(lines) <= 1:
            return lines
        
        print(f"    Sequential selection for {len(lines)} lines:")
        
        if len(lines) > 10:
            print(f"    WARNING: Too many lines ({len(lines)}), limiting to top 10")
            lines = lines[:10]
        
        def calculate_slope(theta):
            angle_deg = np.degrees(theta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            print(f"        Debug slope calc: theta={angle_deg:.1f}°, sin={sin_theta:.3f}, cos={cos_theta:.3f}")
            
            if abs(sin_theta) < 0.01:
                return 0.0
            elif abs(cos_theta) < 0.01:
                return float('inf')
            else:
                slope = -cos_theta / sin_theta
                print(f"        Calculated slope: {slope:.3f}")
                return slope
        
        def get_slope_sign(slope, theta):
            angle_deg = np.degrees(theta)
            
            if 85 <= angle_deg <= 95 or 265 <= angle_deg <= 275:
                return 'vertical'
            elif abs(angle_deg) <= 5 or abs(angle_deg - 180) <= 5:
                return 'horizontal'
            elif 0 < angle_deg < 90 or 180 < angle_deg < 270:
                return 'positive'
            else:
                return 'negative'
        
        # Sort by quality
        sorted_lines = sorted(lines, key=lambda x: (x[2], x[3]), reverse=True)
        
        selected_lines = []
        
        # Select best line
        best_line = sorted_lines[0]
        best_rho, best_theta, best_support, best_hits, best_type = best_line
        best_slope = calculate_slope(best_theta)
        best_slope_sign = get_slope_sign(best_slope, best_theta)
        best_angle = np.degrees(best_theta)
        
        selected_lines.append(best_line)
        print(f"    SELECTED Line 1 (BEST): rho={best_rho:.1f}, angle={best_angle:.1f}°, slope={best_slope:.3f} ({best_slope_sign}), support={best_support:.3f}")
        
        # Search for best opposite slope
        print(f"    Searching for line with opposite slope sign to '{best_slope_sign}'...")
        
        best_opposite = None
        best_opposite_quality = -1
        
        for candidate in sorted_lines[1:]:
            candidate_rho, candidate_theta, candidate_support, candidate_hits, candidate_type = candidate
            candidate_slope = calculate_slope(candidate_theta)
            candidate_slope_sign = get_slope_sign(candidate_slope, candidate_theta)
            candidate_angle = np.degrees(candidate_theta)
            
            rho_diff = abs(best_rho - candidate_rho)
            angle_diff = min(abs(best_angle - candidate_angle),
                           abs(best_angle - candidate_angle + 180),
                           abs(best_angle - candidate_angle - 180))
            
            is_too_similar = (rho_diff < 50 and angle_diff < 10)
            
            print(f"      Candidate: rho={candidate_rho:.1f}, angle={candidate_angle:.1f}°, slope={candidate_slope:.3f} ({candidate_slope_sign})")
            print(f"        rho_diff={rho_diff:.1f}px, angle_diff={angle_diff:.1f}°, too_similar={is_too_similar}")
            
            has_opposite_sign = False
            
            if best_slope_sign == 'positive' and candidate_slope_sign == 'negative':
                has_opposite_sign = True
            elif best_slope_sign == 'negative' and candidate_slope_sign == 'positive':
                has_opposite_sign = True
            elif best_slope_sign == 'horizontal' and candidate_slope_sign in ['positive', 'negative']:
                has_opposite_sign = True
            elif best_slope_sign in ['positive', 'negative'] and candidate_slope_sign == 'horizontal':
                has_opposite_sign = True
            elif best_slope_sign == 'vertical' and candidate_slope_sign != 'vertical':
                has_opposite_sign = True
            elif best_slope_sign != 'vertical' and candidate_slope_sign == 'vertical':
                has_opposite_sign = True
            
            print(f"        opposite_sign={has_opposite_sign} ({best_slope_sign} vs {candidate_slope_sign})")
            
            if is_too_similar:
                print(f"        REJECTED: Too similar")
                continue
                
            if not has_opposite_sign:
                print(f"        REJECTED: Same slope type")
                continue
            
            candidate_quality = candidate_support * candidate_hits
            
            if candidate_quality > best_opposite_quality:
                best_opposite = candidate
                best_opposite_quality = candidate_quality
                print(f"        NEW BEST OPPOSITE: quality={candidate_quality:.1f}")
            else:
                print(f"        REJECTED: Lower quality than current best opposite")
        
        if best_opposite:
            selected_lines.append(best_opposite)
            opp_rho, opp_theta, opp_support, opp_hits, opp_type = best_opposite
            opp_slope = calculate_slope(opp_theta)
            opp_slope_sign = get_slope_sign(opp_slope, opp_theta)
            opp_angle = np.degrees(opp_theta)
            print(f"    SELECTED Line 2 (OPPOSITE): rho={opp_rho:.1f}, angle={opp_angle:.1f}°, slope={opp_slope:.3f} ({opp_slope_sign}), support={opp_support:.3f}")
        else:
            print(f"    NO OPPOSITE LINE FOUND")
        
        print(f"    Final result: {len(selected_lines)} line(s)")
        return selected_lines
    
    def find_best_lines(self, edge_images: List[Tuple[str, np.ndarray]]) -> List[Tuple[float, float]]:
        """Find the best whiteboard edge lines - ORIGINAL LOGIC"""
        h, w = edge_images[0][1].shape
        all_scored_lines = []
        
        # Try each edge detection method
        for method_name, edges in edge_images:
            print(f"\n  Analyzing {method_name}:")
            
            edge_pixel_count = np.count_nonzero(edges)
            if edge_pixel_count < 20:
                print(f"    Skipping - too few edge pixels ({edge_pixel_count})")
                continue
            
            print(f"    Edge pixels available: {edge_pixel_count}")
            
            # Try different Hough parameter sets
            hough_configs = [
                {'threshold': 100, 'rho': 1, 'theta': np.pi/180},
                {'threshold': 80, 'rho': 1, 'theta': np.pi/180},
                {'threshold': 60, 'rho': 1, 'theta': np.pi/180},
                {'threshold': 40, 'rho': 1, 'theta': np.pi/180},
            ]
            
            method_lines = []
            
            for config in hough_configs:
                lines = cv2.HoughLines(edges, **config)
                
                if lines is not None and len(lines) > 0:
                    print(f"    HoughLines (thresh={config['threshold']}): {len(lines)} raw lines")
                    
                    # Limit to top 20 lines
                    lines = lines[:20]
                    print(f"    Limiting to first {len(lines)} lines")
                    
                    scored_candidates = []
                    
                    for line in lines:
                        rho, theta = line[0]
                        angle_deg = np.degrees(theta)
                        
                        # Angle classification
                        is_horizontal = (abs(angle_deg) < 15) or (abs(angle_deg - 180) < 15)
                        is_vertical = (75 < angle_deg < 105)
                        is_diagonal = (15 <= abs(angle_deg) <= 75) or (105 <= abs(angle_deg) <= 165)
                        
                        if is_horizontal or is_vertical or is_diagonal:
                            support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                            
                            if support_ratio > 0.05 and edge_hits > 10:
                                if is_horizontal:
                                    edge_type = 'H'
                                elif is_vertical:
                                    edge_type = 'V'
                                else:
                                    edge_type = 'D'
                                
                                scored_candidates.append((rho, theta, support_ratio, edge_hits, edge_type))
                                print(f"      {edge_type} candidate: rho={rho:.1f}, angle={angle_deg:.1f}°, support={support_ratio:.3f}, hits={edge_hits}")
                    
                    print(f"    After scoring: {len(scored_candidates)} candidates")
                    
                    if scored_candidates:
                        method_lines.extend(scored_candidates)
                        print(f"    Method contributed: {len(scored_candidates)} lines")
                        
                        if len(method_lines) >= 2:
                            print(f"    Stopping - already have {len(method_lines)} lines")
                            break
            
            all_scored_lines.extend(method_lines)
            
            if len(all_scored_lines) >= 2:
                print(f"  Stopping methods - already have {len(all_scored_lines)} lines")
                break
        
        # Early deduplication
        if all_scored_lines:
            print(f"\n  Applying early deduplication...")
            all_scored_lines = self.deduplicate_lines(all_scored_lines)
        
        # Apply geometry validation
        if len(all_scored_lines) > 1:
            print(f"  Applying sequential selection...")
            all_scored_lines = self.validate_whiteboard_geometry(all_scored_lines, (h, w))
        
        # Final safety check
        if len(all_scored_lines) > 2:
            print(f"  ERROR: Have {len(all_scored_lines)} lines, limiting to 2")
            all_scored_lines = all_scored_lines[:2]
        
        if not all_scored_lines:
            print("  No lines found")
            return []
        
        # Sort by quality
        all_scored_lines.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        print(f"\n  Final validated lines:")
        for i, (rho, theta, support, hits, edge_type) in enumerate(all_scored_lines):
            angle = np.degrees(theta)
            print(f"    {i+1}. {edge_type}: rho={rho:.1f}, angle={angle:.1f}°, support={support:.3f}, hits={hits}")
        
        # Return only the line coordinates
        return [(rho, theta) for rho, theta, support, hits, edge_type in all_scored_lines]
    
    def detect_whiteboard_edges(self, image_path: str) -> Optional[List[Tuple[float, float]]]:
        """Main detection function - ORIGINAL FLOW"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        print(f"\nProcessing: {os.path.basename(image_path)}")
        print("=" * 60)
        
        try:
            # Step 1: Find whiteboard surface
            print("Step 1: Finding whiteboard surface...")
            surface_mask = self.find_whiteboard_surface_region(image)
            
            surface_area = np.count_nonzero(surface_mask)
            if surface_area < 500:
                print(f"  FAIL: Surface area too small ({surface_area} pixels)")
                return None
            
            # Step 2: Create boundary region
            print("\nStep 2: Creating boundary region...")
            boundary_region = self.create_edge_detection_regions(surface_mask)
            
            # Step 3: Detect edges
            print("\nStep 3: Detecting edges...")
            edge_images = self.detect_clean_edges(image, boundary_region)
            
            # Step 4: Find lines
            print("\nStep 4: Finding lines...")
            detected_lines = self.find_best_lines(edge_images)
            
            if not detected_lines:
                print("  No lines detected")
                if self.debug:
                    self.debug_visualize(image, surface_mask, edge_images, [], os.path.basename(image_path))
                return None
            
            print(f"\nSUCCESS: Found {len(detected_lines)} edge line(s)")
            for i, (rho, theta) in enumerate(detected_lines):
                angle = np.degrees(theta)
                edge_type = "horizontal" if abs(angle) < 45 or abs(angle) > 135 else "vertical"
                print(f"  Edge {i+1}: {edge_type}, rho={rho:.1f}, angle={angle:.1f}°")
            
            if self.debug:
                best_edge_image = max(edge_images, key=lambda x: np.count_nonzero(x[1]))
                self.debug_visualize(image, surface_mask, [best_edge_image], detected_lines, 
                                   os.path.basename(image_path))
            
            return detected_lines
            
        except Exception as e:
            print(f"  ERROR during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def debug_visualize(self, image: np.ndarray, surface_mask: np.ndarray, 
                       edge_images: List[Tuple[str, np.ndarray]], lines: List[Tuple[float, float]], 
                       filename: str):
        """Debug visualization - ORIGINAL LOGIC"""
        # Skip visualization if not in main thread (causes issues on macOS)
        import threading
        if threading.current_thread() != threading.main_thread():
            print("  Skipping visualization (not in main thread)")
            return
            
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Original image
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f'Original: {filename}')
            axes[0, 0].axis('off')
            
            # Surface mask
            axes[0, 1].imshow(surface_mask, cmap='gray')
            axes[0, 1].set_title('Whiteboard Surface')
            axes[0, 1].axis('off')
            
            # Best edge image
            if edge_images:
                best_edges = edge_images[0][1]
                axes[1, 0].imshow(best_edges, cmap='gray')
                axes[1, 0].set_title(f'Edges: {edge_images[0][0]}')
            else:
                axes[1, 0].imshow(np.zeros_like(surface_mask), cmap='gray')
                axes[1, 0].set_title('No Edges')
            axes[1, 0].axis('off')
            
            # Result with lines
            result_img = image.copy()
            colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            
            print(f"  DEBUG: Visualizing {len(lines)} lines")
            
            h_img, w_img = image.shape[:2]
            
            for i, (rho, theta) in enumerate(lines):
                color = colors_bgr[i % len(colors_bgr)]
                angle_deg = np.degrees(theta)
                
                print(f"    Drawing line {i+1}: rho={rho:.1f}, theta={angle_deg:.1f}°, color={color}")
                
                # Find intersections with image boundaries
                intersections = []
                
                a = np.cos(theta)
                b = np.sin(theta)
                
                print(f"      Line equation: a={a:.3f}, b={b:.3f}")
                
                # Check intersection with all four image edges
                # Left edge (x = 0)
                if abs(a) > 0.001:
                    y_left = rho / b if abs(b) > 0.001 else float('inf')
                    if 0 <= y_left <= h_img:
                        intersections.append((0, int(y_left)))
                        print(f"      Left edge intersection: (0, {int(y_left)})")
                
                # Right edge (x = w_img)
                if abs(a) > 0.001:
                    y_right = (rho - w_img * a) / b if abs(b) > 0.001 else float('inf')
                    if 0 <= y_right <= h_img:
                        intersections.append((w_img, int(y_right)))
                        print(f"      Right edge intersection: ({w_img}, {int(y_right)})")
                
                # Top edge (y = 0)
                if abs(b) > 0.001:
                    x_top = rho / a if abs(a) > 0.001 else float('inf')
                    if 0 <= x_top <= w_img:
                        intersections.append((int(x_top), 0))
                        print(f"      Top edge intersection: ({int(x_top)}, 0)")
                
                # Bottom edge (y = h_img)
                if abs(b) > 0.001:
                    x_bottom = (rho - h_img * b) / a if abs(a) > 0.001 else float('inf')
                    if 0 <= x_bottom <= w_img:
                        intersections.append((int(x_bottom), h_img))
                        print(f"      Bottom edge intersection: ({int(x_bottom)}, {h_img})")
                
                # Remove duplicates
                unique_intersections = list(set(intersections))
                print(f"      Total intersections found: {len(unique_intersections)}")
                
                if len(unique_intersections) >= 2:
                    p1, p2 = unique_intersections[0], unique_intersections[1]
                    print(f"      Drawing line from {p1} to {p2}")
                    cv2.line(result_img, p1, p2, color, 4)
                    
                    # Add line label
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    
                    edge_type = "H" if abs(angle_deg) < 45 or abs(angle_deg) > 135 else "V"
                    
                    if edge_images:
                        support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edge_images[0][1])
                        label = f'{edge_type}{i+1}: {support_ratio:.2f}'
                    else:
                        label = f'{edge_type}{i+1}'
                    
                    cv2.putText(result_img, label, (mid_x, mid_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    print(f"      SUCCESS: Line {i+1} drawn with label '{label}'")
                else:
                    print(f"      ERROR: Not enough intersections for line {i+1}")
            
            axes[1, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title(f'Result ({len(lines)} lines)')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save to file instead of showing (safer for threading)
            output_filename = f'debug_{filename.replace(".jpg", "").replace(".jpeg", "")}_result.png'
            plt.savefig(output_filename, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"  Debug visualization saved to: {output_filename}")
            
        except Exception as e:
            print(f"Visualization error: {e}")

def process_images_parallel(detector, image_files):
    """Process multiple images using thread pool for edge detection"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
        # Submit all tasks
        futures = {executor.submit(detector.detect_whiteboard_edges, img): img 
                  for img in image_files}
        
        # Collect results as they complete
        for future in futures:
            image_path = futures[future]
            try:
                result = future.result()
                results[image_path] = result
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path] = None
            print("=" * 60)
    
    return results

def main():
    # Set OpenCV optimization flags
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)  # Optimize for multi-core
    
    # Use save_visualizations=True to save debug images to files (thread-safe)
    # Set debug=False to disable visualization entirely
    detector = WhiteboardEdgeDetector(debug=False, save_visualizations=True)
    
    # Find images
    image_files = []
    for ext in ['jpg', 'jpeg', 'JPG', 'JPEG']:
        image_files.extend(glob.glob(f'*.{ext}'))
    
    if not image_files:
        print("No JPG images found in current directory")
        return {}
    
    print(f"Found {len(image_files)} image(s)")
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Numba JIT available: {NUMBA_AVAILABLE}")
    
    image_files = sorted(image_files)
    
    # For macOS/threading safety, process sequentially if debug is enabled
    if detector.debug and mp.cpu_count() >= 4 and len(image_files) > 3:
        print("Note: Using sequential processing with debug mode for thread safety")
        print("Set debug=False in WhiteboardEdgeDetector() for parallel processing")
    
    # Process based on number of images and cores
    if not detector.debug and len(image_files) > 3 and mp.cpu_count() >= 4:
        print("Using parallel processing for edge detection steps...")
        results = process_images_parallel(detector, image_files)
    else:
        print("Using sequential processing...")
        results = {}
        for image_path in image_files:
            try:
                edges = detector.detect_whiteboard_edges(image_path)
                results[image_path] = edges
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path] = None
            print("=" * 60)
    
    # Summary
    successful = sum(1 for r in results.values() if r is not None and len(r) > 0)
    total_edges = sum(len(r) if r is not None else 0 for r in results.values())
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Successfully detected edges in {successful}/{len(image_files)} images")
    print(f"Total edges detected: {total_edges}")
    
    if successful > 0:
        print(f"Average edges per successful image: {total_edges/successful:.1f}")
    
    if detector.debug and detector.save_visualizations:
        print(f"\nDebug visualizations saved as 'debug_*_result.png' files")
    
    return results

if __name__ == "__main__":
    results = main()