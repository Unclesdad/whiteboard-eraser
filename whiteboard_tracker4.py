import cv2
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Try to import Numba for JIT compilation (optional for Pi)
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def njit(func):
        return func
    def jit(func):
        return func

class OptimizedWhiteboardDetector:
    def __init__(self, debug: bool = False, enable_threading: bool = True):
        self.debug = debug
        self.enable_threading = enable_threading
        # Pre-calculate reusable kernels - KEPT ORIGINAL SIZES for accuracy
        self.kernel_7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_3x3 = np.ones((3, 3), np.uint8)
        self.kernel_25x25 = np.ones((25, 25), np.uint8)  # KEPT ORIGINAL SIZE
        
        # Pre-calculate commonly used values (kept from earlier optimizations)
        self._cos_cache = {}
        self._sin_cache = {}
        
        # Thread pool for parallel processing (reused across frames)
        if self.enable_threading:
            self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="whiteboard")
        else:
            self._thread_pool = None
        
        # Optimize OpenCV for Pi
        cv2.setUseOptimized(True)
        cv2.setNumThreads(2)  # Pi 5 has 4 cores, leave some for system
        
    def find_whiteboard_surface_region(self, image: np.ndarray) -> np.ndarray:
        """Find the whiteboard surface using a focused approach - ORIGINAL LOGIC PRESERVED"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        h_half = h // 2
        
        if self.debug:
            print(f"  Image size: {w}x{h}")
        
        # Focus on lower half where whiteboard surface should be
        lower_half = gray[h_half:, :]
        lower_mean = lower_half.mean()
        if self.debug:
            print(f"  Lower half intensity: min={lower_half.min()}, max={lower_half.max()}, mean={lower_mean:.1f}")
        
        # Pre-calculate percentiles once for efficiency
        percentiles = np.percentile(lower_half, [60, 70, 75])
        p60, p70, p75 = percentiles
        
        # Create mask for whiteboard surface using multiple strategies - ORIGINAL LOGIC
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
        
        # Choose best mask based on having a large connected component - ORIGINAL LOGIC
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
                
                if self.debug:
                    print(f"    {name}: largest_area={largest_area}, coverage={coverage:.3f}")
                
                if coverage > 0.05 and largest_area > best_score:
                    best_score = largest_area
                    best_mask = clean_mask
                    if self.debug:
                        print(f"    -> New best mask: {name}")
        
        if best_mask is None:
            if self.debug:
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
        if self.debug:
            print(f"  Final surface area: {final_area} pixels ({final_area/(h*w):.3f} of image)")
        
        return full_mask
    
    def create_edge_detection_regions(self, surface_mask: np.ndarray) -> np.ndarray:
        """Create focused boundary region for edge detection - ORIGINAL LOGIC PRESERVED"""
        eroded = cv2.erode(surface_mask, self.kernel_25x25, iterations=1)
        dilated = cv2.dilate(surface_mask, self.kernel_25x25, iterations=3)
        boundary_region = cv2.subtract(dilated, eroded)
        
        if self.debug:
            print(f"  Boundary region: {np.count_nonzero(boundary_region)} pixels")
        return boundary_region
    
    def filter_structural_edges(self, edges: np.ndarray, boundary_region: np.ndarray) -> np.ndarray:
        """Filter edges to focus on structural boundaries - ORIGINAL LOGIC PRESERVED"""
        h, w = edges.shape
        
        # Apply boundary mask
        boundary_edges = cv2.bitwise_and(edges, boundary_region)
        
        if self.debug:
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
                    if self.debug:
                        print(f"      Kept component: area={area}, aspect={aspect_ratio:.1f}, near_boundary={near_boundary}")
                else:
                    if self.debug:
                        print(f"      Rejected component: area={area}, aspect={aspect_ratio:.1f}, not near boundary")
            else:
                if self.debug:
                    print(f"      Rejected component: area={area}, aspect={aspect_ratio:.1f} (too small/round)")
        
        if self.debug:
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
                    if self.debug:
                        print(f"      Kept linear contour with {len(approx)} vertices")
                else:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        cv2.drawContours(final_edges, [contour], -1, 255, -1)
                        linear_contours += 1
                        if self.debug:
                            print(f"      Kept large contour with {len(approx)} vertices, area={area}")
                    else:
                        if self.debug:
                            print(f"      Rejected curved contour with {len(approx)} vertices, area={area}")
        
        if self.debug:
            print(f"    Linear contours kept: {linear_contours}/{len(contours)}")
            print(f"    Final filtered edges: {np.count_nonzero(final_edges)}")
        
        return final_edges
    
    def detect_clean_edges(self, image: np.ndarray, boundary_region: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Detect edges with enhanced filtering - ORIGINAL METHODS PRESERVED + PARALLEL PROCESSING"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pre-compute blurred versions once
        blur5 = cv2.GaussianBlur(gray, (5, 5), 1)
        bilateral = cv2.bilateralFilter(gray, 11, 100, 100)
        
        if self.enable_threading and self._thread_pool:
            # PARALLEL PROCESSING: Run edge detection methods in parallel
            def process_canny_standard():
                edges = cv2.Canny(blur5, 50, 150, apertureSize=3)
                filtered = self.filter_structural_edges(edges, boundary_region)
                return ('Canny_50_150_filtered', filtered)
            
            def process_canny_low():
                edges = cv2.Canny(blur5, 30, 100, apertureSize=3)
                filtered = self.filter_structural_edges(edges, boundary_region)
                return ('Canny_30_100_filtered', filtered)
            
            def process_bilateral():
                edges = cv2.Canny(bilateral, 40, 120, apertureSize=3)
                filtered = self.filter_structural_edges(edges, boundary_region)
                return ('Bilateral_Canny_filtered', filtered)
            
            def process_sobel():
                sobel_x = cv2.Sobel(blur5, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(blur5, cv2.CV_64F, 0, 1, ksize=3)
                sobel_mag = np.hypot(sobel_x, sobel_y)
                sobel_thresh = np.percentile(sobel_mag, 94)
                sobel_edges = ((sobel_mag > sobel_thresh) * 255).astype(np.uint8)
                filtered = self.filter_structural_edges(sobel_edges, boundary_region)
                return ('Sobel_filtered', filtered)
            
            # Submit all edge detection tasks in parallel
            futures = [
                self._thread_pool.submit(process_canny_standard),
                self._thread_pool.submit(process_canny_low),
                self._thread_pool.submit(process_bilateral),
                self._thread_pool.submit(process_sobel)
            ]
            
            # Collect results as they complete
            edge_results = []
            for future in futures:
                try:
                    result = future.result(timeout=2.0)  # 2 second timeout per method
                    edge_results.append(result)
                except Exception as e:
                    if self.debug:
                        print(f"    Edge detection thread failed: {e}")
                    continue
        
        else:
            # SEQUENTIAL PROCESSING: Original method
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
        if self.debug:
            for name, edges in edge_results:
                print(f"  {name}: {np.count_nonzero(edges)} edge pixels")
        
        return edge_results
    
    def score_multiple_lines_parallel(self, lines_data: List[Tuple], edges: np.ndarray) -> List[Tuple]:
        """Score multiple lines in parallel for better performance"""
        if not self.enable_threading or not self._thread_pool or len(lines_data) < 4:
            # Fall back to sequential processing for small line counts
            results = []
            for rho, theta, angle_deg, is_horizontal, is_vertical, is_diagonal in lines_data:
                support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                results.append((rho, theta, angle_deg, is_horizontal, is_vertical, is_diagonal, support_ratio, edge_hits))
            return results
        
        # PARALLEL LINE SCORING: Split lines into chunks for parallel processing
        chunk_size = max(2, len(lines_data) // 2)  # Use 2 threads
        chunks = [lines_data[i:i + chunk_size] for i in range(0, len(lines_data), chunk_size)]
        
        def score_chunk(chunk):
            chunk_results = []
            for rho, theta, angle_deg, is_horizontal, is_vertical, is_diagonal in chunk:
                support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                chunk_results.append((rho, theta, angle_deg, is_horizontal, is_vertical, is_diagonal, support_ratio, edge_hits))
            return chunk_results
        
        # Submit chunks to thread pool
        futures = [self._thread_pool.submit(score_chunk, chunk) for chunk in chunks]
        
        # Collect results
        results = []
        for future in futures:
            try:
                chunk_results = future.result(timeout=1.0)
                results.extend(chunk_results)
            except Exception as e:
                if self.debug:
                    print(f"    Line scoring thread failed: {e}")
                continue
        
        return results
    @staticmethod
    @njit
    def score_line_numba(rho, theta, edges, h, w):
        """Numba-optimized line scoring if available - ORIGINAL LOGIC"""
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
        """Score how well a line matches the actual edge pixels - ORIGINAL LOGIC PRESERVED"""
        h, w = edges.shape
        
        if NUMBA_AVAILABLE:
            return self.score_line_numba(rho, theta, edges, h, w)
        
        # Fallback to vectorized numpy version - ORIGINAL LOGIC
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
        """Remove duplicate lines - ORIGINAL LOGIC PRESERVED"""
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
                    if self.debug:
                        print(f"      Removing duplicate: rho_diff={rho_diff:.1f}, angle_diff={angle_diff:.1f}°")
                    if support > existing[2]:
                        unique_lines[i] = line
                        if self.debug:
                            print(f"        Replaced with better support ({support:.3f} > {existing[2]:.3f})")
                    else:
                        if self.debug:
                            print(f"        Kept existing with better support ({existing[2]:.3f} > {support:.3f})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        if self.debug:
            print(f"    Deduplication: {len(lines)} → {len(unique_lines)} lines")
        return unique_lines
    
    def validate_whiteboard_geometry(self, lines: List[Tuple[float, float, float, int, str]], 
                                   image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, int, str]]:
        """Sequential selection - ORIGINAL LOGIC PRESERVED"""
        if len(lines) <= 1:
            return lines
        
        if self.debug:
            print(f"    Sequential selection for {len(lines)} lines:")
        
        if len(lines) > 10:
            if self.debug:
                print(f"    WARNING: Too many lines ({len(lines)}), limiting to top 10")
            lines = lines[:10]
        
        def calculate_slope(theta):
            angle_deg = np.degrees(theta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            if self.debug:
                print(f"        Debug slope calc: theta={angle_deg:.1f}°, sin={sin_theta:.3f}, cos={cos_theta:.3f}")
            
            if abs(sin_theta) < 0.01:
                return 0.0
            elif abs(cos_theta) < 0.01:
                return float('inf')
            else:
                slope = -cos_theta / sin_theta
                if self.debug:
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
        if self.debug:
            print(f"    SELECTED Line 1 (BEST): rho={best_rho:.1f}, angle={best_angle:.1f}°, slope={best_slope:.3f} ({best_slope_sign}), support={best_support:.3f}")
        
        # Search for best opposite slope
        if self.debug:
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
            
            if self.debug:
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
            
            if self.debug:
                print(f"        opposite_sign={has_opposite_sign} ({best_slope_sign} vs {candidate_slope_sign})")
            
            if is_too_similar:
                if self.debug:
                    print(f"        REJECTED: Too similar")
                continue
                
            if not has_opposite_sign:
                if self.debug:
                    print(f"        REJECTED: Same slope type")
                continue
            
            candidate_quality = candidate_support * candidate_hits
            
            if candidate_quality > best_opposite_quality:
                best_opposite = candidate
                best_opposite_quality = candidate_quality
                if self.debug:
                    print(f"        NEW BEST OPPOSITE: quality={candidate_quality:.1f}")
            else:
                if self.debug:
                    print(f"        REJECTED: Lower quality than current best opposite")
        
        if best_opposite:
            selected_lines.append(best_opposite)
            opp_rho, opp_theta, opp_support, opp_hits, opp_type = best_opposite
            opp_slope = calculate_slope(opp_theta)
            opp_slope_sign = get_slope_sign(opp_slope, opp_theta)
            opp_angle = np.degrees(opp_theta)
            if self.debug:
                print(f"    SELECTED Line 2 (OPPOSITE): rho={opp_rho:.1f}, angle={opp_angle:.1f}°, slope={opp_slope:.3f} ({opp_slope_sign}), support={opp_support:.3f}")
        else:
            if self.debug:
                print(f"    NO OPPOSITE LINE FOUND")
        
        if self.debug:
            print(f"    Final result: {len(selected_lines)} line(s)")
        return selected_lines
    
    def find_best_lines(self, edge_images: List[Tuple[str, np.ndarray]]) -> List[Tuple[float, float]]:
        """Find the best whiteboard edge lines - ORIGINAL LOGIC PRESERVED"""
        h, w = edge_images[0][1].shape
        all_scored_lines = []
        
        # Try each edge detection method
        for method_name, edges in edge_images:
            if self.debug:
                print(f"\n  Analyzing {method_name}:")
            
            edge_pixel_count = np.count_nonzero(edges)
            if edge_pixel_count < 20:
                if self.debug:
                    print(f"    Skipping - too few edge pixels ({edge_pixel_count})")
                continue
            
            if self.debug:
                print(f"    Edge pixels available: {edge_pixel_count}")
            
            # Try different Hough parameter sets - ORIGINAL THRESHOLDS
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
                    if self.debug:
                        print(f"    HoughLines (thresh={config['threshold']}): {len(lines)} raw lines")
                    
                    # Limit to top 20 lines - ORIGINAL LIMIT
                    lines = lines[:20]
                    if self.debug:
                        print(f"    Limiting to first {len(lines)} lines")
                    
                    scored_candidates = []
                    
                    # ORIGINAL LOGIC: Process each line individually
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
                                if self.debug:
                                    print(f"      {edge_type} candidate: rho={rho:.1f}, angle={angle_deg:.1f}°, support={support_ratio:.3f}, hits={edge_hits}")
                    
                    if self.debug:
                        print(f"    After scoring: {len(scored_candidates)} candidates")
                    
                    if scored_candidates:
                        method_lines.extend(scored_candidates)
                        if self.debug:
                            print(f"    Method contributed: {len(scored_candidates)} lines")
                        
                        if len(method_lines) >= 2:
                            if self.debug:
                                print(f"    Stopping - already have {len(method_lines)} lines")
                            break
            
            all_scored_lines.extend(method_lines)
            
            if len(all_scored_lines) >= 2:
                if self.debug:
                    print(f"  Stopping methods - already have {len(all_scored_lines)} lines")
                break
        
        # Early deduplication
        if all_scored_lines:
            if self.debug:
                print(f"\n  Applying early deduplication...")
            all_scored_lines = self.deduplicate_lines(all_scored_lines)
        
        # Apply geometry validation
        if len(all_scored_lines) > 1:
            if self.debug:
                print(f"  Applying sequential selection...")
            all_scored_lines = self.validate_whiteboard_geometry(all_scored_lines, (h, w))
        
        # Final safety check
        if len(all_scored_lines) > 2:
            if self.debug:
                print(f"  ERROR: Have {len(all_scored_lines)} lines, limiting to 2")
            all_scored_lines = all_scored_lines[:2]
        
        if not all_scored_lines:
            if self.debug:
                print("  No lines found")
            return []
        
        # Sort by quality
        all_scored_lines.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        if self.debug:
            print(f"\n  Final validated lines:")
            for i, (rho, theta, support, hits, edge_type) in enumerate(all_scored_lines):
                angle = np.degrees(theta)
                print(f"    {i+1}. {edge_type}: rho={rho:.1f}, angle={angle:.1f}°, support={support:.3f}, hits={hits}")
        
        # Return only the line coordinates
        return [(rho, theta) for rho, theta, support, hits, edge_type in all_scored_lines]
    
    def __del__(self):
        """Clean up thread pool on destruction"""
        if hasattr(self, '_thread_pool') and self._thread_pool:
            self._thread_pool.shutdown(wait=False)
    
    def detect_whiteboard_edges(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Main detection function - ORIGINAL FLOW PRESERVED, optimized for camera input"""
        if image is None:
            return None
        
        try:
            # Step 1: Find whiteboard surface
            surface_mask = self.find_whiteboard_surface_region(image)
            
            surface_area = np.count_nonzero(surface_mask)
            if surface_area < 500:
                if self.debug:
                    print(f"  FAIL: Surface area too small ({surface_area} pixels)")
                return None
            
            # Step 2: Create boundary region
            boundary_region = self.create_edge_detection_regions(surface_mask)
            
            # Step 3: Detect edges
            edge_images = self.detect_clean_edges(image, boundary_region)
            
            # Step 4: Find lines
            detected_lines = self.find_best_lines(edge_images)
            
            if not detected_lines:
                if self.debug:
                    print("  No lines detected")
                return None
            
            if self.debug:
                print(f"\nSUCCESS: Found {len(detected_lines)} edge line(s)")
                for i, (rho, theta) in enumerate(detected_lines):
                    angle = np.degrees(theta)
                    edge_type = "horizontal" if abs(angle) < 45 or abs(angle) > 135 else "vertical"
                    print(f"  Edge {i+1}: {edge_type}, rho={rho:.1f}, angle={angle:.1f}°")
            
            return detected_lines
            
        except Exception as e:
            if self.debug:
                print(f"  ERROR during processing: {str(e)}")
            return None


def process_single_image(image_path: str, detector: OptimizedWhiteboardDetector) -> Optional[List[Tuple[float, float]]]:
    """Process a single image file - for testing"""
    image = cv2.imread(image_path)
    return detector.detect_whiteboard_edges(image)

def create_debug_visualization(image, surface_mask, edges, lines, filename):
    """Create 4-panel debug visualization matching the original format"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top left: Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Original: {os.path.basename(filename)}')
        axes[0, 0].axis('off')
        
        # Top right: Whiteboard surface mask
        axes[0, 1].imshow(surface_mask, cmap='gray')
        axes[0, 1].set_title('Whiteboard Surface')
        axes[0, 1].axis('off')
        
        # Bottom left: Filtered edges (Sobel for consistency)
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edges: Sobel_filtered')
        axes[1, 0].axis('off')
        
        # Bottom right: Result with detected lines
        result_img = image.copy()
        
        if lines:
            h, w = image.shape[:2]
            colors_bgr = [(0, 0, 255), (0, 255, 0)]  # Red, Green
            
            for i, (rho, theta) in enumerate(lines):
                color = colors_bgr[i % len(colors_bgr)]
                
                # Calculate line endpoints
                a, b = np.cos(theta), np.sin(theta)
                
                # Find intersections with image boundaries
                intersections = []
                
                # Left edge (x = 0)
                if abs(b) > 0.001:
                    y_left = rho / b
                    if 0 <= y_left <= h:
                        intersections.append((0, int(y_left)))
                
                # Right edge (x = w)
                if abs(b) > 0.001:
                    y_right = (rho - w * a) / b
                    if 0 <= y_right <= h:
                        intersections.append((w, int(y_right)))
                
                # Top edge (y = 0)
                if abs(a) > 0.001:
                    x_top = rho / a
                    if 0 <= x_top <= w:
                        intersections.append((int(x_top), 0))
                
                # Bottom edge (y = h)
                if abs(a) > 0.001:
                    x_bottom = (rho - h * b) / a
                    if 0 <= x_bottom <= w:
                        intersections.append((int(x_bottom), h))
                
                # Draw line if we have enough intersections
                if len(intersections) >= 2:
                    cv2.line(result_img, intersections[0], intersections[1], color, 3)
        
        axes[1, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Result ({len(lines) if lines else 0} lines)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        base_name = os.path.splitext(filename)[0]
        output_filename = f'debug_{base_name}_result.png'
        plt.savefig(output_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Debug visualization saved: {output_filename}")
        
    except Exception as e:
        print(f"Visualization error: {e}")

def main():
    """Debug main function - processes all images and saves visualizations"""
    detector = OptimizedWhiteboardDetector(debug=True, enable_threading=True)
    
    # Find images
    image_files = []
    for ext in ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']:
        image_files.extend(glob.glob(f'*.{ext}'))
    
    if not image_files:
        print("No images found in current directory")
        return
    
    print(f"Found {len(image_files)} image(s) - processing for debug visualization...")
    print(f"Numba JIT available: {NUMBA_AVAILABLE}")
    
    for image_path in image_files:
        print(f"\nProcessing: {image_path}")
        print("=" * 60)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ERROR: Could not load {image_path}")
            continue
        
        start_time = time.time()
        
        try:
            # Step 1: Find whiteboard surface
            print("Step 1: Finding whiteboard surface...")
            surface_mask = detector.find_whiteboard_surface_region(image)
            
            surface_area = np.count_nonzero(surface_mask)
            if surface_area < 500:
                print(f"  FAIL: Surface area too small ({surface_area} pixels)")
                continue
            
            # Step 2: Create boundary region  
            print("\nStep 2: Creating boundary region...")
            boundary_region = detector.create_edge_detection_regions(surface_mask)
            
            # Step 3: Detect edges (get all edge images for visualization)
            print("\nStep 3: Detecting edges...")
            edge_images = detector.detect_clean_edges(image, boundary_region)
            
            # Get the Sobel edge detection for visualization
            sobel_edges = None
            for name, edges in edge_images:
                if 'Sobel' in name:
                    sobel_edges = edges
                    break
            
            # Fallback to first edge method if Sobel not found
            if sobel_edges is None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sobel_edges = edge_images[0][1] if edge_images else np.zeros_like(gray)
            
            # Step 4: Find lines
            print("\nStep 4: Finding lines...")
            detected_lines = detector.find_best_lines(edge_images)
            
            process_time = time.time() - start_time
            
            print(f"\nProcessing completed in {process_time:.2f}s")
            print(f"Found {len(detected_lines) if detected_lines else 0} edges")
            
            # Create visualization like the original example
            create_debug_visualization(image, surface_mask, sobel_edges, detected_lines, image_path)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 60)
    
    print("\nDebug processing complete!")

if __name__ == "__main__":
    main()