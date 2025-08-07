import cv2
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class WhiteboardEdgeDetector:
    def __init__(self):
        self.debug = True
        
    def find_whiteboard_surface_region(self, image: np.ndarray) -> np.ndarray:
        """
        Find the whiteboard surface using a focused approach
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        print(f"  Image size: {w}x{h}")
        
        # Focus on lower half where whiteboard surface should be
        lower_half = gray[h//2:, :]
        print(f"  Lower half intensity: min={lower_half.min()}, max={lower_half.max()}, mean={lower_half.mean():.1f}")
        
        # Create mask for whiteboard surface using multiple strategies
        masks = []
        
        # Strategy 1: Reasonably bright regions
        bright_thresh = max(150, np.percentile(lower_half, 70))
        bright_mask = (lower_half > bright_thresh).astype(np.uint8) * 255
        masks.append(('bright', bright_mask))
        
        # Strategy 2: Top 40% brightest pixels
        top_thresh = np.percentile(lower_half, 60)
        top_mask = (lower_half > top_thresh).astype(np.uint8) * 255
        masks.append(('top60', top_mask))
        
        # Strategy 3: Otsu thresholding
        _, otsu_mask = cv2.threshold(lower_half, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(('otsu', otsu_mask))
        
        # Choose best mask based on having a large connected component
        best_mask = None
        best_score = 0
        
        for name, mask in masks:
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
            
            if num_labels > 1:
                largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
                coverage = largest_area / (lower_half.shape[0] * lower_half.shape[1])
                
                print(f"    {name}: largest_area={largest_area}, coverage={coverage:.3f}")
                
                if coverage > 0.05 and largest_area > best_score:  # At least 5% coverage
                    best_score = largest_area
                    best_mask = clean_mask
                    print(f"    -> New best mask: {name}")
        
        if best_mask is None:
            print("  No suitable surface mask found, using fallback")
            # Fallback: simple brightness threshold
            fallback_thresh = np.percentile(lower_half, 75)
            best_mask = (lower_half > fallback_thresh).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create full image mask
        full_mask = np.zeros_like(gray, dtype=np.uint8)
        full_mask[h//2:, :] = best_mask
        
        # Keep only largest component in full mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
        if num_labels > 1:
            largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            full_mask = (labels == largest_component).astype(np.uint8) * 255
        
        final_area = np.sum(full_mask > 0)
        print(f"  Final surface area: {final_area} pixels ({final_area/(h*w):.3f} of image)")
        
        return full_mask
    
    def create_edge_detection_regions(self, surface_mask: np.ndarray) -> np.ndarray:
        """
        Create focused boundary region for edge detection - more generous for frame detection
        """
        # Create boundary region around the whiteboard surface
        kernel = np.ones((25, 25), np.uint8)  # Larger kernel for wider boundary region
        eroded = cv2.erode(surface_mask, kernel, iterations=1)
        dilated = cv2.dilate(surface_mask, kernel, iterations=3)  # More dilation
        boundary_region = cv2.subtract(dilated, eroded)
        
        pixels = np.sum(boundary_region > 0)
        print(f"  Boundary region: {pixels} pixels")
        
        return boundary_region
    
    def filter_structural_edges(self, edges: np.ndarray, boundary_region: np.ndarray) -> np.ndarray:
        """
        Filter edges to focus on structural boundaries (frame edges) and ignore content (drawings)
        """
        h, w = edges.shape
        
        # First apply boundary mask - only keep edges near the surface boundary
        boundary_edges = cv2.bitwise_and(edges, boundary_region)
        
        print(f"    Edges before boundary filtering: {np.sum(edges > 0)}")
        print(f"    Edges after boundary filtering: {np.sum(boundary_edges > 0)}")
        
        # Additional content filtering - remove edges that look like drawings/text
        
        # 1. Remove small connected components (likely text/small drawings)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(boundary_edges, connectivity=8)
        
        min_component_size = 8  # Reduced - was too strict
        filtered_edges = np.zeros_like(boundary_edges)
        
        structural_components = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Keep components that could be structural edges:
            # - Large enough area
            # - Elongated shape (frame edges should be long and thin)
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            
            # Relaxed criteria for structural edges
            if area >= min_component_size and aspect_ratio > 1.5:  # More lenient aspect ratio
                # Additional check: should be near image edges or be very linear
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                
                # More lenient boundary check - whiteboard edges can be anywhere
                near_boundary = (x < w * 0.25 or x + width > w * 0.75 or 
                               y < h * 0.25 or y + height > h * 0.75)
                
                # Accept if reasonably elongated OR near boundary OR large enough
                if near_boundary or aspect_ratio > 3 or area > 30:
                    component_mask = (labels == i)
                    filtered_edges[component_mask] = 255
                    structural_components += 1
                    print(f"      Kept component: area={area}, aspect={aspect_ratio:.1f}, near_boundary={near_boundary}")
                else:
                    print(f"      Rejected component: area={area}, aspect={aspect_ratio:.1f}, not near boundary")
            else:
                print(f"      Rejected component: area={area}, aspect={aspect_ratio:.1f} (too small/round)")
        
        print(f"    Structural components kept: {structural_components}/{num_labels-1}")
        print(f"    Edges after content filtering: {np.sum(filtered_edges > 0)}")
        
        # 2. Morphological operations to clean up and connect nearby structural edges
        kernel_clean = np.ones((3, 3), np.uint8)
        filtered_edges = cv2.morphologyEx(filtered_edges, cv2.MORPH_CLOSE, kernel_clean)
        
        # 3. Final cleanup: remove remaining curved or irregular patterns
        # Use contour analysis to remove non-linear shapes
        contours, _ = cv2.findContours(filtered_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_edges = np.zeros_like(filtered_edges)
        linear_contours = 0
        
        for contour in contours:
            # Approximate contour to see how linear it is
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Linear edges should approximate to very few points
            if len(approx) <= 6:  # More lenient - was 4
                cv2.drawContours(final_edges, [contour], -1, 255, -1)
                linear_contours += 1
                print(f"      Kept linear contour with {len(approx)} vertices")
            else:
                area = cv2.contourArea(contour)
                # Keep larger contours even if not perfectly linear
                if area > 50:  # Large contours might be frame edges
                    cv2.drawContours(final_edges, [contour], -1, 255, -1)
                    linear_contours += 1
                    print(f"      Kept large contour with {len(approx)} vertices, area={area}")
                else:
                    print(f"      Rejected curved contour with {len(approx)} vertices, area={area}")
        
        print(f"    Linear contours kept: {linear_contours}/{len(contours)}")
        print(f"    Final filtered edges: {np.sum(final_edges > 0)}")
        
        return final_edges
    
    def detect_clean_edges(self, image: np.ndarray, boundary_region: np.ndarray) -> List[np.ndarray]:
        """
        Detect edges with enhanced filtering to exclude drawings/content
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_images = []
        
        # Method 1: Standard Canny with moderate blur
        blur1 = cv2.GaussianBlur(gray, (5, 5), 1)
        edges1 = cv2.Canny(blur1, 50, 150, apertureSize=3)
        filtered1 = self.filter_structural_edges(edges1, boundary_region)
        edge_images.append(('Canny_50_150_filtered', filtered1))
        
        # Method 2: Lower threshold Canny
        edges2 = cv2.Canny(blur1, 30, 100, apertureSize=3)
        filtered2 = self.filter_structural_edges(edges2, boundary_region)
        edge_images.append(('Canny_30_100_filtered', filtered2))
        
        # Method 3: Bilateral filter + Canny (preserves strong edges, reduces noise)
        bilateral = cv2.bilateralFilter(gray, 11, 100, 100)
        edges3 = cv2.Canny(bilateral, 40, 120, apertureSize=3)
        filtered3 = self.filter_structural_edges(edges3, boundary_region)
        edge_images.append(('Bilateral_Canny_filtered', filtered3))
        
        # Method 4: Sobel with high threshold for strongest edges + filtering
        sobel_x = cv2.Sobel(blur1, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur1, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_thresh = np.percentile(sobel_mag, 94)  # Higher threshold to get only strongest
        sobel_edges = (sobel_mag > sobel_thresh).astype(np.uint8) * 255
        filtered4 = self.filter_structural_edges(sobel_edges, boundary_region)
        edge_images.append(('Sobel_filtered', filtered4))
        
        # Report edge pixel counts
        for name, edges in edge_images:
            pixel_count = np.sum(edges > 0)
            print(f"  {name}: {pixel_count} edge pixels")
        
        return edge_images
    
    def score_line_against_edges(self, rho: float, theta: float, edges: np.ndarray) -> Tuple[float, int]:
        """
        Score how well a line matches the actual edge pixels with improved sampling
        """
        h, w = edges.shape
        
        # Generate points along the line
        a = np.cos(theta)
        b = np.sin(theta)
        
        # Sample points along the line within image bounds
        edge_hits = 0
        total_samples = 0
        
        # Use denser sampling for better accuracy
        if abs(b) > abs(a):  # More horizontal line
            # Sample along x with higher density
            for x in range(0, w, 1):  # Every pixel
                y = (rho - x * a) / b
                if 0 <= y < h:
                    y_int = int(round(y))
                    if 0 <= y_int < h:
                        total_samples += 1
                        # Check in a small neighborhood around the line
                        hit_found = False
                        for dy in [-1, 0, 1]:  # Check 3x1 neighborhood
                            check_y = y_int + dy
                            if 0 <= check_y < h and edges[check_y, x] > 0:
                                hit_found = True
                                break
                        if hit_found:
                            edge_hits += 1
        else:  # More vertical line
            # Sample along y with higher density
            for y in range(0, h, 1):  # Every pixel
                x = (rho - y * b) / a
                if 0 <= x < w:
                    x_int = int(round(x))
                    if 0 <= x_int < w:
                        total_samples += 1
                        # Check in a small neighborhood around the line
                        hit_found = False
                        for dx in [-1, 0, 1]:  # Check 1x3 neighborhood
                            check_x = x_int + dx
                            if 0 <= check_x < w and edges[y, check_x] > 0:
                                hit_found = True
                                break
                        if hit_found:
                            edge_hits += 1
        
        if total_samples == 0:
            return 0.0, 0
        
        support_ratio = edge_hits / total_samples
        return support_ratio, edge_hits
    
    def classify_line_by_slope_and_position(self, rho: float, theta: float, image_shape: Tuple[int, int]) -> Tuple[str, str, float]:
        """
        Classify line by orientation and position, accounting for whiteboard geometry
        Returns: (orientation, position, slope)
        """
        h, w = image_shape
        angle_deg = np.degrees(theta)
        
        # Calculate slope for diagonal lines
        if 20 <= abs(angle_deg) <= 70 or 110 <= abs(angle_deg) <= 160:
            # This is a diagonal line - calculate actual slope
            dx = np.cos(theta)
            dy = np.sin(theta)
            slope = dy / dx if abs(dx) > 0.001 else float('inf')
            
            # Determine position based on where line intersects image
            a = np.cos(theta)
            b = np.sin(theta)
            
            # Find intersection with middle horizontal line
            mid_y = h // 2
            if abs(b) > 0.001:
                x_at_mid = (rho - mid_y * b) / a
                
                if x_at_mid < w // 2:
                    position = "left"
                else:
                    position = "right"
            else:
                position = "unknown"
            
            return "diagonal", position, slope
        
        # Standard horizontal/vertical classification
        elif (abs(angle_deg) < 20) or (abs(angle_deg - 180) < 20):
            return "horizontal", "center", 0.0
        elif (70 < angle_deg < 110):
            return "vertical", "center", float('inf')
        else:
            return "other", "unknown", 0.0
    
    def deduplicate_lines(self, lines: List[Tuple[float, float, float, int, str]]) -> List[Tuple[float, float, float, int, str]]:
        """
        Remove duplicate lines early in the pipeline
        """
        if len(lines) <= 1:
            return lines
        
        unique_lines = []
        
        for rho, theta, support, hits, edge_type in lines:
            angle_deg = np.degrees(theta)
            
            is_duplicate = False
            for i, (existing_rho, existing_theta, existing_support, existing_hits, existing_type) in enumerate(unique_lines):
                existing_angle = np.degrees(existing_theta)
                
                # Calculate differences
                rho_diff = abs(rho - existing_rho)
                angle_diff = min(abs(angle_deg - existing_angle), 
                               abs(angle_deg - existing_angle + 180),
                               abs(angle_deg - existing_angle - 180))
                
                # Lines are duplicates if both rho and angle are very similar
                if rho_diff < 10 and angle_diff < 2:
                    print(f"      Removing duplicate: rho_diff={rho_diff:.1f}, angle_diff={angle_diff:.1f}°")
                    
                    # Keep the one with better support
                    if support > existing_support:
                        # Replace existing with current (better) line
                        unique_lines[i] = (rho, theta, support, hits, edge_type)
                        print(f"        Replaced with better support ({support:.3f} > {existing_support:.3f})")
                    else:
                        print(f"        Kept existing with better support ({existing_support:.3f} > {support:.3f})")
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append((rho, theta, support, hits, edge_type))
        
        print(f"    Deduplication: {len(lines)} → {len(unique_lines)} lines")
        return unique_lines
    
    def validate_whiteboard_geometry(self, lines: List[Tuple[float, float, float, int, str]], 
                                   image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, int, str]]:
        """
        Sequential selection: Find best line first, then find second line with opposite slope sign
        """
        if len(lines) <= 1:
            return lines
        
        print(f"    Sequential selection for {len(lines)} lines:")
        
        # SAFETY CHECK: Never process more than 10 lines
        if len(lines) > 10:
            print(f"    WARNING: Too many lines ({len(lines)}), limiting to top 10")
            lines = lines[:10]
        
        # Helper function to calculate line slope consistently
        def calculate_slope(theta):
            # Convert from polar form (rho, theta) to Cartesian slope
            # For line: x*cos(theta) + y*sin(theta) = rho
            # Slope = -cos(theta)/sin(theta) when sin(theta) != 0
            
            angle_deg = np.degrees(theta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            print(f"        Debug slope calc: theta={angle_deg:.1f}°, sin={sin_theta:.3f}, cos={cos_theta:.3f}")
            
            if abs(sin_theta) < 0.01:  # Nearly horizontal line (theta ≈ 0° or 180°)
                return 0.0
            elif abs(cos_theta) < 0.01:  # Nearly vertical line (theta ≈ 90° or 270°)
                return float('inf')
            else:
                # Standard slope calculation
                slope = -cos_theta / sin_theta
                print(f"        Calculated slope: {slope:.3f}")
                return slope
        
        # Helper function to determine slope sign more robustly
        def get_slope_sign(slope, theta):
            angle_deg = np.degrees(theta)
            
            # Use angle-based classification for better accuracy
            if 85 <= angle_deg <= 95 or 265 <= angle_deg <= 275:  # Nearly vertical
                return 'vertical'
            elif abs(angle_deg) <= 5 or abs(angle_deg - 180) <= 5:  # Nearly horizontal
                return 'horizontal'
            elif 0 < angle_deg < 90 or 180 < angle_deg < 270:  # Positive slope quadrants
                return 'positive'  
            else:  # 90 < angle_deg < 180 or 270 < angle_deg < 360
                return 'negative'
        
        # Sort all lines by quality (support ratio and edge hits)
        sorted_lines = sorted(lines, key=lambda x: (x[2], x[3]), reverse=True)
        
        selected_lines = []
        
        # STEP 1: Select the BEST line (highest quality)
        best_line = sorted_lines[0]
        best_rho, best_theta, best_support, best_hits, best_type = best_line
        best_slope = calculate_slope(best_theta)
        best_slope_sign = get_slope_sign(best_slope, best_theta)
        best_angle = np.degrees(best_theta)
        
        selected_lines.append(best_line)
        print(f"    SELECTED Line 1 (BEST): rho={best_rho:.1f}, angle={best_angle:.1f}°, slope={best_slope:.3f} ({best_slope_sign}), support={best_support:.3f}")
        
        # STEP 2: Search for the best line with OPPOSITE slope sign
        print(f"    Searching for line with opposite slope sign to '{best_slope_sign}'...")
        
        best_opposite = None
        best_opposite_quality = -1
        
        for candidate in sorted_lines[1:]:  # Skip the first (already selected)
            candidate_rho, candidate_theta, candidate_support, candidate_hits, candidate_type = candidate
            candidate_slope = calculate_slope(candidate_theta)
            candidate_slope_sign = get_slope_sign(candidate_slope, candidate_theta)
            candidate_angle = np.degrees(candidate_theta)
            
            # ENHANCED similarity check - these lines are too similar!
            rho_diff = abs(best_rho - candidate_rho)
            angle_diff = min(abs(best_angle - candidate_angle), 
                            abs(best_angle - candidate_angle + 180),
                            abs(best_angle - candidate_angle - 180))
            
            # Much stricter similarity thresholds
            is_too_similar = (rho_diff < 50 and angle_diff < 10)  # Tightened from 30/5
            
            print(f"      Candidate: rho={candidate_rho:.1f}, angle={candidate_angle:.1f}°, slope={candidate_slope:.3f} ({candidate_slope_sign})")
            print(f"        rho_diff={rho_diff:.1f}px, angle_diff={angle_diff:.1f}°, too_similar={is_too_similar}")
            
            # Check if this candidate has opposite slope sign
            has_opposite_sign = False
            
            # Define what constitutes "opposite" slope signs for whiteboard corners
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
                print(f"        REJECTED: Too similar (rho_diff={rho_diff:.1f}px, angle_diff={angle_diff:.1f}°)")
                continue
                
            if not has_opposite_sign:
                print(f"        REJECTED: Same slope type ({best_slope_sign} vs {candidate_slope_sign})")
                continue
            
            # This candidate has opposite slope sign and is sufficiently different
            candidate_quality = candidate_support * candidate_hits  # Combined quality metric
            
            if candidate_quality > best_opposite_quality:
                best_opposite = candidate
                best_opposite_quality = candidate_quality
                print(f"        NEW BEST OPPOSITE: quality={candidate_quality:.1f}")
            else:
                print(f"        REJECTED: Lower quality than current best opposite")
        
        # Add the best opposite line if found
        if best_opposite:
            selected_lines.append(best_opposite)
            opp_rho, opp_theta, opp_support, opp_hits, opp_type = best_opposite
            opp_slope = calculate_slope(opp_theta)
            opp_slope_sign = get_slope_sign(opp_slope, opp_theta)
            opp_angle = np.degrees(opp_theta)
            print(f"    SELECTED Line 2 (OPPOSITE): rho={opp_rho:.1f}, angle={opp_angle:.1f}°, slope={opp_slope:.3f} ({opp_slope_sign}), support={opp_support:.3f}")
            print(f"    Final slope comparison: {best_slope:.3f} ({best_slope_sign}) vs {opp_slope:.3f} ({opp_slope_sign})")
        else:
            print(f"    NO OPPOSITE LINE FOUND: Only returning the best line")
            print(f"    All other candidates were either too similar or had the same slope type")
        
        print(f"    Final result: {len(selected_lines)} line(s)")
        return selected_lines
    
    def find_best_lines(self, edge_images: List[Tuple[str, np.ndarray]]) -> List[Tuple[float, float]]:
        """
        Find the best whiteboard edge lines with early deduplication and strict limits
        """
        h, w = edge_images[0][1].shape
        all_scored_lines = []
        
        # Try each edge detection method
        for method_name, edges in edge_images:
            print(f"\n  Analyzing {method_name}:")
            
            edge_pixel_count = np.sum(edges > 0)
            if edge_pixel_count < 20:
                print(f"    Skipping - too few edge pixels ({edge_pixel_count})")
                continue
            
            print(f"    Edge pixels available: {edge_pixel_count}")
            
            # Try different Hough parameter sets - start with higher thresholds
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
                    
                    # STRICT LIMIT: Only look at top 20 lines to avoid explosion
                    lines = lines[:20]
                    print(f"    Limiting to first {len(lines)} lines")
                    
                    # Score all limited lines
                    scored_candidates = []
                    
                    for line in lines:
                        rho, theta = line[0]
                        angle_deg = np.degrees(theta)
                        
                        # Strict angle classification
                        is_horizontal = (abs(angle_deg) < 15) or (abs(angle_deg - 180) < 15)
                        is_vertical = (75 < angle_deg < 105)
                        is_diagonal = (15 <= abs(angle_deg) <= 75) or (105 <= abs(angle_deg) <= 165)
                        
                        if is_horizontal or is_vertical or is_diagonal:
                            # Score the line against actual edge pixels
                            support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                            
                            # More strict thresholds to limit results
                            if support_ratio > 0.05 and edge_hits > 10:
                                if is_horizontal:
                                    edge_type = 'H'
                                elif is_vertical:
                                    edge_type = 'V'
                                else:
                                    edge_type = 'D'  # Diagonal
                                
                                scored_candidates.append((rho, theta, support_ratio, edge_hits, edge_type))
                                print(f"      {edge_type} candidate: rho={rho:.1f}, angle={angle_deg:.1f}°, support={support_ratio:.3f}, hits={edge_hits}")
                    
                    print(f"    After scoring: {len(scored_candidates)} candidates")
                    
                    if scored_candidates:
                        method_lines.extend(scored_candidates)
                        print(f"    Method contributed: {len(scored_candidates)} lines")
                        
                        if len(method_lines) >= 2:  # Stop if we have enough
                            print(f"    Stopping - already have {len(method_lines)} lines")
                            break
            
            all_scored_lines.extend(method_lines)
            
            # GLOBAL LIMIT: Stop if we already have 2 lines
            if len(all_scored_lines) >= 2:
                print(f"  Stopping methods - already have {len(all_scored_lines)} lines")
                break
        
        # EARLY DEDUPLICATION - add this before geometry validation
        if all_scored_lines:
            print(f"\n  Applying early deduplication...")
            all_scored_lines = self.deduplicate_lines(all_scored_lines)
        
        # Apply geometry validation if we have multiple lines
        if len(all_scored_lines) > 1:
            print(f"  Applying sequential selection...")
            all_scored_lines = self.validate_whiteboard_geometry(all_scored_lines, (h, w))
        
        # FINAL SAFETY CHECK
        if len(all_scored_lines) > 2:
            print(f"  ERROR: Have {len(all_scored_lines)} lines, this should never happen!")
            all_scored_lines = all_scored_lines[:2]
            print(f"  Emergency limit applied - reduced to {len(all_scored_lines)} lines")
        
        if not all_scored_lines:
            print("  No lines found")
            return []
        
        # Sort by quality and return
        all_scored_lines.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        print(f"\n  Final validated lines:")
        for i, (rho, theta, support, hits, edge_type) in enumerate(all_scored_lines):
            angle = np.degrees(theta)
            print(f"    {i+1}. {edge_type}: rho={rho:.1f}, angle={angle:.1f}°, support={support:.3f}, hits={hits}")
        
        # Return only the line coordinates
        final_lines = [(rho, theta) for rho, theta, support, hits, edge_type in all_scored_lines]
        
        print(f"\n  FINAL RESULT: {len(final_lines)} lines (MAX ALLOWED: 2)")
        return final_lines
    
    def detect_whiteboard_edges(self, image_path: str) -> Optional[List[Tuple[float, float]]]:
        """
        Main detection function focused on clean edge detection
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        print(f"\nProcessing: {os.path.basename(image_path)}")
        print("=" * 60)
        
        try:
            # Step 1: Find whiteboard surface region
            print("Step 1: Finding whiteboard surface...")
            surface_mask = self.find_whiteboard_surface_region(image)
            
            surface_area = np.sum(surface_mask > 0)
            if surface_area < 500:
                print(f"  FAIL: Surface area too small ({surface_area} pixels)")
                return None
            
            # Step 2: Create boundary region for edge detection
            print("\nStep 2: Creating boundary region...")
            boundary_region = self.create_edge_detection_regions(surface_mask)
            
            # Step 3: Detect clean structural edges
            print("\nStep 3: Detecting edges...")
            edge_images = self.detect_clean_edges(image, boundary_region)
            
            # Step 4: Find the best lines
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
                # Use the best edge image for visualization
                best_edge_image = max(edge_images, key=lambda x: np.sum(x[1] > 0))
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
        """
        Clean debug visualization focused on the core detection
        """
        try:
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
            
            for i, (rho, theta) in enumerate(lines):
                color = colors_bgr[i % len(colors_bgr)]
                angle_deg = np.degrees(theta)
                
                print(f"    Drawing line {i+1}: rho={rho:.1f}, theta={angle_deg:.1f}°, color={color}")
                
                # Improved line drawing - find actual intersections with image boundaries
                h_img, w_img = image.shape[:2]
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
                elif len(unique_intersections) == 1:
                    print(f"      ERROR: Only one intersection found: {unique_intersections[0]}")
                    print(f"      Line equation may be degenerate or outside image bounds")
                else:
                    print(f"      ERROR: No valid intersections found for line {i+1}")
                    print(f"      rho={rho:.1f}, theta={angle_deg:.1f}°")
                    print(f"      This line will not be visible in the result")
            
            axes[1, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title(f'Result ({len(lines)} lines)')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")

def main():
    detector = WhiteboardEdgeDetector()
    
    # Find images
    image_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print("No JPG images found in current directory")
        return
    
    print(f"Found {len(image_files)} image(s)")
    
    results = {}
    
    # Process each image
    for image_path in sorted(image_files):
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
    
    return results

if __name__ == "__main__":
    results = main()