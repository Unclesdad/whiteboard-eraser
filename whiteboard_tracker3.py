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
        Create focused boundary region for edge detection
        """
        # Create boundary region around the whiteboard surface
        kernel = np.ones((15, 15), np.uint8)
        eroded = cv2.erode(surface_mask, kernel, iterations=1)
        dilated = cv2.dilate(surface_mask, kernel, iterations=2)
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
        
        min_component_size = 15  # Minimum size for a structural edge component
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
            
            # More strict criteria for structural edges
            if area >= min_component_size and aspect_ratio > 2.5:  # Long and thin
                # Additional check: should be near image edges or be very linear
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                
                # Structural edges are typically near image boundaries
                near_boundary = (x < w * 0.15 or x + width > w * 0.85 or 
                               y < h * 0.15 or y + height > h * 0.85)
                
                if near_boundary or aspect_ratio > 4:  # Very elongated or near boundary
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
            if len(approx) <= 4:  # Very linear
                cv2.drawContours(final_edges, [contour], -1, 255, -1)
                linear_contours += 1
                print(f"      Kept linear contour with {len(approx)} vertices")
            else:
                area = cv2.contourArea(contour)
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
    
    def validate_whiteboard_geometry(self, lines: List[Tuple[float, float, float, int, str]], 
                                   image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, int, str]]:
        """
        Validate lines based on whiteboard corner geometry - STRICT 2 LINE LIMIT
        """
        if len(lines) <= 1:
            return lines
        
        print(f"    Validating whiteboard geometry for {len(lines)} lines:")
        
        # SAFETY CHECK: Never process more than 10 lines
        if len(lines) > 10:
            print(f"    WARNING: Too many lines ({len(lines)}), limiting to top 10")
            lines = lines[:10]
        
        # Classify all lines
        classified_lines = []
        for rho, theta, support, hits, edge_type in lines:
            orientation, position, slope = self.classify_line_by_slope_and_position(rho, theta, image_shape)
            classified_lines.append((rho, theta, support, hits, edge_type, orientation, position, slope))
            
            angle_deg = np.degrees(theta)
            print(f"      Line: rho={rho:.1f}, angle={angle_deg:.1f}°, {orientation}, {position}, slope={slope:.2f}")
        
        # If we have diagonal lines, apply whiteboard corner constraints
        diagonal_lines = [line for line in classified_lines if line[5] == "diagonal"]
        
        if len(diagonal_lines) >= 2:
            print(f"    Found {len(diagonal_lines)} diagonal lines - applying corner constraints")
            
            # Separate by position and slope
            left_positive = [line for line in diagonal_lines if line[6] == "left" and line[7] > 0]
            right_negative = [line for line in diagonal_lines if line[6] == "right" and line[7] < 0]
            
            print(f"      Left positive slopes: {len(left_positive)}")
            print(f"      Right negative slopes: {len(right_negative)}")
            
            # Find the single best valid pair
            if left_positive and right_negative:
                # Sort each group by quality and take the best from each
                left_positive.sort(key=lambda x: (x[2], x[3]), reverse=True)  # Sort by support, hits
                right_negative.sort(key=lambda x: (x[2], x[3]), reverse=True)
                
                best_left = left_positive[0]
                best_right = right_negative[0]
                
                print(f"    Selected corner pair:")
                print(f"      Left: rho={best_left[0]:.1f}, slope={best_left[7]:.2f}, support={best_left[2]:.3f}")
                print(f"      Right: rho={best_right[0]:.1f}, slope={best_right[7]:.2f}, support={best_right[2]:.3f}")
                
                # Return exactly 2 lines
                return [(best_left[0], best_left[1], best_left[2], best_left[3], 'L'),
                        (best_right[0], best_right[1], best_right[2], best_right[3], 'R')]
        
        # If no valid diagonal pairs, return the best 2 lines overall
        print(f"    No valid corner pairs found, selecting best 2 lines overall")
        
        # Sort all lines by quality and take top 2
        classified_lines.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        # ABSOLUTE LIMIT: Return at most 2 lines
        max_lines = min(2, len(classified_lines))
        selected_lines = classified_lines[:max_lines]
        
        print(f"    Selected {len(selected_lines)} best lines:")
        for i, line in enumerate(selected_lines):
            print(f"      {i+1}. rho={line[0]:.1f}, angle={np.degrees(line[1]):.1f}°, support={line[2]:.3f}")
        
        return [(line[0], line[1], line[2], line[3], line[4]) for line in selected_lines]
    
    def find_best_lines(self, edge_images: List[Tuple[str, np.ndarray]]) -> List[Tuple[float, float]]:
        """
        Find the best whiteboard edge lines with strict limits and validation
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
                        # STRICT LIMIT: Apply geometry validation but limit results
                        validated_lines = self.validate_whiteboard_geometry(scored_candidates, (h, w))
                        
                        # ABSOLUTE LIMIT: Never more than 2 lines total
                        if len(validated_lines) > 2:
                            print(f"    WARNING: Geometry validation returned {len(validated_lines)} lines, limiting to 2")
                            validated_lines = validated_lines[:2]
                        
                        method_lines.extend(validated_lines)
                        print(f"    Method contributed: {len(validated_lines)} lines")
                        
                        if len(method_lines) >= 2:  # Stop if we have enough
                            print(f"    Stopping - already have {len(method_lines)} lines")
                            break
            
            all_scored_lines.extend(method_lines)
            
            # GLOBAL LIMIT: Stop if we already have 2 lines
            if len(all_scored_lines) >= 2:
                print(f"  Stopping methods - already have {len(all_scored_lines)} lines")
                break
        
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
            
            for i, (rho, theta) in enumerate(lines):
                color = colors_bgr[i % len(colors_bgr)]
                
                # Improved line drawing - find actual intersections with image boundaries
                h_img, w_img = image.shape[:2]
                intersections = []
                
                a = np.cos(theta)
                b = np.sin(theta)
                
                # Check intersection with all four image edges
                # Left edge (x = 0)
                if abs(a) > 0.001:
                    y_left = rho / b if abs(b) > 0.001 else float('inf')
                    if 0 <= y_left <= h_img:
                        intersections.append((0, int(y_left)))
                
                # Right edge (x = w_img)
                if abs(a) > 0.001:
                    y_right = (rho - w_img * a) / b if abs(b) > 0.001 else float('inf')
                    if 0 <= y_right <= h_img:
                        intersections.append((w_img, int(y_right)))
                
                # Top edge (y = 0)
                if abs(b) > 0.001:
                    x_top = rho / a if abs(a) > 0.001 else float('inf')
                    if 0 <= x_top <= w_img:
                        intersections.append((int(x_top), 0))
                
                # Bottom edge (y = h_img)
                if abs(b) > 0.001:
                    x_bottom = (rho - h_img * b) / a if abs(a) > 0.001 else float('inf')
                    if 0 <= x_bottom <= w_img:
                        intersections.append((int(x_bottom), h_img))
                
                # Remove duplicates
                unique_intersections = list(set(intersections))
                
                if len(unique_intersections) >= 2:
                    p1, p2 = unique_intersections[0], unique_intersections[1]
                    cv2.line(result_img, p1, p2, color, 4)
                    
                    # Add line label
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    
                    angle = np.degrees(theta)
                    edge_type = "H" if abs(angle) < 45 or abs(angle) > 135 else "V"
                    
                    if edge_images:
                        support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edge_images[0][1])
                        label = f'{edge_type}{i+1}: {support_ratio:.2f}'
                    else:
                        label = f'{edge_type}{i+1}'
                    
                    cv2.putText(result_img, label, (mid_x, mid_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
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