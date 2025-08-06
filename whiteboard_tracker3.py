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
        Find the whiteboard surface - focus on lower portion and uniform regions
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
    
    def create_edge_detection_regions(self, surface_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create regions for edge detection around the whiteboard surface
        """
        # Create multiple boundary regions with different widths
        kernels = [
            np.ones((5, 5), np.uint8),   # Tight boundary
            np.ones((10, 10), np.uint8), # Medium boundary  
            np.ones((15, 15), np.uint8)  # Wide boundary
        ]
        
        boundary_regions = []
        
        for i, kernel in enumerate(kernels):
            eroded = cv2.erode(surface_mask, kernel, iterations=1)
            dilated = cv2.dilate(surface_mask, kernel, iterations=1)
            boundary = cv2.subtract(dilated, eroded)
            
            pixels = np.sum(boundary > 0)
            print(f"  Boundary region {i+1}: {pixels} pixels")
            boundary_regions.append(boundary)
        
        # Combine all boundary regions
        combined_boundary = boundary_regions[0]
        for boundary in boundary_regions[1:]:
            combined_boundary = cv2.bitwise_or(combined_boundary, boundary)
        
        return combined_boundary, surface_mask
    
    def detect_edges_multiple_methods(self, image: np.ndarray, boundary_region: np.ndarray) -> List[np.ndarray]:
        """
        Try multiple edge detection methods
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_images = []
        
        # Method 1: Standard Canny
        blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
        edges1 = cv2.Canny(blur1, 50, 150)
        masked1 = cv2.bitwise_and(edges1, boundary_region)
        edge_images.append(('Canny_50_150', masked1))
        
        # Method 2: Lower threshold Canny
        edges2 = cv2.Canny(blur1, 30, 100)
        masked2 = cv2.bitwise_and(edges2, boundary_region)
        edge_images.append(('Canny_30_100', masked2))
        
        # Method 3: Bilateral filter + Canny
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        edges3 = cv2.Canny(bilateral, 40, 120)
        masked3 = cv2.bitwise_and(edges3, boundary_region)
        edge_images.append(('Bilateral_Canny', masked3))
        
        # Method 4: Sobel edges
        sobel_x = cv2.Sobel(blur1, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur1, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_thresh = np.percentile(sobel_mag, 90)
        sobel_edges = (sobel_mag > sobel_thresh).astype(np.uint8) * 255
        masked4 = cv2.bitwise_and(sobel_edges, boundary_region)
        edge_images.append(('Sobel', masked4))
        
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
    
    def find_lines_progressive(self, edge_images: List[Tuple[str, np.ndarray]]) -> List[Tuple[float, float]]:
        """
        Try to find lines with special focus on finding both horizontal and vertical edges
        """
        h, w = edge_images[0][1].shape
        all_scored_lines = []
        
        # Try each edge detection method
        for method_name, edges in edge_images:
            print(f"\n  Trying line detection on {method_name}:")
            
            edge_pixel_count = np.sum(edges > 0)
            if edge_pixel_count < 20:
                print(f"    Skipping - too few edge pixels ({edge_pixel_count})")
                continue
            
            print(f"    Edge pixels available: {edge_pixel_count}")
            
            # Try multiple Hough parameter sets - be more aggressive about finding lines
            hough_configs = [
                {'threshold': 50, 'rho': 1, 'theta': np.pi/180},
                {'threshold': 40, 'rho': 1, 'theta': np.pi/180},
                {'threshold': 30, 'rho': 1, 'theta': np.pi/180},
                {'threshold': 25, 'rho': 1, 'theta': np.pi/180},
                {'threshold': 20, 'rho': 1, 'theta': np.pi/180},
            ]
            
            # Also try HoughLinesP (probabilistic) which might catch different lines
            print(f"    Trying HoughLinesP...")
            linesP = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, 
                                    minLineLength=50, maxLineGap=20)
            
            if linesP is not None:
                print(f"    HoughLinesP found {len(linesP)} line segments")
                # Convert line segments to polar form
                for line_seg in linesP:
                    x1, y1, x2, y2 = line_seg[0]
                    
                    # Convert to polar coordinates
                    if x1 == x2:  # Vertical line
                        rho = abs(x1)
                        theta = np.pi / 2
                    elif y1 == y2:  # Horizontal line  
                        rho = abs(y1)
                        theta = 0 if y1 >= 0 else np.pi
                    else:
                        # General case
                        dx = x2 - x1
                        dy = y2 - y1
                        angle = np.arctan2(dy, dx)
                        if angle < 0:
                            angle += np.pi
                        
                        # Calculate rho
                        rho = abs(x1 * np.cos(angle) + y1 * np.sin(angle))
                        theta = angle
                    
                    angle_deg = np.degrees(theta)
                    
                    # Check orientation
                    is_horizontal = (abs(angle_deg) < 20) or (abs(angle_deg - 180) < 20)
                    is_vertical = (70 < angle_deg < 110)
                    
                    if is_horizontal or is_vertical:
                        # Score this line
                        support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                        
                        if support_ratio > 0.05 and edge_hits > 5:  # Lower thresholds
                            edge_type = 'H' if is_horizontal else 'V'
                            all_scored_lines.append((rho, theta, support_ratio, edge_hits, edge_type, 'HoughP'))
                            print(f"      {edge_type} line (HoughP): rho={rho:.1f}, angle={angle_deg:.1f}°, support={support_ratio:.3f}, hits={edge_hits}")
            
            # Try standard Hough Lines
            best_lines_this_method = []
            
            for config in hough_configs:
                lines = cv2.HoughLines(edges, **config)
                
                if lines is not None and len(lines) > 0:
                    print(f"    HoughLines (thresh={config['threshold']}): {len(lines)} raw lines")
                    
                    # Separate horizontal and vertical lines and treat them differently
                    horizontal_candidates = []
                    vertical_candidates = []
                    
                    for line in lines:
                        rho, theta = line[0]
                        angle_deg = np.degrees(theta)
                        
                        # Classify by orientation
                        is_horizontal = (abs(angle_deg) < 20) or (abs(angle_deg - 180) < 20)
                        is_vertical = (70 < angle_deg < 110)
                        
                        if is_horizontal:
                            # Score the line
                            support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                            if support_ratio > 0.05 and edge_hits > 5:  # Lower thresholds
                                horizontal_candidates.append((rho, theta, support_ratio, edge_hits, 'H'))
                                print(f"      H candidate: rho={rho:.1f}, angle={angle_deg:.1f}°, support={support_ratio:.3f}, hits={edge_hits}")
                        
                        elif is_vertical:
                            # Score the line
                            support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                            if support_ratio > 0.05 and edge_hits > 5:  # Lower thresholds
                                vertical_candidates.append((rho, theta, support_ratio, edge_hits, 'V'))
                                print(f"      V candidate: rho={rho:.1f}, angle={angle_deg:.1f}°, support={support_ratio:.3f}, hits={edge_hits}")
                    
                    # Keep best from each orientation
                    if horizontal_candidates:
                        horizontal_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
                        best_lines_this_method.extend(horizontal_candidates[:2])  # Keep top 2 horizontal
                        print(f"    -> Kept {min(2, len(horizontal_candidates))} horizontal lines")
                    
                    if vertical_candidates:
                        vertical_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
                        best_lines_this_method.extend(vertical_candidates[:2])  # Keep top 2 vertical
                        print(f"    -> Kept {min(2, len(vertical_candidates))} vertical lines")
                    
                    if best_lines_this_method:
                        break  # Found good lines, stop trying lower thresholds
            
            if best_lines_this_method:
                all_scored_lines.extend(best_lines_this_method)
                print(f"  Method {method_name} total: {len(best_lines_this_method)} scored lines")
        
        if not all_scored_lines:
            print("  No lines passed scoring criteria")
            return []
        
        # Sort all lines by score and remove similar ones, but ensure we keep both H and V if they exist
        all_scored_lines.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        print(f"\n  All scored lines:")
        for i, (rho, theta, support, hits, edge_type, *source) in enumerate(all_scored_lines):
            angle = np.degrees(theta)
            src = source[0] if source else 'Hough'
            print(f"    {i+1}. {edge_type} line ({src}): rho={rho:.1f}, angle={angle:.1f}°, support={support:.3f}, hits={hits}")
        
        # Smart deduplication - keep best from each orientation
        horizontal_lines = [line for line in all_scored_lines if line[4] == 'H']
        vertical_lines = [line for line in all_scored_lines if line[4] == 'V']
        
        final_lines = []
        
        # Keep best horizontal lines (remove similar ones)
        for rho, theta, support, hits, edge_type, *source in horizontal_lines:
            is_similar = False
            for existing_rho, existing_theta in final_lines:
                if abs(rho - existing_rho) < 20 and abs(theta - existing_theta) < 0.1:
                    is_similar = True
                    break
            if not is_similar:
                final_lines.append((rho, theta))
                if len([l for l in final_lines if np.degrees(l[1]) < 45 or np.degrees(l[1]) > 135]) >= 2:
                    break  # Max 2 horizontal lines
        
        # Keep best vertical lines (remove similar ones)
        for rho, theta, support, hits, edge_type, *source in vertical_lines:
            is_similar = False
            for existing_rho, existing_theta in final_lines:
                if abs(rho - existing_rho) < 20 and abs(theta - existing_theta) < 0.1:
                    is_similar = True
                    break
            if not is_similar:
                final_lines.append((rho, theta))
                if len([l for l in final_lines if 45 <= np.degrees(l[1]) <= 135]) >= 2:
                    break  # Max 2 vertical lines
        
        print(f"\n  Final unique lines: {len(final_lines)}")
        for i, (rho, theta) in enumerate(final_lines):
            angle = np.degrees(theta)
            edge_type = "H" if angle < 45 or angle > 135 else "V"
            print(f"    {i+1}. Final {edge_type} line: rho={rho:.1f}, angle={angle:.1f}°")
        
        return final_lines
    
    def remove_similar_lines(self, lines: List[Tuple[float, float]], 
                           rho_tolerance: float = 20, theta_tolerance: float = 0.1) -> List[Tuple[float, float]]:
        """
        Remove similar lines
        """
        if len(lines) <= 1:
            return lines
        
        unique_lines = []
        
        for rho, theta in lines:
            is_similar = False
            
            for existing_rho, existing_theta in unique_lines:
                rho_diff = abs(rho - existing_rho)
                theta_diff = min(abs(theta - existing_theta), 
                               abs(theta - existing_theta + np.pi),
                               abs(theta - existing_theta - np.pi))
                
                if rho_diff < rho_tolerance and theta_diff < theta_tolerance:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_lines.append((rho, theta))
        
        return unique_lines
    
    def detect_whiteboard_edges(self, image_path: str) -> Optional[List[Tuple[float, float]]]:
        """
        Main detection function with comprehensive debugging
        """
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
            
            surface_area = np.sum(surface_mask > 0)
            if surface_area < 500:
                print(f"  FAIL: Surface area too small ({surface_area} pixels)")
                return None
            
            # Step 2: Create boundary regions
            print("\nStep 2: Creating edge detection regions...")
            boundary_region, _ = self.create_edge_detection_regions(surface_mask)
            
            boundary_pixels = np.sum(boundary_region > 0)
            if boundary_pixels < 100:
                print(f"  FAIL: Boundary region too small ({boundary_pixels} pixels)")
                return None
            
            # Step 3: Detect edges with multiple methods
            print("\nStep 3: Detecting edges...")
            edge_images = self.detect_edges_multiple_methods(image, boundary_region)
            
            # Step 4: Find lines
            print("\nStep 4: Finding lines...")
            detected_lines = self.find_lines_progressive(edge_images)
            
            if not detected_lines:
                print("  FAIL: No lines detected")
                if self.debug:
                    self.debug_visualize(image, surface_mask, edge_images, [], os.path.basename(image_path))
                return None
            
            # Limit to reasonable number
            if len(detected_lines) > 4:
                print(f"  Limiting from {len(detected_lines)} to 4 lines")
                detected_lines = detected_lines[:4]
            
            print(f"\nSUCCESS: Found {len(detected_lines)} edge line(s)")
            for i, (rho, theta) in enumerate(detected_lines):
                angle = np.degrees(theta)
                edge_type = "horizontal" if abs(angle) < 45 else "vertical"
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
        Debug visualization with line validation overlay
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
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
                axes[0, 2].imshow(best_edges, cmap='gray')
                axes[0, 2].set_title(f'Edges: {edge_images[0][0]}')
            else:
                axes[0, 2].imshow(np.zeros_like(surface_mask), cmap='gray')
                axes[0, 2].set_title('No Edges')
            axes[0, 2].axis('off')
            
            # Edge validation - show which pixels each line passes through
            if edge_images and lines:
                validation_img = cv2.cvtColor(best_edges, cv2.COLOR_GRAY2BGR)
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
                
                for i, (rho, theta) in enumerate(lines):
                    color = colors[i % len(colors)]
                    h, w = best_edges.shape
                    
                    # Draw line path and mark edge intersections
                    a = np.cos(theta)
                    b = np.sin(theta)
                    
                    # Sample and mark points along the line
                    if abs(b) > abs(a):  # More horizontal
                        for x in range(0, w, 1):
                            y = (rho - x * a) / b
                            if 0 <= y < h:
                                y = int(round(y))
                                if best_edges[y, x] > 0:
                                    # Mark edge intersections with colored circles
                                    cv2.circle(validation_img, (x, y), 2, color, -1)
                    else:  # More vertical
                        for y in range(0, h, 1):
                            x = (rho - y * b) / a
                            if 0 <= x < w:
                                x = int(round(x))
                                if best_edges[y, x] > 0:
                                    cv2.circle(validation_img, (x, y), 2, color, -1)
                
                axes[1, 0].imshow(validation_img)
                axes[1, 0].set_title('Line-Edge Intersections')
            else:
                axes[1, 0].imshow(np.zeros_like(surface_mask), cmap='gray')
                axes[1, 0].set_title('No Line Validation')
            axes[1, 0].axis('off')
            
            # Original edges without lines for comparison
            if edge_images:
                axes[1, 1].imshow(best_edges, cmap='gray')
                axes[1, 1].set_title('Pure Edge Detection')
            else:
                axes[1, 1].imshow(np.zeros_like(surface_mask), cmap='gray')
                axes[1, 1].set_title('No Edges')
            axes[1, 1].axis('off')
            
            # Result with lines - fix the line drawing
            result_img = image.copy()
            colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            
            for i, (rho, theta) in enumerate(lines):
                color = colors_bgr[i % len(colors_bgr)]
                
                # Improved line drawing - find actual intersections with image boundaries
                a = np.cos(theta)
                b = np.sin(theta)
                
                # Find all possible intersections with image boundaries
                h_img, w_img = image.shape[:2]
                intersections = []
                
                # Check intersection with left edge (x = 0)
                if abs(a) > 0.001:
                    y_left = rho / b if abs(b) > 0.001 else float('inf')
                    if 0 <= y_left <= h_img:
                        intersections.append((0, int(y_left)))
                
                # Check intersection with right edge (x = w_img)
                if abs(a) > 0.001:
                    y_right = (rho - w_img * a) / b if abs(b) > 0.001 else float('inf')
                    if 0 <= y_right <= h_img:
                        intersections.append((w_img, int(y_right)))
                
                # Check intersection with top edge (y = 0)
                if abs(b) > 0.001:
                    x_top = rho / a if abs(a) > 0.001 else float('inf')
                    if 0 <= x_top <= w_img:
                        intersections.append((int(x_top), 0))
                
                # Check intersection with bottom edge (y = h_img)
                if abs(b) > 0.001:
                    x_bottom = (rho - h_img * b) / a if abs(a) > 0.001 else float('inf')
                    if 0 <= x_bottom <= w_img:
                        intersections.append((int(x_bottom), h_img))
                
                # Remove duplicates and sort
                unique_intersections = list(set(intersections))
                
                if len(unique_intersections) >= 2:
                    # Use first two intersections
                    p1, p2 = unique_intersections[0], unique_intersections[1]
                    cv2.line(result_img, p1, p2, color, 4)
                    
                    # Add validation info at midpoint
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    
                    if edge_images:
                        support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edge_images[0][1])
                        angle = np.degrees(theta)
                        edge_type = "H" if abs(angle) < 45 or abs(angle) > 135 else "V"
                        
                        label = f'{edge_type}{i+1}: {support_ratio:.2f}'
                        cv2.putText(result_img, label, (mid_x, mid_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    print(f"    Line {i+1}: Drew from {p1} to {p2}")
                else:
                    print(f"    Line {i+1}: Could not find valid intersections")
                    print(f"      rho={rho:.1f}, theta={np.degrees(theta):.1f}°")
                    print(f"      Found intersections: {intersections}")
            
            axes[1, 2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title(f'Result ({len(lines)} lines)')
            axes[1, 2].axis('off')
            
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