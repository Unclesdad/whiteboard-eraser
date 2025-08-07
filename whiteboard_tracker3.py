import cv2
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class WhiteboardEdgeDetector:
    def __init__(self, debug: bool = True):
        self.debug = debug
        # Pre-calculate reusable kernels
        self.kernel_7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_3x3 = np.ones((3, 3), np.uint8)
        self.kernel_25x25 = np.ones((25, 25), np.uint8)
        
    def find_whiteboard_surface_region(self, image: np.ndarray) -> np.ndarray:
        """Find the whiteboard surface using a focused approach"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        h_half = h // 2
        
        print(f"  Image size: {w}x{h}")
        
        # Focus on lower half where whiteboard surface should be
        lower_half = gray[h_half:, :]
        lower_mean = lower_half.mean()
        print(f"  Lower half intensity: min={lower_half.min()}, max={lower_half.max()}, mean={lower_mean:.1f}")
        
        # Pre-calculate percentiles once
        p60 = np.percentile(lower_half, 60)
        p70 = np.percentile(lower_half, 70)
        p75 = np.percentile(lower_half, 75)
        
        # Strategy 1: Reasonably bright regions
        bright_thresh = max(150, p70)
        bright_mask = ((lower_half > bright_thresh) * 255).astype(np.uint8)
        
        # Strategy 2: Top 40% brightest pixels
        top_mask = ((lower_half > p60) * 255).astype(np.uint8)
        
        # Strategy 3: Otsu thresholding
        _, otsu_mask = cv2.threshold(lower_half, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Choose best mask based on having a large connected component
        best_mask = None
        best_score = 0
        area_threshold = lower_half.size * 0.05  # Pre-calculate 5% coverage
        
        for name, mask in [('bright', bright_mask), ('top60', top_mask), ('otsu', otsu_mask)]:
            # Clean up mask
            clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_7x7)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.kernel_7x7)
            
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
            
            if num_labels > 1:
                # Use vectorized operation to find largest area
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
        """Create focused boundary region for edge detection"""
        eroded = cv2.erode(surface_mask, self.kernel_25x25, iterations=1)
        dilated = cv2.dilate(surface_mask, self.kernel_25x25, iterations=3)
        boundary_region = cv2.subtract(dilated, eroded)
        
        print(f"  Boundary region: {np.count_nonzero(boundary_region)} pixels")
        return boundary_region
    
    def filter_structural_edges(self, edges: np.ndarray, boundary_region: np.ndarray) -> np.ndarray:
        """Filter edges to focus on structural boundaries and ignore content"""
        h, w = edges.shape
        
        # Apply boundary mask
        boundary_edges = cv2.bitwise_and(edges, boundary_region)
        
        edge_count_before = np.count_nonzero(edges)
        edge_count_after = np.count_nonzero(boundary_edges)
        print(f"    Edges before boundary filtering: {edge_count_before}")
        print(f"    Edges after boundary filtering: {edge_count_after}")
        
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(boundary_edges, connectivity=8)
        
        if num_labels <= 1:
            return boundary_edges
        
        filtered_edges = np.zeros_like(boundary_edges)
        structural_components = 0
        
        # Pre-calculate boundary thresholds
        w_025 = w * 0.25
        w_075 = w * 0.75
        h_025 = h * 0.25
        h_075 = h * 0.75
        
        # Vectorized processing of components
        for i in range(1, min(num_labels, 50)):  # Limit max components to process
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 8:  # Early skip for tiny components
                continue
                
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Quick aspect ratio check
            min_dim = max(min(width, height), 1)
            aspect_ratio = max(width, height) / min_dim
            
            if aspect_ratio <= 1.5:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            
            # Check if near boundary or large enough
            near_boundary = (x < w_025 or x + width > w_075 or 
                           y < h_025 or y + height > h_075)
            
            if near_boundary or aspect_ratio > 3 or area > 30:
                filtered_edges[labels == i] = 255
                structural_components += 1
        
        print(f"    Structural components kept: {structural_components}/{min(num_labels-1, 49)}")
        
        # Morphological cleanup
        filtered_edges = cv2.morphologyEx(filtered_edges, cv2.MORPH_CLOSE, self.kernel_3x3)
        
        # Final contour-based filtering
        contours, _ = cv2.findContours(filtered_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return filtered_edges
        
        final_edges = np.zeros_like(filtered_edges)
        linear_contours = 0
        
        for contour in contours[:20]:  # Limit contours to process
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 10:  # Skip tiny contours
                continue
                
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) <= 6:
                cv2.drawContours(final_edges, [contour], -1, 255, -1)
                linear_contours += 1
            else:
                area = cv2.contourArea(contour)
                if area > 50:
                    cv2.drawContours(final_edges, [contour], -1, 255, -1)
                    linear_contours += 1
        
        print(f"    Linear contours kept: {linear_contours}/{min(len(contours), 20)}")
        print(f"    Final filtered edges: {np.count_nonzero(final_edges)}")
        
        return final_edges
    
    def detect_clean_edges(self, image: np.ndarray, boundary_region: np.ndarray) -> List[np.ndarray]:
        """Detect edges with enhanced filtering"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pre-compute blurred versions once
        blur5 = cv2.GaussianBlur(gray, (5, 5), 1)
        bilateral = cv2.bilateralFilter(gray, 11, 100, 100)
        
        edge_results = []
        
        # Method 1: Standard Canny
        edges1 = cv2.Canny(blur5, 50, 150, apertureSize=3)
        filtered1 = self.filter_structural_edges(edges1, boundary_region)
        edge_results.append(('Canny_50_150_filtered', filtered1))
        
        # Method 2: Lower threshold Canny
        edges2 = cv2.Canny(blur5, 30, 100, apertureSize=3)
        filtered2 = self.filter_structural_edges(edges2, boundary_region)
        edge_results.append(('Canny_30_100_filtered', filtered2))
        
        # Method 3: Bilateral + Canny
        edges3 = cv2.Canny(bilateral, 40, 120, apertureSize=3)
        filtered3 = self.filter_structural_edges(edges3, boundary_region)
        edge_results.append(('Bilateral_Canny_filtered', filtered3))
        
        # Method 4: Sobel with high threshold
        sobel_x = cv2.Sobel(blur5, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur5, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.hypot(sobel_x, sobel_y)  # Faster than sqrt(x^2 + y^2)
        sobel_thresh = np.percentile(sobel_mag, 94)
        sobel_edges = ((sobel_mag > sobel_thresh) * 255).astype(np.uint8)
        filtered4 = self.filter_structural_edges(sobel_edges, boundary_region)
        edge_results.append(('Sobel_filtered', filtered4))
        
        # Report edge pixel counts
        for name, edges in edge_results:
            print(f"  {name}: {np.count_nonzero(edges)} edge pixels")
        
        return edge_results
    
    def score_line_against_edges(self, rho: float, theta: float, edges: np.ndarray) -> Tuple[float, int]:
        """Score how well a line matches the actual edge pixels"""
        h, w = edges.shape
        a = np.cos(theta)
        b = np.sin(theta)
        
        edge_hits = 0
        total_samples = 0
        
        # Create neighborhood check mask for efficiency
        neighborhood = np.array([-1, 0, 1])
        
        if abs(b) > abs(a):  # More horizontal line
            # Vectorized x sampling
            x_range = np.arange(0, w, 1)
            y_values = (rho - x_range * a) / b
            
            # Filter valid y values
            valid_mask = (y_values >= 0) & (y_values < h)
            valid_x = x_range[valid_mask]
            valid_y = y_values[valid_mask].astype(int)
            
            total_samples = len(valid_x)
            
            # Check neighborhoods efficiently
            for x, y in zip(valid_x, valid_y):
                y_check = np.clip(y + neighborhood, 0, h-1)
                if np.any(edges[y_check, x] > 0):
                    edge_hits += 1
        else:  # More vertical line
            # Vectorized y sampling
            y_range = np.arange(0, h, 1)
            x_values = (rho - y_range * b) / a
            
            # Filter valid x values
            valid_mask = (x_values >= 0) & (x_values < w)
            valid_y = y_range[valid_mask]
            valid_x = x_values[valid_mask].astype(int)
            
            total_samples = len(valid_y)
            
            # Check neighborhoods efficiently
            for x, y in zip(valid_x, valid_y):
                x_check = np.clip(x + neighborhood, 0, w-1)
                if np.any(edges[y, x_check] > 0):
                    edge_hits += 1
        
        if total_samples == 0:
            return 0.0, 0
        
        return edge_hits / total_samples, edge_hits
    
    def deduplicate_lines(self, lines: List[Tuple[float, float, float, int, str]]) -> List[Tuple[float, float, float, int, str]]:
        """Remove duplicate lines early in the pipeline"""
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
                    if support > existing[2]:  # Better support
                        unique_lines[i] = line
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        print(f"    Deduplication: {len(lines)} → {len(unique_lines)} lines")
        return unique_lines
    
    def validate_whiteboard_geometry(self, lines: List[Tuple[float, float, float, int, str]], 
                                   image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, int, str]]:
        """Sequential selection: Find best line first, then find second line with opposite slope sign"""
        if len(lines) <= 1:
            return lines
        
        print(f"    Sequential selection for {len(lines)} lines:")
        
        # Limit to top 10 lines
        if len(lines) > 10:
            print(f"    WARNING: Too many lines ({len(lines)}), limiting to top 10")
            lines = lines[:10]
        
        def get_slope_sign(theta):
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
        
        # Select best line
        best_line = sorted_lines[0]
        best_theta = best_line[1]
        best_slope_sign = get_slope_sign(best_theta)
        
        selected_lines = [best_line]
        print(f"    SELECTED Line 1: slope_sign={best_slope_sign}, support={best_line[2]:.3f}")
        
        # Find best opposite slope line
        for candidate in sorted_lines[1:]:
            candidate_theta = candidate[1]
            candidate_slope_sign = get_slope_sign(candidate_theta)
            
            # Check similarity
            rho_diff = abs(best_line[0] - candidate[0])
            angle_diff = abs(np.degrees(best_theta) - np.degrees(candidate_theta))
            
            if rho_diff < 50 and angle_diff < 10:
                continue
            
            # Check for opposite slope
            opposite_pairs = [
                ('positive', 'negative'), ('negative', 'positive'),
                ('horizontal', 'positive'), ('horizontal', 'negative'),
                ('positive', 'horizontal'), ('negative', 'horizontal'),
                ('vertical', 'positive'), ('vertical', 'negative'),
                ('positive', 'vertical'), ('negative', 'vertical')
            ]
            
            if (best_slope_sign, candidate_slope_sign) in opposite_pairs:
                selected_lines.append(candidate)
                print(f"    SELECTED Line 2: slope_sign={candidate_slope_sign}, support={candidate[2]:.3f}")
                break
        
        print(f"    Final result: {len(selected_lines)} line(s)")
        return selected_lines
    
    def find_best_lines(self, edge_images: List[Tuple[str, np.ndarray]]) -> List[Tuple[float, float]]:
        """Find the best whiteboard edge lines"""
        h, w = edge_images[0][1].shape
        all_scored_lines = []
        
        # Hough configs
        hough_configs = [
            {'threshold': 100, 'rho': 1, 'theta': np.pi/180},
            {'threshold': 80, 'rho': 1, 'theta': np.pi/180},
            {'threshold': 60, 'rho': 1, 'theta': np.pi/180},
            {'threshold': 40, 'rho': 1, 'theta': np.pi/180},
        ]
        
        # Process each edge method
        for method_name, edges in edge_images:
            print(f"\n  Analyzing {method_name}:")
            
            edge_pixel_count = np.count_nonzero(edges)
            if edge_pixel_count < 20:
                print(f"    Skipping - too few edge pixels ({edge_pixel_count})")
                continue
            
            method_lines = []
            
            for config in hough_configs:
                lines = cv2.HoughLines(edges, **config)
                
                if lines is not None and len(lines) > 0:
                    print(f"    HoughLines (thresh={config['threshold']}): {len(lines)} raw lines")
                    
                    # Limit to top 20 lines
                    lines = lines[:20]
                    
                    for line in lines:
                        rho, theta = line[0]
                        angle_deg = np.degrees(theta)
                        
                        # Quick angle classification
                        is_valid = (abs(angle_deg) < 15 or abs(angle_deg - 180) < 15 or
                                  (75 < angle_deg < 105) or
                                  (15 <= abs(angle_deg) <= 75) or (105 <= abs(angle_deg) <= 165))
                        
                        if is_valid:
                            support_ratio, edge_hits = self.score_line_against_edges(rho, theta, edges)
                            
                            if support_ratio > 0.05 and edge_hits > 10:
                                edge_type = 'H' if abs(angle_deg) < 15 or abs(angle_deg - 180) < 15 else \
                                           'V' if 75 < angle_deg < 105 else 'D'
                                
                                method_lines.append((rho, theta, support_ratio, edge_hits, edge_type))
                    
                    if len(method_lines) >= 2:
                        break
            
            all_scored_lines.extend(method_lines)
            
            if len(all_scored_lines) >= 2:
                break
        
        # Deduplicate and validate
        if all_scored_lines:
            all_scored_lines = self.deduplicate_lines(all_scored_lines)
            
            if len(all_scored_lines) > 1:
                all_scored_lines = self.validate_whiteboard_geometry(all_scored_lines, (h, w))
        
        # Final safety check
        if len(all_scored_lines) > 2:
            all_scored_lines = all_scored_lines[:2]
        
        # Sort by quality
        all_scored_lines.sort(key=lambda x: (x[2], x[3]), reverse=True)
        
        print(f"\n  Final validated lines:")
        for i, (rho, theta, support, hits, edge_type) in enumerate(all_scored_lines):
            print(f"    {i+1}. {edge_type}: rho={rho:.1f}, angle={np.degrees(theta):.1f}°, support={support:.3f}")
        
        return [(rho, theta) for rho, theta, _, _, _ in all_scored_lines]
    
    def detect_whiteboard_edges(self, image_path: str) -> Optional[List[Tuple[float, float]]]:
        """Main detection function"""
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
            
            if np.count_nonzero(surface_mask) < 500:
                print(f"  FAIL: Surface area too small")
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
        """Clean debug visualization"""
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
                axes[1, 0].imshow(edge_images[0][1], cmap='gray')
                axes[1, 0].set_title(f'Edges: {edge_images[0][0]}')
            else:
                axes[1, 0].imshow(np.zeros_like(surface_mask), cmap='gray')
                axes[1, 0].set_title('No Edges')
            axes[1, 0].axis('off')
            
            # Result with lines
            result_img = image.copy()
            colors_bgr = [(0, 0, 255), (0, 255, 0)]
            
            h_img, w_img = image.shape[:2]
            
            for i, (rho, theta) in enumerate(lines):
                color = colors_bgr[i % len(colors_bgr)]
                a = np.cos(theta)
                b = np.sin(theta)
                
                # Find line intersections with image boundaries
                pts = []
                
                # Intersection calculations
                if abs(a) > 0.001:
                    y_left = rho / b if abs(b) > 0.001 else -1
                    if 0 <= y_left <= h_img:
                        pts.append((0, int(y_left)))
                    
                    y_right = (rho - w_img * a) / b if abs(b) > 0.001 else -1
                    if 0 <= y_right <= h_img:
                        pts.append((w_img, int(y_right)))
                
                if abs(b) > 0.001:
                    x_top = rho / a if abs(a) > 0.001 else -1
                    if 0 <= x_top <= w_img:
                        pts.append((int(x_top), 0))
                    
                    x_bottom = (rho - h_img * b) / a if abs(a) > 0.001 else -1
                    if 0 <= x_bottom <= w_img:
                        pts.append((int(x_bottom), h_img))
                
                # Draw line if we have valid intersections
                pts = list(set(pts))  # Remove duplicates
                if len(pts) >= 2:
                    cv2.line(result_img, pts[0], pts[1], color, 4)
                    
                    # Add label
                    mid_x = (pts[0][0] + pts[1][0]) // 2
                    mid_y = (pts[0][1] + pts[1][1]) // 2
                    label = f'E{i+1}'
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
    
    # Find images efficiently
    image_files = []
    for ext in ['jpg', 'jpeg', 'JPG', 'JPEG']:
        image_files.extend(glob.glob(f'*.{ext}'))
    
    if not image_files:
        print("No JPG images found in current directory")
        return {}
    
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