import cv2
import os
import glob
import numpy as np

# Method 1: Using glob (recommended)
image_folder = "."  # current directory, or specify path like "path/to/images"
jpg_files = glob.glob(os.path.join(image_folder, "*.jpg"))

print(jpg_files)
images = []
for jpg_file in jpg_files:
    img = cv2.imread(jpg_file)
    if img is not None:  # check if image loaded successfully
        images.append(img)
        print(f"Loaded: {jpg_file}")

print(f"Total images loaded: {len(images)}")

for image in images:
    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 150])    # Lower HSV values for white
    upper_white = np.array([180, 30, 255]) # Upper HSV values for white

    # Create white mask - white pixels become 255, others become 0
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    black_mask = cv2.bitwise_not(white_mask)

    kernel = np.ones((5, 5), np.uint8)

    # opened = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    # Use morphological closing to thicken and connect the lines
    closed = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    # Edge detection (not always necessary, but helps Hough sometimes)
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(closed, rho=1, theta=np.pi/180, threshold=100,
                        minLineLength=50, maxLineGap=10)

    line_data = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Approximate thickness by checking the area around the line
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            dx = x2-x1
            if dx == 0:
                continue
            
            slope = (y2-y1)/dx

            length = np.hypot(y2-y1,x2-x1)

            get_length = lambda t: np.hypot(t[2]-t[0], t[3]-t[1])

            if length < width / 5:
                continue

            if max(y2,y1) < 0.5 * height:
                continue

            if min(x1, x2) > width * 0.10 and max(x1, x2) < width * 0.9:
                continue

            for prev_line in line_data:
                if get_length(prev_line[0]) > length and abs(prev_line[2] - np.arctan(slope)) < 0.05:
                    continue

            startx = min(x1, x2)
            sign = lambda h: round(h/abs(h)) if h != 0 else 0
            area = 0

            RANGE = 2

            for t in np.linspace(x1, x2, num=abs(dx), dtype=int):
                f = lambda x: y1 + slope * (x - x1)
                y_center = round(f(t))
                x_center = t

                y1_slice = max(0, y_center - RANGE)
                y2_slice = min(closed.shape[0], y_center + RANGE + 1)
                x1_slice = max(0, x_center - RANGE)
                x2_slice = min(closed.shape[1], x_center + RANGE + 1)
                slice = closed[y1_slice:y2_slice,x1_slice:x2_slice]
                if slice.size != 0:
                    area += slice.mean() / 255
            
            avg_area = area / (dx)
            # print(avg_area)

            value = avg_area * 2

            # print(max(y1,y2) / height)

            value += 3 * (max(y1,y2) / height - 0.5)

            value += 2 * (length / width)

            print(value)

            line_data.append(((x1, y1, x2, y2), value, np.arctan(slope)))

        # Sort by y (vertical position), descending
        print(len(line_data))

        line_data.sort(key=lambda item: item[1], reverse=True)

        # Pick bottommost two lines (ideally from distinct regions)
        bottom_lines = line_data[:1]

        # Draw for visualization
        result = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        for line, _, _ in bottom_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 10)

        cv2.imshow("Detected Bottom Lines", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No lines detected.")


