import cv2
import numpy as np
import os
from picamera2 import Picamera2
from collections import deque

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

picam2 = initialize_camera()
if picam2 is None:
    print("Exiting program. Camera could not be initialized.")
    exit()

# Color thresholds for shape detection (HSV ranges)
COLOR_RANGES = {
    'red': ([0, 100, 50], [10, 255, 255]),
    'green': ([35, 100, 100], [85, 255, 255]),
    'blue': ([95, 50, 50], [145, 255, 255]),
    'yellow': ([15, 100, 100], [45, 255, 255])
}

# Blue range for arrow detection
ARROW_BLUE_RANGE = ([95, 50, 50], [145, 255, 255])  # Matches blue in COLOR_RANGES

def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in COLOR_RANGES.items():
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.bitwise_or(mask, color_mask)
    track_mask = cv2.inRange(hsv, np.array([30, 50, 50]), np.array([50, 100, 100]))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(track_mask))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("debug_mask.jpg", mask)
    return mask

def detect_colored_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = preprocess_frame(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    max_area = 0.5 * frame.shape[0] * frame.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > max_area:
            continue
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        roi = hsv[y:y+h, x:x+w]
        for color, (lower, upper) in COLOR_RANGES.items():
            color_mask = cv2.inRange(roi, np.array(lower), np.array(upper))
            if cv2.countNonZero(color_mask) > 0.1 * w * h:
                break
        else:
            color = "unknown"
        sides = len(approx)
        print(f"Shape Contour: Sides = {sides}, Area = {area}")
        if sides == 3:
            shape = "Triangle"
        elif sides == 4:
            aspect_ratio = float(w)/h
            shape = "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"
        elif sides == 5:
            shape = "Pentagon"
        elif sides == 6:
            shape = "Hexagon"
        else:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            print(f"Shape Circularity: {circularity}")
            shape = "Circle" if circularity > 0.5 else "Unknown"
        shapes.append((x, y, w, h, f"{color} {shape}"))
    return shapes

def compute_angle(pt1, pt2, pt3):
    """Compute angle at pt2 between pt1-pt2-pt3 in degrees."""
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def detect_arrow_direction(frame):
    # Convert to HSV and create blue mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 50, 50])
    upper_blue = np.array([145, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Enhanced morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Find the largest blue contour
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500:
        return None
    
    # Create a mask of just the largest blue object
    obj_mask = np.zeros_like(blue_mask)
    cv2.drawContours(obj_mask, [largest_contour], -1, 255, -1)
    
    # Method 1: Moments and orientation
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return None
    
    # Calculate orientation
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    _, _, angle = cv2.fitEllipse(largest_contour)
    
    # Method 2: Convex hull defects (more reliable for arrows)
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    defects = cv2.convexityDefects(largest_contour, hull)
    
    if defects is not None and len(defects) > 2:
        # Find the deepest defect (likely the arrow head)
        defects = defects[:,0,:]
        far_points = [tuple(largest_contour[d[2]][0]) for d in defects]
        start_points = [tuple(largest_contour[d[0]][0]) for d in defects]
        end_points = [tuple(largest_contour[d[1]][0]) for d in defects]
        
        # Find the point with maximum depth
        max_defect_idx = np.argmax(defects[:,3])
        far_point = far_points[max_defect_idx]
        start_point = start_points[max_defect_idx]
        end_point = end_points[max_defect_idx]
        
        # Calculate vectors
        vec1 = np.array(start_point) - np.array(far_point)
        vec2 = np.array(end_point) - np.array(far_point)
        
        # Calculate angle between vectors
        angle = np.arctan2(vec1[1], vec1[0]) - np.arctan2(vec2[1], vec2[0])
        angle = np.abs(angle * 180 / np.pi)
        
        # Only consider sharp angles (arrow heads are typically < 90 degrees)
        if angle < 90:
            # Calculate direction vector (points away from arrow head)
            direction_vec = (np.array(start_point) + np.array(end_point)) / 2 - np.array(far_point)
            
            # Determine primary direction
            if np.abs(direction_vec[0]) > np.abs(direction_vec[1]):
                direction = "right" if direction_vec[0] > 0 else "left"
            else:
                direction = "down" if direction_vec[1] > 0 else "up"
            
            # Draw debug information
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, [largest_contour], -1, (0,255,0), 2)
            cv2.circle(debug_frame, far_point, 7, (0,0,255), -1)
            cv2.circle(debug_frame, start_point, 5, (255,0,0), -1)
            cv2.circle(debug_frame, end_point, 5, (255,0,0), -1)
            cv2.line(debug_frame, start_point, end_point, (0,255,255), 2)
            cv2.line(debug_frame, far_point, tuple((np.array(start_point)+np.array(end_point))//2), (255,0,255), 2)
            cv2.putText(debug_frame, f"Direction: {direction}", (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imwrite("debug_arrow.jpg", debug_frame)
            
            return direction
    
    # Fallback method using orientation if convexity defects didn't work
    angle_rad = angle * np.pi / 180
    direction_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    if np.abs(direction_vec[0]) > np.abs(direction_vec[1]):
        direction = "right" if direction_vec[0] > 0 else "left"
    else:
        direction = "down" if direction_vec[1] > 0 else "up"
    
    return direction

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        print(f"HSV at ({x}, {y}): {hsv[y, x]}")

def main_loop():
    prev_detections = deque(maxlen=5)
    cv2.namedWindow("Track Symbol Detection")
    while True:
        frame = picam2.capture_array()
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        cv2.setMouseCallback("Track Symbol Detection", mouse_callback, frame)
        cv2.imwrite("raw_frame.jpg", frame)
        shapes = detect_colored_shapes(frame)
        arrow_dir = detect_arrow_direction(frame)
        for x, y, w, h, label in shapes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if arrow_dir:
            cv2.putText(frame, f"Arrow: {arrow_dir}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Track Symbol Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main_loop()