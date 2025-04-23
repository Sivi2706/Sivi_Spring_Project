import cv2
import numpy as np
import os
from picamera2 import Picamera2
from collections import deque
import RPi.GPIO as GPIO
import time
import json

# GPIO Pins for Motor Control
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 800     # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height
TURN_THRESHOLD = 100       # Error threshold for pivoting
REVERSE_SPEED = 40         # Speed when reversing
USE_ROI = True             # Enable ROI for line detection
ROI_HEIGHT = 150           # Height of the ROI from the bottom

# Variables for encoder counts
right_counter = 0
left_counter = 0

# Calibration file
CALIBRATION_FILE = "color_calibration.json"

# Default color ranges (HSV format)
default_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),    # Lower red range
        ([170, 167, 154], [180, 247, 234])  # Upper red range
    ],
    'blue': [
        ([100, 167, 60], [130, 255, 95])    # Blue range
    ],
    'green': [
        ([40, 180, 110], [75, 255, 190])    # Green range
    ],
    'yellow': [
        ([25, 150, 150], [35, 255, 255])    # Yellow range
    ],
    'black': [
        ([0, 0, 0], [179, 100, 75])         # Black range
    ]
}

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Allow camera to warm up
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

# GPIO Setup
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    right_pwm = GPIO.PWM(ENA, 1000)
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    return right_pwm, left_pwm

# Encoder callbacks
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

# Motor control functions
def pivot_turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    left_pwm.ChangeDutyCycle(BASE_SPEED)

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Load calibrated color ranges
def load_color_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                loaded_ranges = json.load(f)
                print("Loaded calibrated color ranges from file.")
                return loaded_ranges
        except Exception as e:
            print(f"Error loading calibration file: {e}")
    print("Using default color ranges.")
    return default_color_ranges

# Get user's color priority choice
def get_color_choices():
    print("\nAvailable line colors to follow (priority order):")
    print("r = red (highest priority)")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black (lowest priority)")
    print("q = quit program")
    
    color_map = {'r': 'red', 'b': 'blue', 'g': 'green', 'y': 'yellow', 'k': 'black'}
    while True:
        choices = input("\nEnter color priorities (e.g., 'rbk'): ").lower()
        if choices == 'q':
            return None
        seen = set()
        unique_choices = [c for c in choices if c in color_map and c not in seen and not seen.add(c)]
        if unique_choices:
            selected_colors = [color_map[c] for c in unique_choices]
            if 'black' not in selected_colors:
                selected_colors.append('black')
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        print("Invalid choice. Please try again.")

# Load and preprocess reference images using ORB
def load_reference_images():
    reference_images = {}
    orb = cv2.ORB_create()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(script_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(script_dir, filename)
            img = cv2.imread(img_path, 0)
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                reference_images[filename] = (keypoints, descriptors, img)
            else:
                print(f"Warning: No features detected in {filename}")
    return reference_images, orb

# ORB Feature Matching with Orientation Detection
# ORB Feature Matching with Orientation Detection
def match_image(frame, reference_images, orb):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return None, None
    matches_dict = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_matches = None
    best_ref_keypoints = None
    
    # Increase this threshold to require more matching features
    min_match_threshold = 50  # Increased from 30
    
    for name, (ref_keypoints, ref_descriptors, ref_img) in reference_images.items():
        matches = bf.match(descriptors, ref_descriptors)
        matches_dict[name] = len(matches)
        # Only consider matches above our higher threshold
        if len(matches) > min_match_threshold:
            matches = sorted(matches, key=lambda x: x.distance)
            # Reject matches with average distance above threshold
            avg_distance = sum(m.distance for m in matches[:20]) / min(20, len(matches))
            if avg_distance > 40:  # Lower values are better matches
                continue
                
            if not best_match or len(matches) > matches_dict[best_match]:
                best_match = name
                best_matches = matches[:10]
                best_ref_keypoints = ref_keypoints
    
    print("\nMatch Results:")
    for name, count in sorted(matches_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {count} matches")
    
    if best_match:
        # Additional verification for homography quality
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([best_ref_keypoints[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Check if we have enough inliers after RANSAC (good matches)
        inlier_count = np.sum(mask) if mask is not None else 0
        if inlier_count < 7:  # Require at least 7 good matches
            return None, None
            
        if M is not None:
            h, w = reference_images[best_match][2].shape
            ref_center = (w // 2, h // 2)
            ref_tip = (w // 2, 0)
            center = cv2.perspectiveTransform(np.array([[ref_center]], dtype=np.float32), M)[0][0]
            tip = cv2.perspectiveTransform(np.array([[ref_tip]], dtype=np.float32), M)[0][0]
            dx = tip[0] - center[0]
            dy = tip[1] - center[1]
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            return best_match, (center, tip, angle)
    return None, None


# Improved Shape Detection Function
def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 30, 200)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_detected = None
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        defect_count = 0
        max_defect = 0
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is not None and len(hull) > 3 and len(contour) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    defect_count = len(defects)
                    max_defect = np.max(defects[:, 0, 3]) if defect_count > 0 else 0
        except cv2.error as e:
            print(f"Convexity defect calculation skipped: {e}")
        print(f"Debug - Sides: {len(approx)}, Circularity: {circularity:.3f}, Defects: {defect_count}, Max Defect: {max_defect}")
        sides = len(approx)
        if sides == 3:
            shape_detected = "Triangle"
        elif sides == 4:
            aspect_ratio = float(w) / h
            shape_detected = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif sides == 5:
            shape_detected = "Pentagon"
        elif sides == 6:
            shape_detected = "Hexagon"
        else:
            if circularity > 0.8:
                shape_detected = "Circle"
            elif 0.5 <= circularity <= 0.8 and defect_count > 0 and max_defect > 300:
                shape_detected = "Pac-Man"
            else:
                shape_detected = "Unknown"
        cv2.putText(frame, shape_detected, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 2)
        break
    return shape_detected

# Line Detection Function
def detect_line(frame, color_priorities, color_ranges):
    if USE_ROI:
        roi_y_start = FRAME_HEIGHT - ROI_HEIGHT
        roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    if USE_ROI:
        cv2.rectangle(frame, (0, roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)
    best_contour = None
    best_color = None
    best_cx, best_cy = -1, -1
    max_area = 0
    valid_contours = {}
    all_available_colors = []
    for color_name in color_priorities:
        color_ranges_for_color = color_ranges.get(color_name, [])
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges_for_color:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        if color_name == 'black':
            cv2.imshow(f"{color_name} Mask", color_mask)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid:
            valid_contours[color_name] = valid
            all_available_colors.append(color_name)
    for color_name in color_priorities:
        if color_name in valid_contours:
            largest_contour = max(valid_contours[color_name], key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > max_area:
                max_area = area
                best_contour = largest_contour
                best_color = color_name
    if best_contour is not None:
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if USE_ROI:
                best_cx = cx
                best_cy = cy + (FRAME_HEIGHT - ROI_HEIGHT)
            else:
                best_cx = cx
                best_cy = cy
            contour_color = (0, 255, 0) if best_color != 'black' else (128, 128, 128)
            if USE_ROI:
                cv2.drawContours(frame[FRAME_HEIGHT-ROI_HEIGHT:FRAME_HEIGHT, 0:FRAME_WIDTH], 
                                [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
            else:
                cv2.drawContours(frame, [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
            cv2.line(frame, (center_x, best_cy), (best_cx, best_cy), (255, 0, 0), 2)
            error = best_cx - center_x
            if USE_ROI:
                hsv_value = hsv[cy, cx]
            else:
                hsv_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[best_cy, best_cx]
            h, s, v = hsv_value
            available_colors_text = "Available: " + ", ".join(all_available_colors)
            cv2.putText(frame, available_colors_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_color = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'black': (128, 128, 128)
            }.get(best_color, (255, 255, 255))
            cv2.putText(frame, f"{best_color.capitalize()} Line, Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(frame, f"HSV: ({h}, {s}, {v})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            return error, True, best_color, all_available_colors
    return 0, False, None, []

# Check if a shape or image is recognizable (not Unknown)
def is_valid_detection(detection):
    return detection is not None and detection != "Unknown"

# Combined Image and Shape Detection with Line Following
# Combined Image and Shape Detection with Line Following
def detect_images_shapes_and_line(frame, prev_detections, reference_images, orb, color_priorities, color_ranges, right_pwm, left_pwm, pause_state, max_len=5):
    # Line detection first
    error, line_found, detected_color, available_colors = detect_line(frame, color_priorities, color_ranges)
    
    # Only perform image/shape detection if a line is found
    match_name = None
    shape_detected = None
    if line_found:
        match_name, orientation = match_image(frame, reference_images, orb)
        if not match_name:
            shape_detected = detect_shapes(frame)
    
    current_detection = match_name if match_name else shape_detected
    prev_detections.append(current_detection)
    if len(prev_detections) > max_len:
        prev_detections.popleft()
    
    valid_detections = [d for d in prev_detections if is_valid_detection(d)]
    detected_name = None
    label = "Detected: None"
    
    if valid_detections:
        detected_name = max(set(valid_detections), key=valid_detections.count)
        label = f"Detected: {detected_name}"
        if match_name and orientation:
            center, tip, angle = orientation
            cv2.arrowedLine(frame, (int(center[0]), int(center[1])), 
                           (int(tip[0]), int(tip[1])), (0, 255, 0), 2, tipLength=0.3)
            label += f" | Angle: {angle:.1f}°"
            print(f"Orientation Detected: {angle:.1f} degrees")
    
    cv2.rectangle(frame, (5, 5), (400, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Check if we need to enter pause state - only for confirmed shapes/images (not Unknown)
    should_pause = False
    if detected_name and not pause_state['active']:
        # Only pause if we have a valid shape or image (not "Unknown")
        if is_valid_detection(detected_name):
            should_pause = True
            pause_state['active'] = True
            pause_state['start_time'] = time.time()
            pause_state['detected_object'] = detected_name
            stop_motors(right_pwm, left_pwm)
            print(f"Confirmed detection: {detected_name}. Pausing for 5 seconds.")
            
            # Display timer on frame
            cv2.rectangle(frame, (5, 45), (400, 85), (0, 0, 0), -1)
            cv2.putText(frame, "Paused: 5.0s remaining", (10, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # If detected as "Unknown", log it but keep moving
            print(f"Ignoring unconfirmed shape detection: {detected_name}")
    
    # Check if we're in pause state
    if pause_state['active']:
        elapsed = time.time() - pause_state['start_time']
        remaining = max(5.0 - elapsed, 0)
        
        # Update display with timer
        cv2.rectangle(frame, (5, 45), (400, 85), (0, 0, 0), -1)
        cv2.putText(frame, f"Paused: {remaining:.1f}s remaining", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Check if pause time is over
        if remaining <= 0:
            pause_state['active'] = False
            print(f"Resume line following after {pause_state['detected_object']} detection")
    
    # Only do line following if not in pause state
    if not pause_state['active']:
        if line_found:
            if error > TURN_THRESHOLD:
                pivot_turn_right(right_pwm, left_pwm)
                print(f"Pivot Turning Right - {detected_color} line")
            elif error < -TURN_THRESHOLD:
                pivot_turn_left(right_pwm, left_pwm)
                print(f"Pivot Turning Left - {detected_color} line")
            else:
                move_forward(right_pwm, left_pwm)
                print(f"Moving Forward - {detected_color} line (Available: {', '.join(available_colors)})")
        else:
            move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            print("Reversing to find line...")
    
    return frame, detected_name, error, line_found, detected_color, available_colors, pause_state


# Main function
def main():
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return
    
    right_pwm, left_pwm = setup_gpio()
    color_ranges = load_color_calibration()
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    reference_images, orb = load_reference_images()
    prev_detections = deque()
    
    # Initialize pause state
    pause_state = {'active': False, 'start_time': 0, 'detected_object': None}
    
    print("Combined image/shape detection and line follower started. Press 'q' to stop.")
    try:
        while True:
            frame = picam2.capture_array()
            
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            output_frame, detected_name, error, line_found, detected_color, available_colors, pause_state = detect_images_shapes_and_line(
                frame, prev_detections, reference_images, orb, color_priorities, color_ranges, right_pwm, left_pwm, pause_state)
            
            cv2.imshow("Combined Detection and Line Follower", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # Small sleep to prevent overwhelming the CPU
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()