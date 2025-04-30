import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import json
import os
from picamera2 import Picamera2
from collections import deque

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
SERVO_PIN = 18            # Servo motor pin
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 500    # Reduced for thinner lines
FRAME_WIDTH = 640         # Camera frame width
FRAME_HEIGHT = 480        # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100      # Error threshold for pivoting

# Recovery parameters
REVERSE_SPEED = 40        # Speed when reversing

# ROI parameters
USE_ROI = True            # Enable ROI for line detection
ROI_HEIGHT = 150          # Height of the ROI from the bottom
SYMBOL_ROI_HEIGHT = 400   # Increased for better symbol detection (upper frame)

# PWM settings
PWM_FREQ = 1000           # Motor PWM frequency
SERVO_FREQ = 50           # Servo PWM frequency
SERVO_NEUTRAL = 7.5       # Neutral position (90 degrees, 7.5% duty cycle)

# Variables to store encoder counts
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
        ([100, 120, 50], [130, 255, 150])   # Adjusted for better blue detection
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

# Initialize Raspberry Pi Camera with exposure settings
def initialize_camera():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        config["controls"]["ExposureTime"] = 20000  # Adjust exposure (in microseconds)
        config["controls"]["AnalogueGain"] = 2.0  # Increase gain for better brightness
        picam2.configure(config)
        picam2.start()
        time.sleep(2)
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

# Ensure frame is in BGR format
def ensure_bgr(frame):
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return frame

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
    else:
        print("No calibration file found. Using default color ranges.")
    return default_color_ranges

# Save calibrated color ranges
def save_color_calibration(color_ranges):
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(color_ranges, f, indent=4)
        print(f"Saved calibrated color ranges to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"Error saving calibration file: {e}")

# Get user's color priority choice
def get_color_choices():
    print("\nAvailable line colors to follow (priority order):")
    print("r = red (highest priority)")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black (lowest priority)")
    print("q = quit program")
    print("\nEnter colors in priority order (e.g., 'rb')")
    
    color_map = {
        'r': 'red',
        'b': 'blue',
        'g': 'green',
        'y': 'yellow',
        'k': 'black'
    }
    
    while True:
        choices = input("\nEnter color priorities (e.g., 'rbk'): ").lower()
        if choices == 'q':
            return None
            
        seen = set()
        unique_choices = []
        for c in choices:
            if c in color_map and c not in seen:
                seen.add(c)
                unique_choices.append(c)
                
        if unique_choices:
            selected_colors = [color_map[c] for c in unique_choices]
            if 'black' not in selected_colors:
                selected_colors.append('black')
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        else:
            print("Invalid choice. Please try again.")

# Encoder callback functions
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

# GPIO Setup
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, SERVO_PIN], GPIO.OUT)
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    left_pwm = GPIO.PWM(ENA, PWM_FREQ)
    right_pwm = GPIO.PWM(ENB, PWM_FREQ)
    left_pwm.start(0)
    right_pwm.start(0)
    
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    servo_pwm.start(SERVO_NEUTRAL)
    
    return left_pwm, right_pwm, servo_pwm

# Motor control functions
def turn_right(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    print("Turning Right")

def turn_left(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    print("Turning Left")

def move_forward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(BASE_SPEED)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    print("Moving Forward")

def move_backward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(REVERSE_SPEED)
    right_pwm.ChangeDutyCycle(REVERSE_SPEED)
    print("Moving Backward")

def stop_motors(left_pwm, right_pwm, servo_pwm):
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    servo_pwm.ChangeDutyCycle(SERVO_NEUTRAL)
    print("Stopped")

# Symbol Detection Functions
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

def match_image(frame, reference_images, orb):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return None
    matches_dict = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best_match = None
    for name, (ref_keypoints, ref_descriptors, ref_img) in reference_images.items():
        matches = bf.knnMatch(descriptors, ref_descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]  # Lowe's ratio test
        matches_dict[name] = len(good_matches)
        if len(good_matches) > 50:
            best_match = name
    for name, count in sorted(matches_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {count} matches")
    return best_match

def detect_shapes(frame, color_ranges):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Color-based segmentation for blue arrow
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = np.zeros_like(gray)
    for lower, upper in color_ranges['blue']:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        blue_mask = cv2.bitwise_or(blue_mask, cv2.inRange(hsv, lower, upper))
    
    # Preprocessing with adjusted parameters
    blurred = cv2.GaussianBlur(blue_mask, (7, 7), 0)  # Increased kernel size
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 20, 150)  # Adjusted thresholds for better edge detection
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Debug intermediate steps
    cv2.imshow("Edges", edges)
    cv2.imshow("Threshold", thresh)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_detected = None
    symbol_mask = np.zeros_like(gray)
    shape_outline_mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Lowered threshold
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        sides = len(approx)
        
        # Debug contour information
        print(f"Contour area: {area}, Circularity: {circularity}, Sides: {sides}")
        
        # Detect enclosing shape (circle or rectangle)
        enclosing_shape = None
        if circularity > 0.6:  # Lowered threshold
            enclosing_shape = "Circle"
        elif sides == 4:
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                enclosing_shape = "Square"
            else:
                enclosing_shape = "Rectangle"
        
        # Arrow detection with direction
        if 4 <= sides <= 10:  # Wider range for arrow-like shapes
            try:
                hull = cv2.convexHull(contour, returnPoints=False)
                if hull is not None and len(hull) > 3 and len(contour) > 3:
                    defects = cv2.convexityDefects(contour, hull)
                    if defects is not None and len(defects) > 0:
                        max_defect = np.max(defects[:, 0, 3])
                        if max_defect > 300:
                            defect_idx = np.argmax(defects[:, 0, 3])
                            start_idx = defects[defect_idx, 0, 0]
                            end_idx = defects[defect_idx, 0, 1]
                            far_idx = defects[defect_idx, 0, 2]
                            far_point = tuple(contour[far_idx][0])
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                dx = far_point[0] - cx
                                dy = far_point[1] - cy
                                angle = np.arctan2(dy, dx) * 180 / np.pi
                                if enclosing_shape == "Circle":
                                    if -45 <= angle <= 45 or 315 <= angle < 360:
                                        shape_detected = "Right"
                                    elif 135 <= angle <= 225:
                                        shape_detected = "Left"
                                    elif 45 < angle < 135:
                                        shape_detected = "Up"
                                    elif 225 < angle < 315:
                                        shape_detected = "Down"
                                elif enclosing_shape == "Rectangle":
                                    if -45 <= angle <= 45 or 315 <= angle < 360:
                                        shape_detected = "Right"
                                    elif 135 <= angle <= 225:
                                        shape_detected = "Left"
                            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                            cv2.drawContours(symbol_mask, [approx], -1, 255, -1)  # Filled mask
                            cv2.drawContours(shape_outline_mask, [approx], -1, 255, 2)  # Outline mask
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(frame, shape_detected, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            break
            except cv2.error as e:
                print(f"Convexity defect calculation skipped: {e}")
    
    return shape_detected, symbol_mask, shape_outline_mask

def detect_images(frame, prev_detections, reference_images, orb, color_ranges, max_len=20):
    if reference_images:
        match_name = match_image(frame, reference_images, orb)
    else:
        match_name = None
    shape_detected, symbol_mask, shape_outline_mask = None, np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if not match_name:
        shape_detected, symbol_mask, shape_outline_mask = detect_shapes(frame, color_ranges)
    current_detection = match_name if match_name else shape_detected
    prev_detections.append(current_detection)
    if len(prev_detections) > max_len:
        prev_detections.popleft()
    valid_detections = [d for d in prev_detections if d is not None]
    detected_name = None
    if valid_detections and valid_detections.count(valid_detections[0]) >= 2:  # Lowered threshold
        detected_name = max(set(valid_detections), key=valid_detections.count)
        label = f"Symbol: {detected_name}"
    else:
        label = "Symbol: None"
    cv2.rectangle(frame, (5, 120), (250, 150), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, detected_name, symbol_mask, shape_outline_mask

# Announce Symbol
def announce_symbol(symbol, last_announced, announce_interval=2.0):
    if symbol and symbol != last_announced[0]:
        current_time = time.time()
        if current_time - last_announced[1] > announce_interval:
            print(f"Symbol Detected: {symbol}")
            last_announced[0] = symbol
            last_announced[1] = current_time
    return last_announced

# Calibrate a specific color
def calibrate_color(picam2, color_ranges, color_name):
    print(f"\nCalibrating {color_name} line detection...")
    print(f"Place the camera to view the {color_name} line and press 'c' to capture and calibrate.")
    print("Press 'q' to skip calibration for this color.")
    
    initial_ranges = {
        'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
        'blue': [([100, 120, 50], [130, 255, 150])],
        'green': [([35, 100, 50], [85, 255, 255])],
        'yellow': [([20, 100, 100], [40, 255, 255])],
        'black': [([0, 0, 0], [180, 100, 80])]
    }
    
    frame_count = 0
    hsv_samples = []
    
    while True:
        frame = picam2.capture_array()
        frame = ensure_bgr(frame)
        
        if USE_ROI:
            roi_y_start = FRAME_HEIGHT - ROI_HEIGHT
            cv2.rectangle(frame, (0, roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)
            roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        else:
            roi = frame
            
        cv2.putText(frame, f"Press 'c' to calibrate {color_name} line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in initial_ranges[color_name]:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv_roi, lower, upper))
        cv2.imshow("Initial Mask", color_mask)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"Skipped calibration for {color_name}.")
            cv2.destroyWindow("Calibration")
            cv2.destroyWindow("Initial Mask")
            return False
            
        if key == ord('c'):
            frame_count += 1
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                color_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(color_contour) > MIN_CONTOUR_AREA:
                    mask = np.zeros_like(color_mask)
                    cv2.drawContours(mask, [color_contour], -1, 255, -1)
                    roi_pixels = hsv_roi[mask == 255]
                    if len(roi_pixels) > 0:
                        hsv_samples.append(roi_pixels)
                    else:
                        print(f"No {color_name} pixels found in contour. Try again.")
                else:
                    print(f"{color_name.capitalize()} contour too small. Try again.")
            else:
                print(f"No {color_name} line detected. Try again.")
                
            if frame_count >= 3:
                if hsv_samples:
                    all_pixels = np.concatenate(hsv_samples, axis=0)
                    if color_name == 'red':
                        lower_red_pixels = all_pixels[all_pixels[:, 0] <= 10]
                        upper_red_pixels = all_pixels[all_pixels[:, 0] >= 160]
                        if len(lower_red_pixels) > 0:
                            h_min_l = max(0, np.min(lower_red_pixels[:, 0]) - 5)
                            h_max_l = min(10, np.max(lower_red_pixels[:, 0]) + 5)
                            s_min_l = max(0, np.min(lower_red_pixels[:, 1]) - 20)
                            s_max_l = min(255, np.max(lower_red_pixels[:, 1]) + 20)
                            v_min_l = max(0, np.min(lower_red_pixels[:, 2]) - 20)
                            v_max_l = min(255, np.max(lower_red_pixels[:, 2]) + 20)
                            color_ranges['red'][0] = ([h_min_l, s_min_l, v_min_l], [h_max_l, s_max_l, v_max_l])
                            print(f"Lower red range: [{h_min_l}, {s_min_l}, {v_min_l}] to [{h_max_l}, {s_max_l}, {v_max_l}]")
                        if len(upper_red_pixels) > 0:
                            h_min_u = max(160, np.min(upper_red_pixels[:, 0]) - 5)
                            h_max_u = min(180, np.max(upper_red_pixels[:, 0]) + 5)
                            s_min_u = max(0, np.min(upper_red_pixels[:, 1]) - 20)
                            s_max_u = min(255, np.max(upper_red_pixels[:, 1]) + 20)
                            v_min_u = max(0, np.min(upper_red_pixels[:, 2]) - 20)
                            v_max_u = min(255, np.max(upper_red_pixels[:, 2]) + 20)
                            color_ranges['red'][1] = ([h_min_u, s_min_u, v_min_u], [h_max_u, s_max_u, v_max_u])
                            print(f"Upper red range: [{h_min_u}, {s_min_u}, {v_min_u}] to [{h_max_u}, {s_max_u}, {v_max_u}]")
                    else:
                        h_min = max(0, np.min(all_pixels[:, 0]) - 10)
                        h_max = min(179, np.max(all_pixels[:, 0]) + 10)
                        s_min = max(0, np.min(all_pixels[:, 1]) - 20)
                        s_max = min(255, np.max(all_pixels[:, 1]) + 20)
                        v_min = max(0, np.min(all_pixels[:, 2]) - 20)
                        v_max = min(255, np.max(all_pixels[:, 2]) + 20)
                        color_ranges[color_name] = [([h_min, s_min, v_min], [h_max, s_max, v_max])]
                        print(f"{color_name.capitalize()} range: [{h_min}, {s_min}, {v_min}] to [{h_max}, {s_max}, {v_max}]")
                    
                    calibrated_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                    for lower, upper in color_ranges[color_name]:
                        lower = np.array(lower, dtype=np.uint8)
                        upper = np.array(upper, dtype=np.uint8)
                        calibrated_mask = cv2.bitwise_or(calibrated_mask, cv2.inRange(hsv_roi, lower, upper))
                    
                    cv2.imshow(f"Calibrated {color_name.capitalize()} Line Mask", calibrated_mask)
                    cv2.waitKey(2000)
                    cv2.destroyWindow(f"Calibrated {color_name.capitalize()} Line Mask")
                    cv2.destroyWindow("Calibration")
                    cv2.destroyWindow("Initial Mask")
                    return True
                else:
                    print(f"No valid {color_name} samples collected. Try again.")
                    frame_count = 0
                    hsv_samples = []

# Line detection function
def detect_line(frame, color_priorities, color_ranges):
    frame = ensure_bgr(frame)
    
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
    debug_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for color_name in color_priorities:
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges.get(color_name, []):
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
        
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        if color_name == color_priorities[0]:
            debug_mask = color_mask
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid:
            valid_contours[color_name] = valid
            all_available_colors.append(color_name)
            for cnt in valid:
                print(f"{color_name} contour area: {cv2.contourArea(cnt)}")

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
            best_cx = cx if not USE_ROI else cx
            best_cy = cy if not USE_ROI else cy + (FRAME_HEIGHT - ROI_HEIGHT)

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

            hsv_value = hsv[cy, cx] if USE_ROI else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[best_cy, best_cx]
            h, s, v = hsv_value
            print(f"Detected {best_color} at HSV: ({h}, {s}, {v})")

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

            return error, True, best_color, all_available_colors, debug_mask

    return 0, False, None, [], debug_mask

# Main function
def main():
    left_pwm, right_pwm, servo_pwm = setup_gpio()
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        GPIO.cleanup()
        return
    
    reference_images, orb = load_reference_images()
    color_ranges = load_color_calibration()
    
    colors_to_calibrate = ['red', 'blue', 'green', 'yellow', 'black']
    print("\nStarting calibration for all colors...")
    for color in colors_to_calibrate:
        success = calibrate_color(picam2, color_ranges, color)
        if success:
            print(f"Calibration for {color} completed.")
        else:
            print(f"Using existing or default range for {color}.")
    
    save_color_calibration(color_ranges)
    
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    print("Line follower started. Press 'q' to quit or 'c' to recalibrate.")
    
    recovery_mode = False
    prev_detections = deque()
    last_announced = [None, 0.0]
    frame_count = 0
    symbol_skip = 1  # Reduced to check every frame
    
    try:
        while True:
            frame = picam2.capture_array()
            frame = ensure_bgr(frame)
            error, line_found, detected_color, available_colors, debug_mask = detect_line(frame, color_priorities, color_ranges)
            
            if frame_count % symbol_skip == 0:
                symbol_roi = frame[0:SYMBOL_ROI_HEIGHT, 0:FRAME_WIDTH]
                output_frame, detected_symbol, symbol_mask, shape_outline_mask = detect_images(symbol_roi, prev_detections, reference_images, orb, color_ranges)
                frame[0:SYMBOL_ROI_HEIGHT, 0:FRAME_WIDTH] = output_frame
                last_announced = announce_symbol(detected_symbol, last_announced)
            else:
                symbol_mask = np.zeros((SYMBOL_ROI_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
                shape_outline_mask = np.zeros((SYMBOL_ROI_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
            frame_count += 1
            
            cv2.imshow("Line Follower", frame)
            debug_mask_colored = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Debug Mask", debug_mask_colored)
            symbol_mask_colored = cv2.cvtColor(symbol_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Symbol Mask", symbol_mask_colored)
            shape_outline_mask_colored = cv2.cvtColor(shape_outline_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Shape Outline Mask", shape_outline_mask_colored)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                for color in colors_to_calibrate:
                    success = calibrate_color(picam2, color_ranges, color)
                    if success:
                        print(f"Recalibration for {color} completed.")
                    else:
                        print(f"Skipped recalibration for {color}.")
                save_color_calibration(color_ranges)
            
            if line_found:
                recovery_mode = False
                if error > TURN_THRESHOLD:
                    turn_right(left_pwm, right_pwm)
                elif error < -TURN_THRESHOLD:
                    turn_left(left_pwm, right_pwm)
                else:
                    move_forward(left_pwm, right_pwm)
            else:
                if not recovery_mode:
                    print("No line detected. Starting recovery...")
                    recovery_mode = True
                move_backward(left_pwm, right_pwm)
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_motors(left_pwm, right_pwm, servo_pwm)
        try:
            left_pwm.stop()
            right_pwm.stop()
            servo_pwm.stop()
        except Exception as pwm_error:
            print(f"PWM cleanup error: {pwm_error}")
        cv2.destroyAllWindows()
        picam2.stop()
        try:
            GPIO.cleanup()
        except Exception as gpio_error:
            print(f"GPIO cleanup error: {gpio_error}")
        print("Resources released")

if __name__ == "__main__":
    main()