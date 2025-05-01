import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import json
import os
from picamera2 import Picamera2
import math
import glob
from collections import deque
from sklearn import svm

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 500     # Reduced to better detect smaller lines
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100       # Error threshold for pivoting

# Recovery parameters
REVERSE_SPEED = 40         # Speed when reversing

# ROI parameters
USE_ROI = True             # Enable ROI for more focused line detection
ROI_HEIGHT = 150           # Height of the ROI from the bottom of the frame

# PWM settings
PWM_FREQ = 1000           # Motor PWM frequency

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Calibration file
CALIBRATION_FILE = "color_calibration.json"

# Image directory
IMG_DIR = "/home/raspberry/Documents/S2V2/Sivi_Spring_Project/Final Week 3/Img recg"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# Template images directory
TEMPLATE_DIR = IMG_DIR
TEMPLATE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg']  # Supported image extensions

# Default color ranges (HSV format) in case calibration file is not found
default_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),    # Lower red range
        ([170, 167, 154], [180, 247, 234])  # Upper red range
    ],
    'blue': [
        ([100, 100, 50], [130, 255, 255])   # Narrowed blue range to match observed HSV (114, 138, 85)
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

# HSV color bounds for arrow detection (blue arrows)
arrow_minHSV = np.array([100, 100, 50])
arrow_maxHSV = np.array([140, 255, 255])

# Function to initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
        picam2.start()
        time.sleep(2)  # Allow camera to warm up
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

# Function to ensure the frame is in BGR format
def ensure_bgr(frame):
    if len(frame.shape) == 2:  # Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # RGBA
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        return frame  # Assume it's already BGR

# Function to load calibrated color ranges
def load_color_calibration():
    """Load calibrated color ranges from file or return defaults"""
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

# Function to save calibrated color ranges
def save_color_calibration(color_ranges):
    """Save calibrated color ranges to file"""
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(color_ranges, f, indent=4)
        print(f"Saved calibrated color ranges to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"Error saving calibration file: {e}")

# Function to get user's color priority choice
def get_color_choices():
    print("\nAvailable line colors to follow (priority order):")
    print("r = red (highest priority)")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black (lowest priority)")
    print("q = quit program")
    print("\nEnter colors in priority order (e.g., 'rb' for red then blue)")
    
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
            
        # Remove duplicates while preserving order
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

    # Setup motor and encoder pins
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Set up encoder interrupts
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    # Setup motor PWM
    left_pwm = GPIO.PWM(ENA, PWM_FREQ)  # Left motor
    right_pwm = GPIO.PWM(ENB, PWM_FREQ) # Right motor
    left_pwm.start(0)
    right_pwm.start(0)

    return left_pwm, right_pwm

# Motor control functions
def turn_right(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    print("Turning Right")

def turn_left(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    print("Turning Left")

def move_forward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(BASE_SPEED)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    print("Moving Forward")

def move_backward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(REVERSE_SPEED)
    right_pwm.ChangeDutyCycle(REVERSE_SPEED)
    print("Moving Backward")

def stop_motors(left_pwm, right_pwm):
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    print("Stopped")

# Function to calibrate a specific color and record HSV values
def calibrate_color(picam2, color_ranges, color_name):
    print(f"\nCalibrating {color_name} line detection...")
    print(f"Place the camera to view the {color_name} line and press 'c' to capture and calibrate.")
    print("Press 'q' to skip calibration for this color.")
    
    # Define initial broad HSV ranges for each color (tightened for blue)
    initial_ranges = {
        'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
        'blue': [([100, 100, 50], [130, 255, 255])],  # Narrower initial range for blue
        'green': [([35, 100, 50], [85, 255, 255])],
        'yellow': [([20, 100, 100], [40, 255, 255])],
        'black': [([0, 0, 0], [180, 100, 80])]
    }
    
    while True:
        frame = picam2.capture_array()
        frame = ensure_bgr(frame)  # Ensure frame is in BGR format
        
        # Draw ROI if enabled
        if USE_ROI:
            roi_y_start = FRAME_HEIGHT - ROI_HEIGHT
            cv2.rectangle(frame, (0, roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)
            roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        else:
            roi = frame
            
        # Convert ROI to HSV to get center pixel value
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Calculate center of ROI
        center_x = FRAME_WIDTH // 2
        center_y = ROI_HEIGHT // 2 if USE_ROI else FRAME_HEIGHT // 2
        # Get HSV value at center
        h, s, v = hsv_roi[center_y, center_x]
        
        # Display calibration instructions
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 80), (0, 0, 0), -1)  # Status panel
        cv2.putText(frame, f"Calibrating {color_name.capitalize()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Center HSV: ({h}, {s}, {v})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"Skipped calibration for {color_name}.")
            cv2.destroyWindow("Calibration")
            return False
            
        if key == ord('c'):
            # Create a mask using initial broad range
            color_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            for lower, upper in initial_ranges[color_name]:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv_roi, lower, upper))
            
            # Find contours in the mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                color_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(color_contour) > MIN_CONTOUR_AREA:
                    # Create a mask from the contour
                    mask = np.zeros_like(color_mask)
                    cv2.drawContours(mask, [color_contour], -1, 255, -1)
                    
                    # Record HSV values from pixels in the contour
                    roi_pixels = hsv_roi[mask == 255]
                    if len(roi_pixels) > 0:
                        if color_name == 'red':
                            # Handle red with two ranges
                            lower_red_pixels = roi_pixels[roi_pixels[:, 0] <= 10]
                            upper_red_pixels = roi_pixels[roi_pixels[:, 0] >= 170]
                            
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
                                h_min_u = max(170, np.min(upper_red_pixels[:, 0]) - 5)
                                h_max_u = min(180, np.max(upper_red_pixels[:, 0]) + 5)
                                s_min_u = max(0, np.min(upper_red_pixels[:, 1]) - 20)
                                s_max_u = min(255, np.max(upper_red_pixels[:, 1]) + 20)
                                v_min_u = max(0, np.min(upper_red_pixels[:, 2]) - 20)
                                v_max_u = min(255, np.max(upper_red_pixels[:, 2]) + 20)
                                color_ranges['red'][1] = ([h_min_u, s_min_u, v_min_u], [h_max_u, s_max_u, v_max_u])
                                print(f"Upper red range: [{h_min_u}, {s_min_u}, {v_min_u}] to [{h_max_u}, {s_max_u}, {v_max_u}]")
                        else:
                            # Single range for other colors
                            h_min = max(0, np.min(roi_pixels[:, 0]) - 10)
                            h_max = min(179, np.max(roi_pixels[:, 0]) + 10)
                            s_min = max(0, np.min(roi_pixels[:, 1]) - 20)
                            s_max = min(255, np.max(roi_pixels[:, 1]) + 20)
                            v_min = max(0, np.min(roi_pixels[:, 2]) - 20)
                            v_max = min(255, np.max(roi_pixels[:, 2]) + 20)
                            color_ranges[color_name] = [([h_min, s_min, v_min], [h_max, s_max, v_max])]
                            print(f"{color_name.capitalize()} range: [{h_min}, {s_min}, {v_min}] to [{h_max}, {s_max}, {v_max}]")
                        
                        # Show the calibrated mask for verification
                        calibrated_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                        for lower, upper in color_ranges[color_name]:
                            lower = np.array(lower, dtype=np.uint8)
                            upper = np.array(upper, dtype=np.uint8)
                            calibrated_mask = cv2.bitwise_or(calibrated_mask, cv2.inRange(hsv_roi, lower, upper))
                        
                        cv2.imshow(f"Calibrated {color_name.capitalize()} Line Mask", calibrated_mask)
                        cv2.waitKey(2000)
                        cv2.destroyWindow(f"Calibrated {color_name.capitalize()} Line Mask")
                        cv2.destroyWindow("Calibration")
                        return True
                    else:
                        print(f"No {color_name} pixels found in contour. Try again.")
                else:
                    print(f"{color_name.capitalize()} contour too small. Try again.")
            else:
                print(f"No {color_name} line detected. Try again.")

# Line detection function
def detect_line(frame, color_priorities, color_ranges):
    frame = ensure_bgr(frame)  # Ensure frame is in BGR format
    
    # Apply ROI if enabled
    if USE_ROI:
        roi_y_start = FRAME_HEIGHT - ROI_HEIGHT
        roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate center of ROI for HSV value
    center_x = FRAME_WIDTH // 2
    center_y = ROI_HEIGHT // 2 if USE_ROI else FRAME_HEIGHT // 2
    h, s, v = hsv[center_y, center_x]
    
    # Draw center line and ROI
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
        
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

        if valid:
            valid_contours[color_name] = valid
            all_available_colors.append(color_name)
            debug_mask = cv2.bitwise_or(debug_mask, color_mask)

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

            # Create status panel
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 120), (0, 0, 0), -1)
            text_color = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'black': (128, 128, 128)
            }.get(best_color, (255, 255, 255))
            
            cv2.putText(frame, f"Line: {best_color.capitalize()} (Error: {error})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(frame, f"HSV: ({h}, {s}, {v})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Colors: {', '.join(all_available_colors)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return error, True, best_color, all_available_colors, debug_mask

    # No line detected
    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 120), (0, 0, 0), -1)
    cv2.putText(frame, "No Line Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Center HSV: ({h}, {s}, {v})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return 0, False, None, [], debug_mask

# Shape detection function with improved consistency
def detect_shape(cnt, frame, hsv_frame):
    shape = "unknown"
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
 
    if area < 500 or len(cnt) < 3:
        return shape, None
 
    # Adjust epsilon for better approximation
    epsilon = 0.02 * peri  # Reduced for more precise shape detection
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_vertices = len(approx)
    
    # First check for triangle (simplest shape)
    if num_vertices == 3:
        return "triangle", cnt
        
    # Check for arrow specifically if it's blue
    elif 4 <= num_vertices <= 8:
        # Get color information to confirm if it's blue
        maskHSV = cv2.inRange(hsv_frame, arrow_minHSV, arrow_maxHSV)
        kernel = np.ones((5, 5), np.uint8)
        maskHSV = cv2.morphologyEx(maskHSV, cv2.MORPH_CLOSE, kernel)
        maskHSV = cv2.morphologyEx(maskHSV, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        x, y, w, h = cv2.boundingRect(cnt)
        x, y = max(x - 10, 0), max(y - 10, 0)
        w = min(w + 20, frame.shape[1] - x)
        h = min(h + 20, frame.shape[0] - y)
        arrow_region = maskHSV[y:y + h, x:x + w]
        
        if arrow_region.size > 0:
            blurIm = cv2.GaussianBlur(arrow_region, (9, 9), 0)
            corners = cv2.goodFeaturesToTrack(blurIm, 2, 0.7, 15)

            if corners is not None and len(corners) >= 2:
                corners = np.int0(corners)
                x0, y0 = corners[0].ravel()
                x1, y1 = corners[1].ravel()
                x0, y0 = x0 + x, y0 + y
                x1, y1 = x1 + x, y1 + y

                cv2.circle(frame, (x0, y0), 5, (0, 0, 255), -1)
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)

                am, bm = (x0 + x1) / 2, (y0 + y1) / 2
                cv2.circle(frame, (int(am), int(bm)), 3, (255, 0, 0), -1)

                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
                cv2.line(frame, (int(cx), int(cy)), (int(am), int(bm)), (255, 0, 0), 2)

                angle = math.degrees(math.atan2(bm - cy, am - cx))
                if -45 <= angle < 45:
                    return "arrow (right)", cnt
                elif 45 <= angle < 135:
                    return "arrow (down)", cnt
                elif -180 <= angle <= -135 or 135 <= angle <= 180:
                    return "arrow (left)", cnt
                elif -135 < angle < -45:
                    return "arrow (up)", cnt
    
    # Continue with regular shape detection
    if num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        return shape, cnt
    elif num_vertices == 5:
        return "pentagon", cnt
    elif num_vertices == 6:
        return "hexagon", cnt
    elif num_vertices > 6:
        circularity = (4 * math.pi * area) / (peri * peri)
        shape = "full circle" if circularity > 0.85 else "partial circle"  # Increased threshold for circularity
        return shape, cnt

    return shape, None

# Load and preprocess reference images with ORB features
def load_templates():
    reference_images = {}
    orb = cv2.ORB_create(nfeatures=1000)  # Increased features for better matching
    for ext in TEMPLATE_EXTENSIONS:
        template_paths = glob.glob(os.path.join(TEMPLATE_DIR, ext))
        for path in template_paths:
            img = cv2.imread(path, 0)  # Load as grayscale
            if img is None:
                print(f"Error: Could not load {path}. Skipping.")
                continue
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                filename = os.path.basename(path)
                reference_images[filename] = (keypoints, descriptors, img)
            else:
                print(f"Warning: No features detected in {path}")
    return reference_images, orb

# Train SVM classifier with directional features
def train_svm_classifier():
    # Training data with red ratio, verticality, horizontality
    features = np.array([
        [0.1, 0.9, 0.0],  # Upward arrow: low red, high verticality
        [0.1, 0.9, 0.0],  # Upward arrow
        [0.1, 0.2, 1.0],  # Leftward arrow: low red, high horizontality
        [0.1, 0.2, 1.0],  # Leftward arrow
        [0.8, 0.3, 0.2],  # Stop sign: high red, moderate features
        [0.7, 0.2, 0.3]   # Stop sign
    ])
    labels = np.array([0, 0, 1, 1, 2, 2])  # 0: up arrow, 1: left arrow, 2: stop sign
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(features, labels)
    return clf

# Validate stop sign characteristics
def validate_stop_sign(image, keypoints):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    red_ratio = np.sum(mask) / (mask.size * 255)
    confidence = 0.5 if red_ratio > 0.2 else 0.3
    return confidence

# Validate arrow orientation using PCA
def validate_orientation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            _, (eigen_vecs, _) = cv2.PCACompute2(cnt.reshape(-1, 2), np.array([]))
            angle = np.arctan2(eigen_vecs[0, 1], eigen_vecs[0, 0]) * 180 / np.pi
            if abs(angle) < 45 or abs(angle - 180) < 45:
                return "up", angle
            elif abs(angle - 90) < 45 or abs(angle - 270) < 45:
                return "left" if angle > 0 else "right", angle
    return None, 0.0

# Enhanced ORB feature matching with orientation validation
def perform_template_matching(frame, orb, reference_images, svm_clf, prev_detections, max_len=5):
    frame_rgb = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    frame_keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        print("No descriptors detected in frame.")
        return gray, "None", 0.0, None, None, None, frame_keypoints, None

    matches_dict = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_matches = None
    best_ref_keypoints = None
    supposed_match = None
    supposed_match_count = 0
    supposed_matches_list = []
    supposed_keypoints = None
    match_threshold = 20  # Stricter matching threshold

    for name, (ref_keypoints, ref_descriptors, ref_img) in reference_images.items():
        matches = bf.match(descriptors, ref_descriptors)
        matches_dict[name] = len(matches)
        if len(matches) > supposed_match_count:
            supposed_match = name
            supposed_match_count = len(matches)
            supposed_matches_list = matches
            supposed_keypoints = ref_keypoints
        if len(matches) > match_threshold:
            matches = sorted(matches, key=lambda x: x.distance)
            if not best_match or len(matches) > matches_dict[best_match]:
                best_match = name
                best_matches = matches[:10]
                best_ref_keypoints = ref_keypoints

    confidence = 0.0
    direction = None
    detected_angle = 0.0
    if supposed_match:
        # Directional validation for arrows
        if "arrow" in supposed_match.lower():
            direction, detected_angle = validate_orientation(frame_rgb)
            ref_direction = "left" if "left" in supposed_match.lower() else "up" if "up" in supposed_match.lower() else None
            if direction and ref_direction and direction != ref_direction:
                confidence = 0.0
                print(f"Direction mismatch: Detected {direction}, Expected {ref_direction}")
            else:
                confidence = 0.5
        elif "stop" in supposed_match.lower():
            confidence = validate_stop_sign(frame_rgb, frame_keypoints)

        # SVM classification with directional features
        red_ratio = np.sum(cv2.inRange(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV), 
                                       np.array([0, 120, 70]), np.array([10, 255, 255]))) / (FRAME_WIDTH * FRAME_HEIGHT * 255)
        contours, _ = cv2.findContours(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        verticality = 1.0 if contours and cv2.boundingRect(contours[0])[3] > cv2.boundingRect(contours[0])[2] else 0.2
        horizontality = 1.0 if contours and cv2.boundingRect(contours[0])[2] > cv2.boundingRect(contours[0])[3] else 0.2
        features = np.array([[red_ratio, verticality, horizontality]])
        svm_pred = svm_clf.predict_proba(features)[0]
        if "arrow" in supposed_match.lower():
            confidence = svm_pred[1 if "left" in supposed_match.lower() else 0]
        elif "stop" in supposed_match.lower():
            confidence = max(confidence, svm_pred[2])

    # Stabilize detection with multi-frame consistency
    current_detection = (supposed_match, confidence)
    prev_detections.append(current_detection)
    if len(prev_detections) > max_len:
        prev_detections.popleft()
    valid_detections = [(d, c) for d, c in prev_detections if d is not None]
    if valid_detections:
        detected_name = max(set([d for d, _ in valid_detections]), key=[d for d, _ in valid_detections].count)
        avg_confidence = sum(c for _, c in valid_detections) / len(valid_detections)
        if avg_confidence < 0.6 and len(set([d for d, _ in valid_detections])) > 1:
            detected_name = None
            avg_confidence = 0.0
    else:
        detected_name = None
        avg_confidence = 0.0

    # Generate match image for debug window
    match_img = None
    if supposed_match and supposed_match_count > 0 and frame_keypoints and supposed_keypoints:
        ref_img = reference_images[supposed_match][2]
        try:
            match_img = cv2.drawMatches(gray, frame_keypoints, ref_img, supposed_keypoints, 
                                       supposed_matches_list[:30], None, flags=2)
            cv2.putText(match_img, f"Supposed Match: {supposed_match}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(match_img, f"Matches: {supposed_match_count}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(match_img, f"Confidence: {avg_confidence:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if "arrow" in supposed_match.lower():
                cv2.putText(match_img, f"Direction: {direction or 'Unknown'}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except cv2.error as e:
            print(f"Error in drawMatches: {e}")
            match_img = gray

    return match_img, detected_name or "None", avg_confidence, supposed_match, supposed_match_count, supposed_matches_list, supposed_keypoints, frame_keypoints

# Main function
def main():
    left_pwm, right_pwm = setup_gpio()
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        GPIO.cleanup()
        return
    
    # Load template images and initialize ORB and SVM
    reference_images, orb = load_templates()
    if not reference_images:
        print(f"No template images found in {TEMPLATE_DIR}. Please add template images.")
    else:
        print(f"Loaded template images: {list(reference_images.keys())}")
    
    svm_clf = train_svm_classifier()
    prev_detections = deque()  # For multi-frame consistency
    
    # Load or initialize color ranges
    color_ranges = load_color_calibration()
    
    # Calibrate all colors before starting
    colors_to_calibrate = ['red', 'blue', 'green', 'yellow', 'black']
    print("\nStarting calibration for all colors...")
    for color in colors_to_calibrate:
        success = calibrate_color(picam2, color_ranges, color)
        if success:
            print(f"Calibration for {color} completed.")
        else:
            print(f"Using existing or default range for {color}.")
    
    # Save calibrated ranges
    save_color_calibration(color_ranges)
    
    # Get color priorities
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    print("Line follower started. Press 'q' to quit or 'c' to recalibrate.")
    
    recovery_mode = False
    # Shape persistence variables
    final_shape_text = "-----"
    shape_counter = 0
    matched_contour = None
    frame_count = 0
    
    try:
        while True:
            frame = picam2.capture_array()
            frame = ensure_bgr(frame)  # Ensure frame is in BGR format
            debug_frame = frame.copy()  # Create a copy for the debug window
            
            # Line detection
            error, line_found, detected_color, available_colors, debug_mask = detect_line(frame, color_priorities, color_ranges)
            
            # Shape detection
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            edges = cv2.Canny(thresh, 120, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 500:  # Filter small contours
                    shape, contour = detect_shape(cnt, frame, hsv_frame)
                    # Shape persistence logic
                    shape_text = shape if shape.startswith("arrow") else shape
                    if final_shape_text != shape_text:
                        shape_counter += 1
                    else:
                        shape_counter = 0
                    if shape_counter >= 5:
                        final_shape_text = shape_text
                        matched_contour = contour

                    print(f"Detected shape: {shape}")
                    # Draw all contours in cyan on debug frame
                    cv2.drawContours(debug_frame, [cnt], -1, (0, 255, 255), 2)  # Changed to cyan
                    # Draw matched contour in green if it matches the final shape
                    if matched_contour is not None and np.array_equal(cnt, matched_contour):
                        cv2.drawContours(debug_frame, [cnt], -1, (0, 255, 0), 3)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Draw shape text on main frame
                        cv2.putText(frame, final_shape_text, (cX, cY), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)  # Changed font
                        # Draw shape text on debug frame for all contours
                        cv2.putText(debug_frame, shape_text, (cX, cY), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)  # Changed font
            
            # Perform template matching with ORB
            match_img, match_name, match_confidence, supposed_match, supposed_match_count, supposed_matches_list, supposed_keypoints, frame_keypoints = perform_template_matching(
                frame, orb, reference_images, svm_clf, prev_detections
            )
            
            # Display status information
            status_text = f"Shape: {final_shape_text} | Frame: {frame_count} | Match: {match_name} ({match_confidence:.2f})"
            cv2.putText(frame, status_text, (10, FRAME_HEIGHT - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frames
            cv2.imshow("Line Follower", frame)
            cv2.imshow("Debug Contours", debug_frame)
            # Show debug mask in new window
            debug_mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(debug_mask_bgr, "Color Mask", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Debug Matching", debug_mask_bgr)
            # Show template matching result
            if match_img is not None:
                cv2.imshow("Template Matching", match_img)
            
            # Save debug images every 10 frames
            if frame_count % 10 == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(IMG_DIR, f"main_{timestamp}_{frame_count}.jpg"), frame)
                cv2.imwrite(os.path.join(IMG_DIR, f"debug_{timestamp}_{frame_count}.jpg"), debug_frame)
                cv2.imwrite(os.path.join(IMG_DIR, f"mask_{timestamp}_{frame_count}.jpg"), debug_mask_bgr)
                if match_img is not None:
                    cv2.imwrite(os.path.join(IMG_DIR, f"match_{timestamp}_{frame_count}.jpg"), match_img)
            
            frame_count += 1
            
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
            
            # Line-following motor control
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
    finally:
        stop_motors(left_pwm, right_pwm)
        left_pwm.stop()
        right_pwm.stop()
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()