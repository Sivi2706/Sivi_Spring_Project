import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import json
import os
from picamera2 import Picamera2
import math

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
        
        # Display HSV value at center of ROI
        cv2.putText(frame, f"Center HSV: ({h}, {s}, {v})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Press 'c' to calibrate {color_name} line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
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

    # Display HSV value at center of ROI
    cv2.putText(frame, f"Center HSV: ({h}, {s}, {v})", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    best_contour = None
    best_color = None
    best_cx, best_cy = -1, -1
    max_area = 0
    valid_contours = {}
    all_available_colors = []

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

# Shape detection function
def detect_shape(cnt, frame, hsv_frame):
    shape = "unknown"
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
 
    if area < 500 or len(cnt) < 3:
         return shape, None
 
    epsilon = 0.03 * peri if area > 1000 else 0.05 * peri
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
        shape = "full circle" if circularity > 0.8 else "partial circle"
        return shape, cnt

    return shape, None

# Main function
def main():
    left_pwm, right_pwm = setup_gpio()
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        GPIO.cleanup()
        return
    
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
    
    try:
        while True:
            frame = picam2.capture_array()
            frame = ensure_bgr(frame)  # Ensure frame is in BGR format
            debug_frame = frame.copy()  # Create a copy for the debug window
            
            # Line detection
            error, line_found, detected_color, available_colors = detect_line(frame, color_priorities, color_ranges)
            
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
                    # Draw all contours in red on debug frame
                    cv2.drawContours(debug_frame, [cnt], -1, (0, 0, 255), 2)
                    # Draw matched contour in green if it matches the final shape
                    if matched_contour is not None and np.array_equal(cnt, matched_contour):
                        cv2.drawContours(debug_frame, [cnt], -1, (0, 255, 0), 3)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Draw shape text on main frame
                        cv2.putText(frame, final_shape_text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # Draw shape text on debug frame for all contours
                        cv2.putText(debug_frame, shape_text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display the frames
            cv2.imshow("Line Follower", frame)
            cv2.imshow("Debug Contours", debug_frame)
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