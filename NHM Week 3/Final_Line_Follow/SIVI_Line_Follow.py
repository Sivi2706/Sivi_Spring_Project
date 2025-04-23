import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import json
import os
from picamera2 import Picamera2

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
PWM_DUTY_CYCLE = 100      # Duty cycle (max speed)
RUN_DURATION = 0.5        # Seconds
MIN_CONTOUR_AREA = 800     # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100       # Error threshold for pivoting

# ROI parameters
USE_ROI = True             # Enable ROI for more focused line detection
ROI_HEIGHT = 150           # Height of the ROI from the bottom of the frame

# PWM settings
PWM_FREQ = 1000           # Motor PWM frequency
SERVO_FREQ = 50           # Servo PWM frequency
SERVO_NEUTRAL = 7.5       # Neutral position (90 degrees, 7.5% duty cycle)

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

    # Setup motor, encoder, and servo pins
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, SERVO_PIN], GPIO.OUT)
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Set up encoder interrupts for line detection and motor control
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    # Setup motor PWM
    right_pwm = GPIO.PWM(ENA, PWM_FREQ)  # Right motor
    left_pwm = GPIO.PWM(ENB, PWM_FREQ)   # Left motor
    right_pwm.start(0)
    left_pwm.start(0)

    # Setup servo PWM
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    servo_pwm.start(SERVO_NEUTRAL)  # Start at neutral

    return right_pwm, left_pwm, servo_pwm

# Motor control functions
def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Moving Forward")
    time.sleep(RUN_DURATION)
    stop_motors(right_pwm, left_pwm)

def move_backward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Moving Backward")
    time.sleep(RUN_DURATION)
    stop_motors(right_pwm, left_pwm)

def turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Turning Right")
    time.sleep(RUN_DURATION)
    stop_motors(right_pwm, left_pwm)

def turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Turning Left")
    time.sleep(RUN_DURATION)
    stop_motors(right_pwm, left_pwm)

def stop_motors(right_pwm, left_pwm, servo_pwm=None):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    if servo_pwm:
        servo_pwm.ChangeDutyCycle(SERVO_NEUTRAL)
    print("Stopped")

# Function to calibrate a specific color and record HSV values
def calibrate_color(picam2, color_ranges, color_name):
    print(f"\nCalibrating {color_name} line detection...")
    print(f"Place the camera to view the {color_name} line and press 'c' to capture and calibrate.")
    print("Press 'q' to skip calibration for this color.")
    
    # Define initial broad HSV ranges for each color
    initial_ranges = {
        'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
        'blue': [([90, 100, 50], [130, 255, 255])],
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
            
        cv2.putText(frame, f"Press 'c' to calibrate {color_name} line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"Skipped calibration for {color_name}.")
            cv2.destroyWindow("Calibration")
            return False
            
        if key == ord('c'):
            # Convert ROI to HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
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

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
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
    
    try:
        while True:
            frame = picam2.capture_array()
            frame = ensure_bgr(frame)  # Ensure frame is in BGR format
            error, line_found, detected_color, available_colors = detect_line(frame, color_priorities, color_ranges)
            
            cv2.imshow("Line Follower", frame)
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
                    turn_right(right_pwm, left_pwm)
                elif error < -TURN_THRESHOLD:
                    turn_left(right_pwm, left_pwm)
                else:
                    move_forward(right_pwm, left_pwm)
            else:
                if not recovery_mode:
                    print("No line detected. Starting recovery...")
                    recovery_mode = True
                move_backward(right_pwm, left_pwm)
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm, servo_pwm)
        right_pwm.stop()
        left_pwm.stop()
        servo_pwm.stop()
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()