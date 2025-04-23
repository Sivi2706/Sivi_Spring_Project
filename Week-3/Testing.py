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
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM pin

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
REVERSE_SPEED = 40        # Speed for reverse when no line detected
REVERSE_DURATION = 0.5    # Seconds to reverse
MIN_CONTOUR_AREA = 800     # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height
TURN_THRESHOLD = 100       # Error threshold for pivoting

# ROI parameters
USE_ROI = True             # Enable ROI for PID error
MIDDLE_ROI_TOP = int(FRAME_HEIGHT * 0.35)     # Start of middle ROI (PID error)
MIDDLE_ROI_BOTTOM = int(FRAME_HEIGHT * 0.65)  # End of middle ROI (PID error)

# Servo parameters
SERVO_FREQ = 50           # Hz
SERVO_CENTER_DUTY = 7.5   # Duty cycle for 90 degrees (1.5 ms pulse)
SERVO_MIN_DUTY = 6.25     # Duty cycle for 45 degrees (~1.25 ms pulse)
SERVO_MAX_DUTY = 8.75     # Duty cycle for 135 degrees (~1.75 ms pulse)
SERVO_SMOOTHING_FACTOR = 0.2  # Smoothing factor for servo (0-1, lower = smoother)
SERVO_ANGLE_THRESHOLD = 2.0  # Minimum angle change to update servo (degrees)
SERVO_ANGLE_DEADZONE = 15.0  # Deadzone around 90 degrees (±15 degrees)

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Servo smoothing variable
last_smoothed_angle = 90.0  # Initial smoothed angle (center)

# Calibration file
CALIBRATION_FILE = "color_calibration.json"

# Default color ranges (HSV format)
default_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),
        ([170, 167, 154], [180, 247, 234])
    ],
    'blue': [
        ([100, 167, 60], [130, 255, 95])
    ],
    'green': [
        ([40, 180, 110], [75, 255, 190])
    ],
    'yellow': [
        ([25, 150, 150], [35, 255, 255])
    ],
    'black': [
        ([0, 0, 0], [179, 100, 75])
    ]
}

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
        print("Run the color_calibration.py script first to create calibrated ranges.")
    
    return default_color_ranges

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
    
    # Motor pins setup
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    
    # Encoder pins setup
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Servo pin setup
    GPIO.setup(ServoMotor, GPIO.OUT)
    
    # Set up encoder interrupts
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    # Set up PWM for motors
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    
    # Set up PWM for servo
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(SERVO_CENTER_DUTY)  # Start at center position
    
    return right_pwm, left_pwm, servo_pwm

# Motor control functions
def pivot_turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)    # Right forward
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

def move_backward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(REVERSE_SPEED)
    left_pwm.ChangeDutyCycle(REVERSE_SPEED)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Servo control function with smoothing and deadzone
def set_servo_angle(servo_pwm, line_angle):
    global last_smoothed_angle
    # Apply exponential moving average for smoothing
    smoothed_angle = (SERVO_SMOOTHING_FACTOR * line_angle) + ((1 - SERVO_SMOOTHING_FACTOR) * last_smoothed_angle)
    
    # Check if angle is within deadzone (75-105 degrees)
    if 90.0 - SERVO_ANGLE_DEADZONE <= smoothed_angle <= 90.0 + SERVO_ANGLE_DEADZONE:
        smoothed_angle = 90.0  # Center the servo
        duty_cycle = SERVO_CENTER_DUTY
    else:
        # Only update if change exceeds threshold
        if abs(smoothed_angle - last_smoothed_angle) > SERVO_ANGLE_THRESHOLD:
            # Clamp angle to 45-135 degrees
            smoothed_angle = max(45.0, min(135.0, smoothed_angle))
            
            # Map smoothed angle (45-135 degrees) to duty cycle (6.25-8.75%)
            duty_cycle = SERVO_MIN_DUTY + ((smoothed_angle - 45.0) / 90.0) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)
            duty_cycle = max(SERVO_MIN_DUTY, min(SERVO_MAX_DUTY, duty_cycle))
        else:
            return last_smoothed_angle  # No update needed
    
    servo_pwm.ChangeDutyCycle(duty_cycle)
    last_smoothed_angle = smoothed_angle
    return smoothed_angle

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

# Function to calibrate black line detection
def calibrate_black_line(picam2, color_ranges):
    print("\nCalibrating black line detection...")
    print("Place the camera to view the black line and press 'c' to capture and calibrate.")
    print("Press 'q' to skip calibration.")
    
    while True:
        frame = picam2.capture_array()
        
        if USE_ROI:
            middle_roi = frame[MIDDLE_ROI_TOP:MIDDLE_ROI_BOTTOM, 0:FRAME_WIDTH]
            cv2.rectangle(frame, (0, MIDDLE_ROI_TOP), (FRAME_WIDTH, MIDDLE_ROI_BOTTOM), (0, 255, 0), 2)
            roi = middle_roi
        else:
            roi = frame
            
        cv2.putText(frame, "Press 'c' to calibrate black line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyWindow("Calibration")
            return
            
        if key == ord('c'):
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            black_mask = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 100, 80]))
            contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                black_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(black_contour) > MIN_CONTOUR_AREA:
                    mask = np.zeros_like(black_mask)
                    cv2.drawContours(mask, [black_contour], -1, 255, -1)
                    roi_pixels = hsv_roi[mask == 255]
                    if len(roi_pixels) > 0:
                        h_min, s_min, v_min = np.min(roi_pixels, axis=0)
                        h_max, s_max, v_max = np.max(roi_pixels, axis=0)
                        
                        h_min = max(0, h_min - 10)
                        s_min = max(0, s_min - 10)
                        v_min = max(0, v_min - 10)
                        h_max = min(179, h_max + 10)
                        s_max = min(255, s_max + 40)
                        v_max = min(255, v_max + 40)
                        
                        color_ranges['black'] = [([h_min, s_min, v_min], [h_max, s_max, v_max])]
                        
                        print(f"Updated black line HSV range: ({h_min}, {s_min}, {v_min}) to ({h_max}, {s_max}, {v_max})")
                        calibrated_mask = cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                        cv2.imshow("Calibrated Black Line Mask", calibrated_mask)
                        cv2.waitKey(2000)
                        cv2.destroyWindow("Calibrated Black Line Mask")
                        break
                    else:
                        print("Could not find black pixels in the contour. Please try again.")
                else:
                    print("Black line contour too small. Please try again.")
            else:
                print("No black line detected. Please try again.")
                
    cv2.destroyWindow("Calibration")

# Line detection function with whole frame for angle and middle ROI for PID
def detect_line(frame, color_priorities, color_ranges):
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Whole frame for line angle
    full_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Middle ROI for PID error (35%-65% from top)
    middle_roi = frame[MIDDLE_ROI_TOP:MIDDLE_ROI_BOTTOM, 0:FRAME_WIDTH]
    middle_hsv = cv2.cvtColor(middle_roi, cv2.COLOR_BGR2HSV)
    cv2.rectangle(frame, (0, MIDDLE_ROI_TOP), (FRAME_WIDTH, MIDDLE_ROI_BOTTOM), (0, 255, 0), 2)
    
    full_best_contour = None
    full_best_color = None
    full_max_area = 0
    line_angle = 90.0  # Default to center if no line detected
    
    middle_best_contour = None
    middle_best_color = None
    middle_best_cx, middle_best_cy = -1, -1
    middle_max_area = 0
    
    valid_contours = {}
    all_available_colors = []
    
    for color_name in color_priorities:
        color_ranges_for_color = color_ranges.get(color_name, [])
        full_color_mask = np.zeros(full_hsv.shape[:2], dtype=np.uint8)
        middle_color_mask = np.zeros(middle_hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges_for_color:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            full_color_mask = cv2.bitwise_or(full_color_mask, cv2.inRange(full_hsv, lower, upper))
            middle_color_mask = cv2.bitwise_or(middle_color_mask, cv2.inRange(middle_hsv, lower, upper))
        
        kernel = np.ones((5, 5), np.uint8)
        full_color_mask = cv2.morphologyEx(full_color_mask, cv2.MORPH_CLOSE, kernel)
        middle_color_mask = cv2.morphologyEx(middle_color_mask, cv2.MORPH_CLOSE, kernel)
        
        if color_name == 'black':
            cv2.imshow(f"{color_name} Full Frame Mask", full_color_mask)
            cv2.imshow(f"{color_name} Middle Mask", middle_color_mask)
        
        full_contours, _ = cv2.findContours(full_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        middle_contours, _ = cv2.findContours(middle_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        full_valid = [cnt for cnt in full_contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        middle_valid = [cnt for cnt in middle_contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        
        if full_valid or middle_valid:
            valid_contours[color_name] = {'full': full_valid, 'middle': middle_valid}
            all_available_colors.append(color_name)
    
    for color_name in color_priorities:
        if color_name in valid_contours:
            # Full frame for line angle
            if valid_contours[color_name]['full']:
                largest_full_contour = max(valid_contours[color_name]['full'], key=cv2.contourArea)
                area = cv2.contourArea(largest_full_contour)
                if area > full_max_area:
                    full_max_area = area
                    full_best_contour = largest_full_contour
                    full_best_color = color_name
            
            # Middle ROI for PID error
            if valid_contours[color_name]['middle']:
                largest_middle_contour = max(valid_contours[color_name]['middle'], key=cv2.contourArea)
                area = cv2.contourArea(largest_middle_contour)
                if area > middle_max_area:
                    middle_max_area = area
                    middle_best_contour = largest_middle_contour
                    middle_best_color = color_name
    
    error = 0
    line_found = False
    metadata = {}
    
    # Process middle ROI for PID error
    if middle_best_contour is not None:
        M = cv2.moments(middle_best_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            middle_best_cx = cx
            middle_best_cy = cy + MIDDLE_ROI_TOP
            
            contour_color = (0, 255, 0) if middle_best_color != 'black' else (128, 128, 128)
            cv2.drawContours(frame[MIDDLE_ROI_TOP:MIDDLE_ROI_BOTTOM, 0:FRAME_WIDTH], 
                            [middle_best_contour], -1, contour_color, 2)
            cv2.circle(frame, (middle_best_cx, middle_best_cy), 5, (255, 0, 0), -1)
            cv2.line(frame, (center_x, middle_best_cy), (middle_best_cx, middle_best_cy), (255, 0, 0), 2)
            
            error = middle_best_cx - center_x
            line_found = True
            
            hsv_value = middle_hsv[cy, cx]
            h, s, v = hsv_value
            metadata['middle_color'] = middle_best_color
            metadata['error'] = error
            metadata['hsv'] = (int(h), int(s), int(v))
    
    # Process full frame for line angle
    if full_best_contour is not None:
        # Fit a line to the contour
        [vx, vy, x0, y0] = cv2.fitLine(full_best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        line_angle = np.arctan2(vy, vx) * 180 / np.pi
        line_angle = line_angle.item() % 180  # Extract scalar and normalize to 0-180 degrees
        
        contour_color = (0, 255, 0) if full_best_color != 'black' else (128, 128, 128)
        cv2.drawContours(frame, [full_best_contour], -1, contour_color, 2)
        
        # Draw fitted line
        left_y = int(y0 - (x0 * vy / vx))
        right_y = int(y0 + ((FRAME_WIDTH - x0) * vy / vx))
        cv2.line(frame, (0, left_y), (FRAME_WIDTH, right_y), (0, 0, 255), 2)
        
        metadata['line_color'] = full_best_color
        metadata['line_angle'] = round(line_angle, 2)
    
    # Display metadata
    y_offset = 30
    for key, value in metadata.items():
        cv2.putText(frame, f"{key}: {value}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
    
    available_colors_text = "Available: " + ", ".join(all_available_colors)
    cv2.putText(frame, available_colors_text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return error, line_found, middle_best_color, all_available_colors, line_angle

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    color_ranges = load_color_calibration()
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        stop_motors(right_pwm, left_pwm)
        servo_pwm.stop()
        right_pwm.stop()
        left_pwm.stop()
        GPIO.cleanup()
        return
    
    calibrate_black_line(picam2, color_ranges)
    
    print("Line follower with whole frame for angle and middle ROI for PID started. Press 'q' or Ctrl+C to stop.")
    
    was_line_lost = False  # Flag to trigger reverse only once per line loss
    
    try:
        while True:
            frame = picam2.capture_array()
            error, line_found, detected_color, available_colors, line_angle = detect_line(frame, color_priorities, color_ranges)
            
            cv2.imshow("Line Follower", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                calibrate_black_line(picam2, color_ranges)
            
            # Control motors based on middle ROI PID error
            if line_found:
                was_line_lost = False
                if error > TURN_THRESHOLD:
                    pivot_turn_right(right_pwm, left_pwm)
                    print(f"Pivot Turning Right - {detected_color} line, Error: {error}")
                elif error < -TURN_THRESHOLD:
                    pivot_turn_left(right_pwm, left_pwm)
                    print(f"Pivot Turning Left - {detected_color} line, Error: {error}")
                else:
                    move_forward(right_pwm, left_pwm)
                    print(f"Moving Forward - {detected_color} line, Error: {error}, Available: {', '.join(available_colors)}")
            else:
                if not was_line_lost:
                    print("No line detected in middle ROI. Reversing...")
                    move_backward(right_pwm, left_pwm)
                    time.sleep(REVERSE_DURATION)
                    stop_motors(right_pwm, left_pwm)
                    was_line_lost = True
                    print("Reverse complete. Stopping.")
                else:
                    stop_motors(right_pwm, left_pwm)
                    print("No line detected in middle ROI. Stopped.")
            
            # Control servo based on full frame line angle with smoothing and deadzone
            smoothed_angle = set_servo_angle(servo_pwm, line_angle)
            print(f"Servo Angle Set to {smoothed_angle:.2f} degrees")
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        servo_pwm.stop()
        right_pwm.stop()
        left_pwm.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()