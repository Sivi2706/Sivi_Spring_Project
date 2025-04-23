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
PWM_FREQ = 1000           # Motor PWM frequency
SERVO_FREQ = 50           # Servo PWM frequency (50 Hz)
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
REVERSE_SPEED = 40        # Speed when reversing
MIN_CONTOUR_AREA = 800    # Minimum area for valid contours
FRAME_WIDTH = 640         # Camera frame width
FRAME_HEIGHT = 480        # Camera frame height
TURN_THRESHOLD = 100      # Error threshold for pivoting

# ROI parameters
USE_ROI = True            # Enable ROI for line detection
BOTTOM_ROI_HEIGHT = int(FRAME_HEIGHT * 0.30)  # 30% for bottom PID ROI
TOP_ROI_HEIGHT = int(FRAME_HEIGHT * 0.70)     # 70% for top line angle ROI

# Servo parameters
SERVO_CENTER = 7.5        # Duty cycle for 90° (center, typically 7.5% at 50 Hz)
SERVO_RANGE = 2.5         # ±2.5% duty cycle for ±45° (5% to 10% duty cycle)
SERVO_MIN_DUTY = SERVO_CENTER - SERVO_RANGE  # 5% duty cycle (~45°)
SERVO_MAX_DUTY = SERVO_CENTER + SERVO_RANGE  # 10% duty cycle (~135°)

# PID parameters
KP = 0.1                  # Proportional gain
KI = 0.01                 # Integral gain
KD = 0.05                 # Derivative gain
integral = 0
last_error = 0

# Variables to store encoder counts
right_counter = 0
left_counter = 0

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
    right_pwm = GPIO.PWM(ENA, PWM_FREQ)
    left_pwm = GPIO.PWM(ENB, PWM_FREQ)
    right_pwm.start(0)
    left_pwm.start(0)
    
    # Set up PWM for servo
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(SERVO_CENTER)  # Start at center (90°)
    
    return right_pwm, left_pwm, servo_pwm

# Motor control functions
def move_forward(right_pwm, left_pwm, right_speed, left_speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(min(max(right_speed, 0), 100))
    left_pwm.ChangeDutyCycle(min(max(left_speed, 0), 100))

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def pivot_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Servo control function
def set_servo_angle(servo_pwm, line_angle):
    # Map line angle (0° to 180°) to servo duty cycle (5% to 10%)
    # 90° line angle = 90° servo angle (7.5% duty cycle)
    normalized_angle = (line_angle - 90) / 45  # Scale to -1 to 1
    duty_cycle = SERVO_CENTER + (normalized_angle * SERVO_RANGE)
    duty_cycle = min(max(duty_cycle, SERVO_MIN_DUTY), SERVO_MAX_DUTY)
    servo_pwm.ChangeDutyCycle(duty_cycle)
    return duty_cycle

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
            bottom_roi_y_start = FRAME_HEIGHT - BOTTOM_ROI_HEIGHT
            cv2.rectangle(frame, (0, bottom_roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)
            top_roi_y_end = TOP_ROI_HEIGHT
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, top_roi_y_end), (0, 255, 255), 2)
            roi = frame[bottom_roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
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

# Line detection function with top and bottom ROIs
def detect_line(frame, color_priorities, color_ranges):
    global integral, last_error
    
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Bottom ROI for PID (30% from bottom)
    bottom_roi_y_start = FRAME_HEIGHT - BOTTOM_ROI_HEIGHT
    bottom_roi = frame[bottom_roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
    bottom_hsv = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
    cv2.rectangle(frame, (0, bottom_roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)
    
    # Top ROI for line angle (70% from top)
    top_roi_y_end = TOP_ROI_HEIGHT
    top_roi = frame[0:top_roi_y_end, 0:FRAME_WIDTH]
    top_hsv = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, top_roi_y_end), (0, 255, 255), 2)
    
    bottom_best_contour = None
    bottom_best_color = None
    bottom_best_cx, bottom_best_cy = -1, -1
    bottom_max_area = 0
    
    top_best_contour = None
    top_best_color = None
    top_max_area = 0
    line_angle = 0
    
    valid_contours = {}
    all_available_colors = []
    
    for color_name in color_priorities:
        color_ranges_for_color = color_ranges.get(color_name, [])
        bottom_color_mask = np.zeros(bottom_hsv.shape[:2], dtype=np.uint8)
        top_color_mask = np.zeros(top_hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges_for_color:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            bottom_color_mask = cv2.bitwise_or(bottom_color_mask, cv2.inRange(bottom_hsv, lower, upper))
            top_color_mask = cv2.bitwise_or(top_color_mask, cv2.inRange(top_hsv, lower, upper))
        
        kernel = np.ones((5, 5), np.uint8)
        bottom_color_mask = cv2.morphologyEx(bottom_color_mask, cv2.MORPH_CLOSE, kernel)
        top_color_mask = cv2.morphologyEx(top_color_mask, cv2.MORPH_CLOSE, kernel)
        
        if color_name == 'black':
            cv2.imshow(f"{color_name} Bottom Mask", bottom_color_mask)
            cv2.imshow(f"{color_name} Top Mask", top_color_mask)
        
        bottom_contours, _ = cv2.findContours(bottom_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_contours, _ = cv2.findContours(top_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bottom_valid = [cnt for cnt in bottom_contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        top_valid = [cnt for cnt in top_contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        
        if bottom_valid or top_valid:
            valid_contours[color_name] = {'bottom': bottom_valid, 'top': top_valid}
            all_available_colors.append(color_name)
    
    for color_name in color_priorities:
        if color_name in valid_contours:
            if valid_contours[color_name]['bottom']:
                largest_bottom_contour = max(valid_contours[color_name]['bottom'], key=cv2.contourArea)
                area = cv2.contourArea(largest_bottom_contour)
                if area > bottom_max_area:
                    bottom_max_area = area
                    bottom_best_contour = largest_bottom_contour
                    bottom_best_color = color_name
            
            if valid_contours[color_name]['top']:
                largest_top_contour = max(valid_contours[color_name]['top'], key=cv2.contourArea)
                area = cv2.contourArea(largest_top_contour)
                if area > top_max_area:
                    top_max_area = area
                    top_best_contour = largest_top_contour
                    top_best_color = color_name
    
    error = 0
    line_found = False
    metadata = {}
    
    # Process bottom ROI for PID
    if bottom_best_contour is not None:
        M = cv2.moments(bottom_best_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bottom_best_cx = cx
            bottom_best_cy = cy + bottom_roi_y_start
            
            contour_color = (0, 255, 0) if bottom_best_color != 'black' else (128, 128, 128)
            cv2.drawContours(frame[bottom_roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH], 
                            [bottom_best_contour], -1, contour_color, 2)
            cv2.circle(frame, (bottom_best_cx, bottom_best_cy), 5, (255, 0, 0), -1)
            cv2.line(frame, (center_x, bottom_best_cy), (bottom_best_cx, bottom_best_cy), (255, 0, 0), 2)
            
            error = bottom_best_cx - center_x
            line_found = True
            
            hsv_value = bottom_hsv[cy, cx]
            h, s, v = hsv_value
            metadata['bottom_color'] = bottom_best_color
            metadata['error'] = error
            metadata['hsv'] = (int(h), int(s), int(v))
    
    # Process top ROI for line angle
    if top_best_contour is not None:
        [vx, vy, x0, y0] = cv2.fitLine(top_best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        line_angle = np.arctan2(vy, vx) * 180 / np.pi
        line_angle = line_angle.item() % 180
        
        contour_color = (0, 255, 0) if top_best_color != 'black' else (128, 128, 128)
        cv2.drawContours(frame[0:top_roi_y_end, 0:FRAME_WIDTH], 
                        [top_best_contour], -1, contour_color, 2)
        
        left_y = int(y0 - (x0 * vy / vx))
        right_y = int(y0 + ((FRAME_WIDTH - x0) * vy / vx))
        cv2.line(frame, (0, left_y), (FRAME_WIDTH, right_y), (0, 0, 255), 2)
        
        metadata['top_color'] = top_best_color
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
    
    return error, line_found, bottom_best_color, all_available_colors, line_angle

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    color_ranges = load_color_calibration()
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    calibrate_black_line(picam2, color_ranges)
    
    print("Line follower with PID, angle detection, and servo control started. Press 'q' or Ctrl+C to stop.")
    
    recovery_mode = False
    
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
            
            # PID control
            global integral, last_error
            integral += error
            derivative = error - last_error
            correction = KP * error + KI * integral + KD * derivative
            last_error = error
            
            # Control servo based on line angle
            duty_cycle = set_servo_angle(servo_pwm, line_angle)
            
            if line_found:
                recovery_mode = False
                if error > TURN_THRESHOLD:
                    pivot_right(right_pwm, left_pwm)
                    print(f"Pivot Right - {detected_color} line, Error: {error}, Line Angle: {line_angle}, Servo Duty: {duty_cycle:.2f}")
                elif error < -TURN_THRESHOLD:
                    pivot_left(right_pwm, left_pwm)
                    print(f"Pivot Left - {detected_color} line, Error: {error}, Line Angle: {line_angle}, Servo Duty: {duty_cycle:.2f}")
                else:
                    # Adjust motor speeds based on PID correction
                    right_speed = BASE_SPEED - correction
                    left_speed = BASE_SPEED + correction
                    move_forward(right_pwm, left_pwm, right_speed, left_speed)
                    print(f"Following - {detected_color} line, Error: {error}, Line Angle: {line_angle}, Servo Duty: {duty_cycle:.2f}, R:{right_speed:.1f}, L:{left_speed:.1f}")
            else:
                if not recovery_mode:
                    print("No line detected. Starting recovery...")
                    recovery_mode = True
                move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                print(f"Reversing - No line, Servo Duty: {duty_cycle:.2f}")
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        servo_pwm.ChangeDutyCycle(SERVO_CENTER)  # Reset servo to center
        time.sleep(0.1)
        servo_pwm.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()