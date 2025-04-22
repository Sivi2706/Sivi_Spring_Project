import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import os
import time

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5       # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5      # Duty cycle for 180 degrees
SERVO_FREQ = 50            # 50Hz frequency for servo

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 1000    # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# PID parameters
Kp = 0.5
Ki = 0
Kd = 0.5

# PID variables
integral = 0
previous_error = 0

# Recovery parameters
REVERSE_DURATION = 0.5     # Seconds to reverse
REVERSE_SPEED = 40         # Speed when reversing

# Scanning angles
SCAN_ANGLES = [90, 45, 135]  # Center, right, left
SCAN_TIME_PER_ANGLE = 0.5   # Seconds to wait per scan angle

# Variables to store encoder counts
right_counter = 0
left_counter = 0

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
    
    # Set up encoder interrupts
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    # Set up PWM for motors
    right_pwm = GPIO.PWM(ENA, 1000)  # Right motor
    left_pwm = GPIO.PWM(ENB, 1000)   # Left motor
    right_pwm.start(0)
    left_pwm.start(0)
    
    # Set up PWM for servo
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

# Function to set servo angle
def set_servo_angle_simple(servo_pwm, angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

# Function to perform a turn based on scanned angle
def turn_with_scanned_angle(scanned_angle, servo_pwm, right_pwm, left_pwm):
    turn_time = abs(scanned_angle - 90) / 45.0
    if scanned_angle > 90:
        print(f"Detected angle {scanned_angle}: Pivoting LEFT for {turn_time:.2f} seconds")
        GPIO.output(IN1, GPIO.LOW)    # Left backward
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)    # Right forward
        GPIO.output(IN4, GPIO.HIGH)
        right_pwm.ChangeDutyCycle(TURN_SPEED)
        left_pwm.ChangeDutyCycle(TURN_SPEED)
    elif scanned_angle < 90:
        print(f"Detected angle {scanned_angle}: Pivoting RIGHT for {turn_time:.2f} seconds")
        GPIO.output(IN1, GPIO.HIGH)   # Left forward
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)   # Right backward
        GPIO.output(IN4, GPIO.LOW)
        right_pwm.ChangeDutyCycle(TURN_SPEED)
        left_pwm.ChangeDutyCycle(TURN_SPEED)
    else:
        print("Detected angle 90: No pivot required.")
        return

    time.sleep(turn_time)
    stop_motors(right_pwm, left_pwm)
    print("Resetting servo to 90 degrees")
    set_servo_angle_simple(servo_pwm, 90)

# Motor control functions
def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    return "Moving Forward"

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)
    return "Moving Backward"

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    return "Stopped"

def set_speed(right_pwm, left_pwm, right_speed, left_speed):
    right_speed = max(0, min(100, right_speed))
    left_speed = max(0, min(100, left_speed))
    right_pwm.ChangeDutyCycle(right_speed)
    left_pwm.ChangeDutyCycle(left_speed)

# PID control
def pid_control(error):
    global integral, previous_error
    proportional = error
    integral += error
    derivative = error - previous_error
    control_signal = Kp * proportional + Ki * integral + Kd * derivative
    previous_error = error
    return control_signal

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

# Function to get user's color choices
def get_color_choice():
    print("\nAvailable line colors to follow:")
    print("r = red")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black (lowest priority)")
    print("q = quit program")
    print("Enter colors in priority order (e.g., 'rbgyk' for red > blue > green > yellow > black)")
    
    while True:
        choice = input("\nEnter line colors (e.g., rbgyk): ").lower().strip()
        if choice == 'q':
            return None
        if not choice:
            print("Input cannot be empty. Please enter one or more colors (r, b, g, y, k) or q to quit.")
            continue
        
        valid_chars = {'r', 'b', 'g', 'y', 'k'}
        invalid_chars = set(choice) - valid_chars
        if invalid_chars:
            print(f"Invalid characters: {invalid_chars}. Please use only r, b, g, y, k, or q.")
            continue
        
        color_map = {'r': 'red', 'b': 'blue', 'g': 'green', 'y': 'yellow', 'k': 'black'}
        colors = []
        seen = set()
        for char in choice:
            if char not in seen:
                colors.append(color_map[char])
                seen.add(char)
        
        if not colors:
            print("No valid colors selected. Please enter one or more colors (r, b, g, y, k).")
            continue
        
        return colors

# Multiple color detection
def detect_color(frame, color_names):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_area = 0
    largest_contour = None
    mean_hsv = None
    detected_color = None
    intersection = False
    
    kernel = np.ones((5, 5), np.uint8)
    all_color_ranges = {
        'red': [
            ([0, 167, 154], [10, 247, 234]),
            ([114, 167, 154], [134, 247, 234])
        ],
        'blue': [
            ([6, 167, 60], [26, 255, 95])
        ],
        'green': [
            ([31, 180, 110], [51, 255, 190])
        ],
        'yellow': [
            ([84, 155, 189], [104, 235, 255])
        ],
        'black': [
            ([0, 0, 0], [179, 78, 50])
        ]
    }
    
    contours_all = []
    for color_name in color_names:
        color_ranges = all_color_ranges.get(color_name, [])
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MIN_CONTOUR_AREA:
                    contours_all.append((contour, color_name, area))
    
    if contours_all:
        valid_contours = [c for c in contours_all if c[2] > MIN_CONTOUR_AREA]
        if len(valid_contours) >= 2:
            intersection = True
        
        # Sort contours by color priority (based on color_names order) and then by area
        for color_name in color_names:
            color_contours = [c for c in valid_contours if c[1] == color_name]
            if color_contours:
                contour, color, area = max(color_contours, key=lambda x: x[2])
                if area > max_area:
                    max_area = area
                    largest_contour = contour
                    detected_color = color
                
                mask_temp = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_temp, [largest_contour], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=mask_temp)[:3]
                break  # Stop after finding the highest-priority color
    
    return largest_contour, mean_hsv, detected_color, intersection

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    set_servo_angle_simple(servo_pwm, 90)
    
    print("===== Color Line Follower =====")
    color_names = get_color_choice()
    if color_names is None:
        return
    
    print(f"\nFollowing colors in order: {', '.join(color_names)}. Press 'q' to exit or 'c' to change colors.")
    print("Place the robot on the line to begin...")
    
    state = "NORMAL"
    reverse_start_time = 0
    current_scan_index = 0
    scan_start_time = 0
    detected_scan_angle = None
    error = 0
    
    try:
        while True:
            frame = picam2.capture_array()
            largest_contour, mean_hsv, detected_color, intersection = detect_color(frame, color_names)
            
            movement = "No line detected"
            outline_coords = "N/A"
            left_speed = 0
            right_speed = 0
            
            if state == "NORMAL":
                if largest_contour is not None:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    outline_coords = f"({x}, {y}, {w}, {h})"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    line_center = x + w // 2
                    frame_center = FRAME_WIDTH // 2
                    error = line_center - frame_center
                    control_signal = pid_control(error)
                    
                    right_speed = BASE_SPEED - control_signal
                    left_speed = BASE_SPEED + control_signal
                    set_speed(right_pwm, left_pwm, right_speed, left_speed)
                    movement = move_forward(right_pwm, left_pwm)
                    print(f"Following {detected_color} line")
                    
                    if intersection:
                        print("Intersection detected. Centering servo to 90° and continuing.")
                        set_servo_angle_simple(servo_pwm, 90)
                else:
                    print("Line lost. Reversing...")
                    state = "REVERSING"
                    reverse_start_time = time.time()
                    movement = move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            
            elif state == "REVERSING":
                if time.time() - reverse_start_time >= REVERSE_DURATION:
                    stop_motors(right_pwm, left_pwm)
                    print("Beginning scan for line...")
                    state = "SCANNING"
                    current_scan_index = 0
                    set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                    scan_start_time = time.time()
            
            elif state == "SCANNING":
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    frame = picam2.capture_array()
                    largest_contour, mean_hsv, detected_color, intersection = detect_color(frame, color_names)
                    
                    if intersection:
                        print("Intersection detected during scan. Centering servo to 90° and continuing.")
                        set_servo_angle_simple(servo_pwm, 90)
                        state = "NORMAL"
                    elif largest_contour is not None:
                        detected_scan_angle = SCAN_ANGLES[current_scan_index]
                        print(f"Line detected during scan at servo angle: {detected_scan_angle} (color: {detected_color})")
                        state = "TURNING"
                    else:
                        current_scan_index += 1
                        if current_scan_index < len(SCAN_ANGLES):
                            set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                            scan_start_time = time.time()
                        else:
                            print("No line found during scan. Reversing again...")
                            state = "REVERSING"
                            move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                            reverse_start_time = time.time()
            
            elif state == "TURNING":
                if detected_scan_angle is not None:
                    turn_with_scanned_angle(detected_scan_angle, servo_pwm, right_pwm, left_pwm)
                state = "NORMAL"
            
            # Display metadata
            color_code = f"Tracking: {detected_color if detected_color else 'None'}"
            metadata = [
                color_code,
                f"Command: {movement}",
                f"Outline: {outline_coords}",
                f"Error: {error:.2f}",
                f"Left PWM: {left_speed:.2f}",
                f"Right PWM: {right_speed:.2f}",
                f"Intersection: {'Yes' if intersection else 'No'}"
            ]
            
            for i, text in enumerate(metadata):
                cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if "DISPLAY" in os.environ:
                cv2.imshow(f"Color Line Detection - Following {', '.join(color_names)}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                stop_motors(right_pwm, left_pwm)
                color_names = get_color_choice()
                if color_names is None:
                    break
                print(f"\nNow following colors in order: {', '.join(color_names)}...")
                state = "NORMAL"
                set_servo_angle_simple(servo_pwm, 90)
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle_simple(servo_pwm, 90)
        right_pwm.stop()
        left_pwm.stop()
        servo_pwm.stop()
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()