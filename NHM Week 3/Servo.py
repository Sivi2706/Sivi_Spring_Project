import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import os
import time
import math

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
SERVO_PIN = 18            # Servo motor pin

# PWM settings
PWM_FREQ = 1000           # Motor PWM frequency
SERVO_FREQ = 50           # Servo PWM frequency
PWM_DUTY_CYCLE = 100      # Duty cycle (max speed)

# Servo settings
SERVO_NEUTRAL = 7.5       # 90 degrees (center, 7.5% duty cycle)
SERVO_LEFT = 12.5         # 180 degrees (left, 12.5% duty cycle)
SERVO_RIGHT = 2.5         # 0 degrees (right, 2.5% duty cycle)
MAX_DUTY_CHANGE = 0.5     # Max duty cycle change per frame
MAX_ANGLE_CHANGE = 20.0   # Max angle change per frame (degrees)
MAX_ANGLE_STEP = 10.0     # Max angle step if change is too large
CONTROL_SCALE = 1.0       # Scale factor for normalized control signal

# Servo smoothing
last_duty = SERVO_NEUTRAL  # Track last duty cycle
previous_angle_top = 0     # Track previous top ROI angle

# Corner detection variables
last_detected_time = time.time()
corner_recovery_mode = False
corner_turn_direction = 0
corner_turn_start_time = 0

# Define all available color ranges (HSV format)
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

# Define color priority order
COLOR_PRIORITY = ['red', 'blue', 'green', 'yellow', 'black']

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup motor pins
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, SERVO_PIN], GPIO.OUT)

    # Setup motor PWM
    left_pwm = GPIO.PWM(ENA, PWM_FREQ)  # Left motor
    right_pwm = GPIO.PWM(ENB, PWM_FREQ) # Right motor
    left_pwm.start(0)
    right_pwm.start(0)

    # Setup servo PWM
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    servo_pwm.start(SERVO_NEUTRAL)  # Start at neutral

    return left_pwm, right_pwm, servo_pwm

def move_forward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Moving Forward")
    return "Moving Forward"

def move_backward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Moving Backward")
    return "Moving Backward"

def turn_left(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Turning Left")
    return "Turning Left"

def turn_right(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Turning Right")
    return "Turning Right"

def stop(left_pwm, right_pwm, servo_pwm):
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    servo_pwm.ChangeDutyCycle(SERVO_NEUTRAL)
    print("Stopping")
    return "Stopped"

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
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        else:
            print("Invalid choice. Please try again.")

def detect_priority_color(frame, color_names, roi_type='bottom'):
    """
    Detect colors in priority order within specified ROI
    Bottom ROI (30%) for motor control, Top ROI (30%) for servo
    Returns contour, color, and angle
    """
    height, width = frame.shape[:2]
    if roi_type == 'bottom':
        roi_height = int(height * 0.3)  # Bottom 30%
        roi = frame[height - roi_height:height, :]
        y_offset = height - roi_height
    else:  # top
        roi_height = int(height * 0.3)  # Top 30%
        roi = frame[0:roi_height, :]
        y_offset = 0
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    MIN_AREA = 500
    
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
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_AREA:
                    largest_contour[:, :, 1] += y_offset
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]
                    if angle < -45:
                        angle += 90
                    return largest_contour, color_name, angle
    
    return None, None, 0

def set_servo_angle(angle, servo_pwm):
    """
    Set servo angle based on detected line angle (degrees)
    Maps angle (-45° to +45°) to servo (180° left to 0° right)
    Limits speed and prevents sudden turns
    """
    global last_duty, previous_angle_top
    
    # Check for sudden angle changes
    angle_change = abs(angle - previous_angle_top)
    if angle_change > MAX_ANGLE_CHANGE:
        angle = previous_angle_top + (MAX_ANGLE_STEP * (1 if angle > previous_angle_top else -1))
    
    previous_angle_top = angle
    
    # Normalize angle to [-45, 45] degrees
    normalized_angle = max(-45, min(45, angle))
    
    # Map angle to control signal (-1 for 180°, 0 for 90°, +1 for 0°)
    control = -normalized_angle / 45.0
    
    # Calculate duty cycle
    duty = SERVO_NEUTRAL + (control * (SERVO_RIGHT - SERVO_LEFT) / 2)
    duty = max(SERVO_RIGHT, min(SERVO_LEFT, duty))
    
    # Limit servo speed
    duty_change = duty - last_duty
    if abs(duty_change) > MAX_DUTY_CHANGE:
        duty = last_duty + (MAX_DUTY_CHANGE * (1 if duty_change > 0 else -1))
    
    servo_pwm.ChangeDutyCycle(duty)
    last_duty = duty

def handle_corner_recovery(last_error, left_pwm, right_pwm, servo_pwm):
    global corner_recovery_mode, corner_turn_direction, corner_turn_start_time
    
    current_time = time.time()
    
    if not corner_recovery_mode:
        corner_turn_direction = 1 if last_error > 0 else -1
        corner_recovery_mode = True
        corner_turn_start_time = current_time
        return True
    
    if current_time - corner_turn_start_time < 0.5:
        if corner_turn_direction > 0:
            turn_right(left_pwm, right_pwm)
        else:
            turn_left(left_pwm, right_pwm)
        return True
    else:
        if current_time - corner_turn_start_time < 0.8:
            move_forward(left_pwm, right_pwm)
            servo_pwm.ChangeDutyCycle(SERVO_NEUTRAL)
            return True
        else:
            corner_recovery_mode = False
            return False

def main():
    global last_detected_time, corner_recovery_mode
    
    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    
    # Setup GPIO
    left_pwm, right_pwm, servo_pwm = setup_gpio()
    
    print("===== Improved Priority Color Line Follower with Turn Commands =====")
    
    color_priority = get_color_choices()
    if not color_priority:
        return
    
    print(f"\nColor priority: {' > '.join(color_priority)}")
    print("Press 'q' to quit or 'c' to change colors")
    
    try:
        last_error = 0
        while True:
            frame = picam2.capture_array()
            
            # Bottom ROI for motor control
            contour_bottom, color_name_bottom, line_angle_bottom = detect_priority_color(frame, color_priority, roi_type='bottom')
            
            # Top ROI for servo steering
            contour_top, color_name_top, line_angle_top = detect_priority_color(frame, color_priority, roi_type='top')
            
            movement = "No line detected"
            outline_coords = "N/A"
            current_color = "None"
            error = 0
            
            if contour_bottom is not None:
                last_detected_time = time.time()
                corner_recovery_mode = False
                
                x, y, w, h = cv2.boundingRect(contour_bottom)
                outline_coords = f"({x}, {y}, {w}, {h})"
                current_color = color_name_bottom
                
                color_map = {
                    'red': (0, 0, 255),
                    'blue': (255, 0, 0),
                    'green': (0, 255, 0),
                    'yellow': (0, 255, 255),
                    'black': (0, 0, 0)
                }
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[color_name_bottom], 2)
                
                # Fixed turn commands for motors
                line_center = x + w // 2
                frame_center = frame.shape[1] // 2
                error = line_center - frame_center
                last_error = error
                
                # Threshold to consider the line centered
                CENTER_THRESHOLD = 20
                
                if error < -CENTER_THRESHOLD:  # Line is left
                    movement = turn_left(left_pwm, right_pwm)
                elif error > CENTER_THRESHOLD:  # Line is right
                    movement = turn_right(left_pwm, right_pwm)
                else:  # Line is centered
                    movement = move_forward(left_pwm, right_pwm)
            else:
                time_since_last_detection = time.time() - last_detected_time
                
                servo_pwm.ChangeDutyCycle(SERVO_NEUTRAL)
                
                if time_since_last_detection < 0.5:
                    pass
                elif time_since_last_detection < 1.5:
                    movement = move_backward(left_pwm, right_pwm)
                    time.sleep(0.2)
                    stop(left_pwm, right_pwm, servo_pwm)
                else:
                    if handle_corner_recovery(last_error, left_pwm, right_pwm, servo_pwm):
                        movement = "Corner recovery"
                    else:
                        movement = "Searching for line"
            
            # Servo control based on top ROI
            if contour_top is not None and color_name_top == color_name_bottom:
                set_servo_angle(line_angle_top, servo_pwm)
            else:
                set_servo_angle(0, servo_pwm)
            
            priority_text = f"Priority: {'>'.join(color_priority)}"
            detection_text = f"Detected: {current_color}"
            command_text = f"Command: {movement}"
            error_text = f"Error: {error:.2f}"
            angle_text = f"Line Angle (Top): {line_angle_top:.2f}°"
            recovery_text = f"Recovery: {'Yes' if corner_recovery_mode else 'No'}"
            
            cv2.putText(frame, priority_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, command_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, error_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, angle_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, recovery_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if "DISPLAY" in os.environ:
                cv2.imshow("Improved Color Follower with Turn Commands", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                stop(left_pwm, right_pwm, servo_pwm)
                new_priority = get_color_choices()
                if new_priority:
                    color_priority = new_priority
                    print(f"New priority: {' > '.join(color_priority)}")
    
    except KeyboardInterrupt:
        print("Stopping robot...")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        stop(left_pwm, right_pwm, servo_pwm)
        left_pwm.stop()
        right_pwm.stop()
        servo_pwm.stop()
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()

if _name_ == "_main_":
    main()