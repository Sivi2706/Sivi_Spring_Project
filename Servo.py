import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

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
BASE_SPEED = 50            # Base motor speed (0-100)
TURN_SPEED = 65            # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 1000    # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning (using error from center of ROI)
TURN_THRESHOLD = 90

# Recovery parameters 
REVERSE_DURATION = 0.5     # Seconds to reverse
REVERSE_SPEED = 40         # Speed when reversing

# Updated scanning angles: center at 90, right at 45, left at 135.
SCAN_ANGLES = [90, 45, 135]
SCAN_TIME_PER_ANGLE = 0.5   # Seconds to wait per scan angle

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Global flag for upcoming 90° turn detected in top ROI
upcoming_turn_flag = False
turn_direction = None  # "LEFT" or "RIGHT"

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
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    
    # Set up PWM for servo
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

# Function to set servo angle (simple version for scanning and reset)
def set_servo_angle_simple(servo_pwm, angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Allow time for movement
    servo_pwm.ChangeDutyCycle(0)

# New function: use servo tuning logic to perform a turn based on a scanned angle.
def turn_with_scanned_angle(scanned_angle, servo_pwm, right_pwm, left_pwm):
    # Calculate turn time: assume a 45° turn takes 1 second
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

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

# Modified line detection function which works on any ROI frame.
# It returns error, line_found and intersection.
def detect_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = frame.shape[:2]
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
    
    intersection = False
    if contours:
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if len(valid_contours) >= 2:
            intersection = True
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cv2.circle(frame, (cx, height//2), 5, (255, 0, 0), -1)
                error = cx - center_x
                cv2.putText(frame, f"Error: {error}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return error, True, intersection
    return 0, False, intersection

# Main function with dual ROI logic and updated state machine.
def main():
    global upcoming_turn_flag, turn_direction
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    # Center the servo initially
    set_servo_angle_simple(servo_pwm, 90)
    
    # State variables: NORMAL, REVERSING, REVERSING_FOR_TURN, TURNING_FOR_FLAG, SCANNING
    state = "NORMAL"
    reverse_start_time = 0
    current_scan_index = 0
    scan_start_time = 0
    detected_scan_angle = None
    last_bottom_error = 0

    print("Line follower started. Press 'q' in the display window or Ctrl+C to stop.")
    
    try:
        while True:
            frame = picam2.capture_array()
            # Split the frame into top and bottom ROIs
            top_frame = frame[0:FRAME_HEIGHT//2, :]
            bottom_frame = frame[FRAME_HEIGHT//2:FRAME_HEIGHT, :]
            
            # Process bottom ROI for line following
            error_bottom, line_found_bottom, intersection_bottom = detect_line(bottom_frame)
            last_bottom_error = error_bottom  # Save for possible turn decision
            
            # Process top ROI for upcoming turn detection
            error_top, line_found_top, intersection_top = detect_line(top_frame)
            
            # For visualization, show the full frame with ROI boundaries drawn
            cv2.rectangle(frame, (0,0), (FRAME_WIDTH, FRAME_HEIGHT//2), (255, 0, 0), 2)  # Top ROI
            cv2.rectangle(frame, (0,FRAME_HEIGHT//2), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 0), 2)  # Bottom ROI
            cv2.imshow("Line Follower", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # State machine logic
            if state == "NORMAL":
                # Update upcoming turn flag: if the top ROI loses the line while bottom still sees it,
                # we assume a 90° turn is coming.
                if not line_found_top and line_found_bottom and not upcoming_turn_flag:
                    upcoming_turn_flag = True
                    if last_bottom_error < 0:
                        turn_direction = "LEFT"
                    elif last_bottom_error > 0:
                        turn_direction = "RIGHT"
                    print(f"Upcoming turn detected in top ROI. Setting flag for a {turn_direction} turn.")

                if line_found_bottom:
                    # Regular line following as before based on error in bottom ROI.
                    if error_bottom > TURN_THRESHOLD:
                        pivot_turn_right(right_pwm, left_pwm)
                        print("Pivot Turning Right (NORMAL)")
                    elif error_bottom < -TURN_THRESHOLD:
                        pivot_turn_left(right_pwm, left_pwm)
                        print("Pivot Turning Left (NORMAL)")
                    else:
                        move_forward(right_pwm, left_pwm)
                        print("Moving Forward (NORMAL)")
                else:
                    # Bottom ROI lost the line.
                    if upcoming_turn_flag:
                        print("Bottom ROI lost line with upcoming turn flag set. Reversing for turn...")
                        state = "REVERSING_FOR_TURN"
                        reverse_start_time = time.time()
                        move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                    else:
                        print("Bottom ROI lost line. Reversing for scan...")
                        state = "REVERSING"
                        reverse_start_time = time.time()
                        move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            
            elif state == "REVERSING":
                if time.time() - reverse_start_time >= REVERSE_DURATION:
                    stop_motors(right_pwm, left_pwm)
                    print("Beginning scan for line...")
                    state = "SCANNING"
                    current_scan_index = 0
                    set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                    scan_start_time = time.time()
            
            elif state == "REVERSING_FOR_TURN":
                if time.time() - reverse_start_time >= REVERSE_DURATION:
                    stop_motors(right_pwm, left_pwm)
                    state = "TURNING_FOR_FLAG"
            
            elif state == "TURNING_FOR_FLAG":
                # Use the stored turn_direction flag to execute a turn.
                turn_time = 1.0  # You can adjust this value as needed.
                if turn_direction == "LEFT":
                    print("Executing flagged LEFT turn")
                    pivot_turn_left(right_pwm, left_pwm)
                elif turn_direction == "RIGHT":
                    print("Executing flagged RIGHT turn")
                    pivot_turn_right(right_pwm, left_pwm)
                time.sleep(turn_time)
                stop_motors(right_pwm, left_pwm)
                # Reset the turn flag after executing the turn.
                upcoming_turn_flag = False
                turn_direction = None
                state = "NORMAL"
            
            elif state == "SCANNING":
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    # Check bottom ROI again after moving the servo
                    bottom_frame = picam2.capture_array()[FRAME_HEIGHT//2:FRAME_HEIGHT, :]
                    error_bottom, line_found_bottom, intersection_bottom = detect_line(bottom_frame)
                    
                    # If an intersection is detected in scan, center and go to NORMAL.
                    if intersection_bottom:
                        print("Intersection detected during scan. Resetting servo and returning to NORMAL.")
                        set_servo_angle_simple(servo_pwm, 90)
                        state = "NORMAL"
                    elif line_found_bottom:
                        detected_scan_angle = SCAN_ANGLES[current_scan_index]
                        print(f"Line detected during scan at servo angle: {detected_scan_angle}")
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
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle_simple(servo_pwm, 90)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()
