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
BASE_SPEED = 65            # Base motor speed (0-100)
TURN_SPEED = 50            # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 1000    # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Define ROI height for deceleration error (top 1/3 of the frame)
ROI_HEIGHT = FRAME_HEIGHT // 3

# Threshold for turning (based on main error)
TURN_THRESHOLD = 85       

# Recovery parameters
REVERSE_DURATION = 0.5     # Seconds to reverse
REVERSE_SPEED = 40         # Speed when reversing

# Scanning angles: center at 90, right at 45, left at 135.
SCAN_ANGLES = [90, 45, 135]
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
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    
    # Set up PWM for servo
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

# Function to set servo angle (for scanning and reset)
def set_servo_angle_simple(servo_pwm, angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Allow time for movement
    servo_pwm.ChangeDutyCycle(0)

# Function to perform a pivot turn based on scanned angle.
def turn_with_scanned_angle(scanned_angle, servo_pwm, right_pwm, left_pwm):
    turn_time = abs(scanned_angle - 90) / 45.0  # Assume 45° turn takes 1 second
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

# Modified forward function that decelerates using the deceleration error from the top ROI.
def move_forward(right_pwm, left_pwm, decel_error):
    # Use the deceleration error (from top region) for deceleration.
    deceleration_factor = (TURN_THRESHOLD - abs(decel_error)) / TURN_THRESHOLD
    deceleration_factor = max(0.3, deceleration_factor)  # minimum speed factor
    speed = BASE_SPEED * deceleration_factor

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)
    print(f"Moving Forward at speed: {speed:.2f} (Decel error: {decel_error})")

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

# Modified line detection function:
# Computes two error values:
#   1. main_error: using the largest valid contour in the entire frame.
#   2. decel_error: using the largest valid contour within the top ROI (y < ROI_HEIGHT).
# Also returns line_found and intersection flag.
def detect_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])  # Include dark gray
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_x = FRAME_WIDTH // 2
    # Draw center line (vertical) for reference
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    main_error = 0
    decel_error = 0
    line_found = False
    intersection = False
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
    
    if len(valid_contours) >= 2:
        intersection = True
        
    if valid_contours:
        # Compute main error from the largest contour overall.
        largest_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            main_error = cx - center_x
            line_found = True
            # Draw main error: red marker and line.
            cv2.circle(frame, (cx, int(M["m01"] / M["m00"])), 5, (0, 0, 255), -1)
            cv2.line(frame, (center_x, int(M["m01"] / M["m00"])), (cx, int(M["m01"] / M["m00"])), (0, 0, 255), 2)
            cv2.putText(frame, f"Main Err: {main_error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Compute deceleration error using contours within the top ROI.
        top_contours = []
        for cnt in valid_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cy = int(M["m01"] / M["m00"])
                if cy < ROI_HEIGHT:
                    top_contours.append(cnt)
        if top_contours:
            top_contour = max(top_contours, key=cv2.contourArea)
            M_top = cv2.moments(top_contour)
            if M_top["m00"] != 0:
                cx_top = int(M_top["m10"] / M_top["m00"])
                decel_error = cx_top - center_x
                # Draw deceleration error: blue marker and line.
                cv2.circle(frame, (cx_top, int(M_top["m01"] / M_top["m00"])), 5, (255, 0, 0), -1)
                cv2.line(frame, (center_x, int(M_top["m01"] / M_top["m00"])), (cx_top, int(M_top["m01"] / M_top["m00"])), (255, 0, 0), 2)
                cv2.putText(frame, f"Decel Err: {decel_error}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
    return main_error, decel_error, line_found, intersection

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    # Center the servo initially.
    set_servo_angle_simple(servo_pwm, 90)
    
    # State variables.
    state = "NORMAL"
    reverse_start_time = 0
    current_scan_index = 0
    scan_start_time = 0
    detected_scan_angle = None

    print("Line follower started. Press 'q' in the display window or Ctrl+C to stop.")
    
    try:
        while True:
            frame = picam2.capture_array()
            # Unpack four return values.
            main_error, decel_error, line_found, intersection = detect_line(frame)
            cv2.imshow("Line Follower", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if state == "NORMAL":
                if line_found:
                    # Use the main error for pivot decisions.
                    if main_error > TURN_THRESHOLD:
                        pivot_turn_right(right_pwm, left_pwm)
                        print("Pivot Turning Right")
                    elif main_error < -TURN_THRESHOLD:
                        pivot_turn_left(right_pwm, left_pwm)
                        print("Pivot Turning Left")
                    else:
                        # Use the deceleration error (from top ROI) for slowing forward.
                        move_forward(right_pwm, left_pwm, decel_error)
                else:
                    print("Line lost. Reversing...")
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
            
            elif state == "SCANNING":
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    frame = picam2.capture_array()
                    main_error, decel_error, line_found, intersection = detect_line(frame)
                    if intersection:
                        print("Intersection detected. Centering servo to 90° and adjusting.")
                        set_servo_angle_simple(servo_pwm, 90)
                        state = "NORMAL"
                    elif line_found:
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
