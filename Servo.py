import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors ENA=Right ENB=Left
encoderPinRight = 23       # Right encoder
encoderPinLeft = 24        # Left encoder
ServoMotor = 18            # Servo motor PWM for the camera

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5       # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5      # Duty cycle for 180 degrees
SERVO_FREQ = 50            # 50Hz frequency for servo

# Line following parameters
BASE_SPEED = 40            # Base motor speed (0-100)
TURN_SPEED = 50            # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 1000    # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 50        # Adjust this value based on your needs

# Recovery parameters
REVERSE_DURATION = 0.5     # seconds to reverse
REVERSE_SPEED = 30         # speed when reversing
SCAN_ANGLES = [45, 135, 90]  # left, right, center
SCAN_TIME_PER_ANGLE = 0.5  # seconds per angle
PIVOT_DURATION = 0.5       # seconds to pivot

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

# Function to set servo angle
def set_servo_angle(servo_pwm, angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Give servo time to move
    servo_pwm.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter

# Motor control functions
def pivot_turn_right(right_pwm, left_pwm):
    # Right wheel backward, left wheel forward
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_turn_left(right_pwm, left_pwm):
    # Right wheel forward, left wheel backward
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
    picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.start()
    return picam2

# Line detection function (using full frame)
def detect_line(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define specific range for black color detection
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])  # Increased upper V value to include gray
    
    # Create mask for black regions (using full frame)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw center reference line
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Process contours
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > MIN_CONTOUR_AREA:
            # Calculate the moments of the contour
            M = cv2.moments(largest_contour)
            
            # Draw the contour
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            
            # If moment is valid, calculate centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw the centroid
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
                # Calculate the error (distance from center)
                error = cx - center_x
                
                # Draw line from center to centroid
                cv2.line(frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)
                
                # Display error value
                cv2.putText(frame, f"Error: {error}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                return error, True
    
    # Return zero error if no valid line detected
    return 0, False

# Main function
def main():
    # Initialize GPIO and PWM
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    
    # Initialize camera
    picam2 = setup_camera()
    
    # Set servo to center initially
    set_servo_angle(servo_pwm, 90)
    
    # State variables
    state = "NORMAL"
    reverse_start_time = 0
    current_scan_angle = 0
    scan_start_time = 0
    detected_error = 0
    servo_return_start = 0
    pivot_start_time = 0
    
    print("Line follower started. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Detect line and check if found
            error, line_found = detect_line(frame)
            
            # Display the frame
            cv2.imshow("Line Follower", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # State machine
            if state == "NORMAL":
                if line_found:
                    # Normal line following
                    if error > TURN_THRESHOLD:
                        pivot_turn_right(right_pwm, left_pwm)
                        print("Pivot Turning Right")
                    elif error < -TURN_THRESHOLD:
                        pivot_turn_left(right_pwm, left_pwm)
                        print("Pivot Turning Left")
                    else:
                        move_forward(right_pwm, left_pwm)
                        print("Moving Forward")
                else:
                    # Line lost, start reversing
                    print("Line lost. Reversing...")
                    state = "REVERSING"
                    reverse_start_time = time.time()
                    move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            
            elif state == "REVERSING":
                if time.time() - reverse_start_time >= REVERSE_DURATION:
                    # Stop reversing and start scanning
                    stop_motors(right_pwm, left_pwm)
                    print("Scanning for line...")
                    state = "SCANNING"
                    current_scan_angle = 0
                    set_servo_angle(servo_pwm, SCAN_ANGLES[current_scan_angle])
                    scan_start_time = time.time()
            
            elif state == "SCANNING":
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    # Check for line at current angle
                    frame = picam2.capture_array()
                    error, line_found = detect_line(frame)
                    if line_found:
                        detected_error = error
                        print(f"Line found during scan. Error: {detected_error}")
                        state = "RETURNING_SERVO"
                        set_servo_angle(servo_pwm, 90)
                        servo_return_start = time.time()
                    else:
                        # Move to next scan angle
                        current_scan_angle += 1
                        if current_scan_angle < len(SCAN_ANGLES):
                            set_servo_angle(servo_pwm, SCAN_ANGLES[current_scan_angle])
                            scan_start_time = time.time()
                        else:
                            # All angles scanned, no line found: reverse again
                            print("No line found. Reversing again...")
                            state = "REVERSING"
                            move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                            reverse_start_time = time.time()
            
            elif state == "RETURNING_SERVO":
                if time.time() - servo_return_start >= 0.5:
                    # Servo centered, start pivoting
                    state = "PIVOTING"
                    pivot_start_time = time.time()
                    if detected_error > TURN_THRESHOLD:
                        pivot_turn_right(right_pwm, left_pwm)
                        print("Pivoting Right based on scan error")
                    elif detected_error < -TURN_THRESHOLD:
                        pivot_turn_left(right_pwm, left_pwm)
                        print("Pivoting Left based on scan error")
                    else:
                        # No pivot needed, resume normal
                        state = "NORMAL"
                        print("Line centered. Resuming normal operation.")
            
            elif state == "PIVOTING":
                if time.time() - pivot_start_time >= PIVOT_DURATION:
                    stop_motors(right_pwm, left_pwm)
                    state = "NORMAL"
                    print("Pivot complete. Resuming normal operation.")
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        # Cleanup
        stop_motors(right_pwm, left_pwm)
        set_servo_angle(servo_pwm, 90)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()