import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors ENA=Right ENB=Left
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera

# Constants
WHEEL_DIAMETER = 4.05     # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Line following parameters
BASE_SPEED = 40           # Base motor speed (0-100)
TURN_SPEED = 50           # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 1000   # Minimum area for valid contours
FRAME_WIDTH = 640         # Camera frame width
FRAME_HEIGHT = 480        # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 50       # Adjust this value based on your needs

# Servo parameters
SERVO_CENTER = 7.5        # Center position (neutral)
SERVO_ANGLE_RANGE = 15    # Maximum tilt angle in degrees
SERVO_MIN = 7.5 - (SERVO_ANGLE_RANGE / 18)   # Left limit
SERVO_MAX = 7.5 + (SERVO_ANGLE_RANGE / 18)   # Right limit

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
    
    # Servo setup
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, 50)  # 50 Hz (20ms PWM period)
    servo_pwm.start(SERVO_CENTER)
    
    # Set up encoder interrupts
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    # Set up PWM for motors
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    
    right_pwm.start(0)
    left_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

# Servo control function
def set_servo_position(servo_pwm, position):
    """
    Set servo position
    :param servo_pwm: PWM object for the servo
    :param position: Duty cycle for the servo (between 2.5 and 12.5)
    """
    # Constrain position within limits
    position = max(SERVO_MIN, min(SERVO_MAX, position))
    
    servo_pwm.ChangeDutyCycle(position)
    time.sleep(0.1)  # Allow time for servo to move
    servo_pwm.ChangeDutyCycle(0)  # Stop sending signal to reduce jitter

# Motor control functions
def pivot_turn_right(right_pwm, left_pwm, servo_pwm):
    # Right wheel backward, left wheel forward
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    
    # Tilt servo to the right
    set_servo_position(servo_pwm, SERVO_MAX)

def pivot_turn_left(right_pwm, left_pwm, servo_pwm):
    # Right wheel forward, left wheel backward
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)    # Right forward
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    
    # Tilt servo to the left
    set_servo_position(servo_pwm, SERVO_MIN)

def move_forward(right_pwm, left_pwm, servo_pwm, error):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    left_pwm.ChangeDutyCycle(BASE_SPEED)
    
    # Slight tilt based on line error
    # Normalize error to be within servo angle range
    normalized_tilt = SERVO_CENTER + (error / FRAME_WIDTH) * (SERVO_MAX - SERVO_MIN)
    set_servo_position(servo_pwm, normalized_tilt)

def stop_motors(right_pwm, left_pwm, servo_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    
    # Center the servo
    set_servo_position(servo_pwm, SERVO_CENTER)

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.start()
    return picam2

# Line detection function (using full frame)
def detect_line(frame):
    # (Rest of the line detection code remains the same as in the previous script)
    # ... [previous detect_line function content] ...
    return 0  # Placeholder return

# Main function
def main():
    # Initialize GPIO and PWM
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    
    # Initialize camera
    picam2 = setup_camera()
    
    print("Line follower started. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Detect line and get error
            error = detect_line(frame)
            
            # Basic movement logic based on error
            if error > TURN_THRESHOLD:
                # Pivot turn right
                pivot_turn_right(right_pwm, left_pwm, servo_pwm)
                print(f"Pivot Turning Right at speed: {TURN_SPEED}")
            elif error < -TURN_THRESHOLD:
                # Pivot turn left
                pivot_turn_left(right_pwm, left_pwm, servo_pwm)
                print(f"Pivot Turning Left at speed: {TURN_SPEED}")
            else:
                # Move forward with gentle tilt based on error
                move_forward(right_pwm, left_pwm, servo_pwm, error)
                print(f"Moving Forward at speed: {BASE_SPEED}")
            
            # Display the frame
            cv2.imshow("Line Follower", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        # Cleanup
        stop_motors(right_pwm, left_pwm, servo_pwm)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()