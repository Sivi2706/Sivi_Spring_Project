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
KP = 0.3                   # Proportional control gain
MIN_CONTOUR_AREA = 1000    # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Recovery parameters
REVERSE_DISTANCE_CM = 5    # Distance to reverse (cm)
SCAN_ANGLES = []           # Populated based on last error
SCAN_TIME_PER_ANGLE = 0.3  # Reduced scanning time

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
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

# Motor control functions
def move_forward(right_pwm, left_pwm, right_speed, left_speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(right_speed)
    left_pwm.ChangeDutyCycle(left_speed)

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

# Line detection function 
def detect_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > MIN_CONTOUR_AREA:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                error = cx - center_x
                cv2.line(frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)
                cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return error, True
    return 0, False

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    set_servo_angle(servo_pwm, 90)
    
    state = "NORMAL"
    reverse_start_pulses_right = 0
    reverse_start_pulses_left = 0
    required_pulses = 0
    current_scan_angle = 0
    scan_start_time = 0
    detected_error = 0
    servo_return_start = 0
    last_error = 0
    
    distance_per_pulse = WHEEL_CIRCUMFERENCE / PULSES_PER_REVOLUTION
    
    print("Line follower started. Press Ctrl+C to stop.")
    
    try:
        while True:
            frame = picam2.capture_array()
            error, line_found = detect_line(frame)
            cv2.imshow("Line Follower", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if state == "NORMAL":
                if line_found:
                    last_error = error
                    turn = KP * error
                    left_speed = int(BASE_SPEED + turn)
                    right_speed = int(BASE_SPEED - turn)
                    left_speed = max(0, min(100, left_speed))
                    right_speed = max(0, min(100, right_speed))
                    move_forward(right_pwm, left_pwm, right_speed, left_speed)
                    print(f"Proportional Control | L: {left_speed} | R: {right_speed}")
                else:
                    print("Line lost. Reversing...")
                    state = "REVERSING"
                    reverse_start_pulses_right = right_counter
                    reverse_start_pulses_left = left_counter
                    required_pulses = REVERSE_DISTANCE_CM / distance_per_pulse
                    move_backward(right_pwm, left_pwm, 30)
            
            elif state == "REVERSING":
                current_right = right_counter - reverse_start_pulses_right
                current_left = left_counter - reverse_start_pulses_left
                if current_right >= required_pulses and current_left >= required_pulses:
                    stop_motors(right_pwm, left_pwm)
                    print("Scanning...")
                    state = "SCANNING"
                    SCAN_ANGLES = [135, 45, 90] if last_error > 0 else [45, 135, 90]
                    current_scan_angle = 0
                    set_servo_angle(servo_pwm, SCAN_ANGLES[current_scan_angle])
                    scan_start_time = time.time()
            
            elif state == "SCANNING":
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    error, line_found = detect_line(picam2.capture_array())
                    if line_found:
                        print("Line detected during scan")
                        state = "RETURNING_SERVO"
                        set_servo_angle(servo_pwm, 90)
                        servo_return_start = time.time()
                    else:
                        current_scan_angle += 1
                        if current_scan_angle < len(SCAN_ANGLES):
                            set_servo_angle(servo_pwm, SCAN_ANGLES[current_scan_angle])
                            scan_start_time = time.time()
                        else:
                            print("No line found. Reversing again...")
                            state = "REVERSING"
                            reverse_start_pulses_right = right_counter
                            reverse_start_pulses_left = left_counter
                            move_backward(right_pwm, left_pwm, 30)
            
            elif state == "RETURNING_SERVO":
                if time.time() - servo_return_start >= 0.5:
                    state = "NORMAL"
                    print("Resuming normal operation")
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle(servo_pwm, 90)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()