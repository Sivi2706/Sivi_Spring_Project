import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Define GPIO pins
IN1, IN2 = 22, 27
IN3, IN4 = 17, 4
ENA, ENB = 13, 12
encoderPinRight = 23
encoderPinLeft = 24
ServoMotor = 18

# Constants
WHEEL_DIAMETER = 4.05
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER

# Servo motor parameters
SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5
SERVO_FREQ = 50

# Line following parameters
BASE_SPEED = 65
TURN_SPEED = 50
MIN_CONTOUR_AREA = 1000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Threshold for turning
TURN_THRESHOLD = 85
DECELERATION_THRESHOLD = 120  # Threshold for speed reduction

# Recovery parameters
REVERSE_DURATION = 0.5
REVERSE_SPEED = 40

# Scanning angles
SCAN_ANGLES = [90, 45, 135]
SCAN_TIME_PER_ANGLE = 0.5

# Variables
right_counter = 0
left_counter = 0

# Encoder callbacks
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

    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)

    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)

    right_pwm = GPIO.PWM(ENA, 1000)
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)

    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)

    return right_pwm, left_pwm, servo_pwm

# Set servo angle
def set_servo_angle_simple(servo_pwm, angle):
    angle = max(0, min(angle, 180))
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

# Motor control functions
def move_forward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def pivot_turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

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

# Detect line function
def detect_line(frame, region="bottom"):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)

    if region == "bottom":
        mask_black = mask_black[FRAME_HEIGHT//2:, :]
    else:
        mask_black = mask_black[:FRAME_HEIGHT//3, :]

    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_x = FRAME_WIDTH // 2
    intersection = False

    if contours:
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if len(valid_contours) >= 2:
            intersection = True
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)  # Green line
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - center_x
                return error, True, intersection
    return 0, False, intersection

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    set_servo_angle_simple(servo_pwm, 90)

    try:
        while True:
            frame = picam2.capture_array()

            # Get errors from different regions
            error_main, line_found_main, _ = detect_line(frame, "bottom")
            error_decel, line_found_decel, _ = detect_line(frame, "top")

            # Adjust speed based on deceleration error
            speed = BASE_SPEED - min(abs(error_decel) / DECELERATION_THRESHOLD * BASE_SPEED, BASE_SPEED - 20)
            
            if line_found_main:
                if error_main > TURN_THRESHOLD:
                    pivot_turn_right(right_pwm, left_pwm)
                elif error_main < -TURN_THRESHOLD:
                    pivot_turn_left(right_pwm, left_pwm)
                else:
                    move_forward(right_pwm, left_pwm, speed)
            else:
                stop_motors(right_pwm, left_pwm)

            cv2.putText(frame, f"Main Err: {error_main}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Decel Err: {error_decel}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("Line Follower", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_motors(right_pwm, left_pwm)
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
