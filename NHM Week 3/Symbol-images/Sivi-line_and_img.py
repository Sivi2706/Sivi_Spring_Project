import cv2
import numpy as np
import os
import glob
from picamera2 import Picamera2
from collections import deque
import RPi.GPIO as GPIO
import time
import json

# GPIO Pins for Motor Control
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 800     # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height
TURN_THRESHOLD = 100       # Error threshold for pivoting
REVERSE_SPEED = 40         # Speed when reversing

# Variables for encoder counts
right_counter = 0
left_counter = 0

# Calibration file
CALIBRATION_FILE = "color_calibration.json"

# Default color ranges for line detection (HSV format)
default_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),    # Lower red range
        ([170, 167, 154], [180, 247, 234])  # Upper red range
    ],
    'blue': [
        ([100, 167, 60], [130, 255, 95])    # Blue range
    ],
    'green': [
        ([40, 180, 110], [75, 255, 190])    # Green range
    ],
    'yellow': [
        ([25, 150, 150], [35, 255, 255])    # Yellow range
    ],
    'black': [
        ([0, 0, 0], [179, 100, 75])         # Black range
    ]
}

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Allow camera to warm up

        # Run image preprocessing before entering main loop
        image_metadata = preprocess_images()
        print("Image metadata extracted from .png files:")
        for filename, contours in image_metadata.items():
            print(f"{filename}: {len(contours)} shapes")

        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

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
    return right_pwm, left_pwm

# Encoder callbacks
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

# Motor control functions
def set_motor_speed(right_pwm, left_pwm, left_speed, right_speed):
    GPIO.output(IN1, left_speed > 0)
    GPIO.output(IN2, left_speed < 0)
    GPIO.output(IN3, right_speed > 0)
    GPIO.output(IN4, right_speed < 0)
    left_pwm.ChangeDutyCycle(abs(left_speed))
    right_pwm.ChangeDutyCycle(abs(right_speed))

# Load calibrated color ranges for line detection
def load_calibrated_ranges():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as file:
            return json.load(file)
    else:
        return default_color_ranges

# Get user's color priority choice for line following
def get_color_priorities():
    return ['blue', 'black', 'red']

# Check if contour is complete (not touching frame edges)
def is_contour_complete(contour, frame_shape):
    x, y, w, h = cv2.boundingRect(contour)
    frame_height, frame_width = frame_shape[:2]
    if x <= 1 or y <= 1 or (x + w) >= (frame_width - 1) or (y + h) >= (frame_height - 1):
        return False
    return True

# Simplified shape detection
def detect_shapes(frame):
    shapes = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            shapes.append(approx)

    return shapes

# Preprocess and extract contour metadata from .png images in directory
def preprocess_images(directory="."):
    image_metadata = {}
    for filepath in glob.glob(os.path.join(directory, "*.png")):
        image = cv2.imread(filepath)
        if image is None:
            continue
        contours = detect_shapes(image)
        image_metadata[os.path.basename(filepath)] = contours
    return image_metadata

# Line Detection Function
def detect_line(frame, color_priorities, color_ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    line_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for color in color_priorities:
        if color in color_ranges:
            for lower, upper in color_ranges[color]:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                line_mask = cv2.bitwise_or(line_mask, mask)

    return line_mask

# Main Loop
def main():
    color_ranges = load_calibrated_ranges()
    color_priorities = get_color_priorities()

    picam2 = initialize_camera()
    if not picam2:
        return

    right_pwm, left_pwm = setup_gpio()

    try:
        while True:
            frame = picam2.capture_array()
            mask = detect_line(frame, color_priorities, color_ranges)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
                    M = cv2.moments(largest)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        error = cx - FRAME_WIDTH // 2

                        if abs(error) > TURN_THRESHOLD:
                            if error > 0:
                                set_motor_speed(right_pwm, left_pwm, BASE_SPEED, -BASE_SPEED)
                            else:
                                set_motor_speed(right_pwm, left_pwm, -BASE_SPEED, BASE_SPEED)
                        else:
                            set_motor_speed(right_pwm, left_pwm, BASE_SPEED, BASE_SPEED)
                    else:
                        set_motor_speed(right_pwm, left_pwm, -REVERSE_SPEED, -REVERSE_SPEED)
                else:
                    set_motor_speed(right_pwm, left_pwm, -REVERSE_SPEED, -REVERSE_SPEED)
            else:
                set_motor_speed(right_pwm, left_pwm, -REVERSE_SPEED, -REVERSE_SPEED)

    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
