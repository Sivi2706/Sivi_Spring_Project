from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize camera
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
except Exception as e:
    logging.error(f"Failed to initialize camera: {e}")
    exit(1)

# Define motor driver GPIO pins
motor_in1 = 22  # Left motor forward
motor_in2 = 27  # Left motor backward
motor_in3 = 17  # Right motor forward
motor_in4 = 4   # Right motor backward

# PWM pins for speed control
ENA = 13  # Left motor speed control
ENB = 12  # Right motor speed control

# Setup GPIO Mode
try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([motor_in1, motor_in2, motor_in3, motor_in4, ENA, ENB], GPIO.OUT)
except Exception as e:
    logging.error(f"Failed to setup GPIO: {e}")
    picam2.stop()
    exit(1)

# Setup PWM for speed control (frequency: 1kHz)
try:
    pwm1 = GPIO.PWM(ENA, 1000)  # Left motor PWM
    pwm2 = GPIO.PWM(ENB, 1000)  # Right motor PWM
    pwm1.start(0)
    pwm2.start(0)
except Exception as e:
    logging.error(f"Failed to initialize PWM: {e}")
    picam2.stop()
    GPIO.cleanup()
    exit(1)

# Speed settings
base_speed = 55   # Base speed for forward movement
max_speed = 100   # Maximum speed for turns
reverse_speed = 50  # Speed for reverse movement

# PID parameters
Kp = 0.5  # Proportional gain
Ki = 0    # Integral gain
Kd = 0.5  # Derivative gain

# PID variables
integral = 0
previous_error = 0

def set_speed(left_speed, right_speed):
    try:
        left_speed = max(0, min(100, left_speed))
        right_speed = max(0, min(100, right_speed))
        pwm1.ChangeDutyCycle(left_speed)
        pwm2.ChangeDutyCycle(right_speed)
    except Exception as e:
        logging.error(f"Error setting speed: {e}")

def move_forward():
    try:
        GPIO.output(motor_in1, GPIO.HIGH)
        GPIO.output(motor_in2, GPIO.LOW)
        GPIO.output(motor_in3, GPIO.HIGH)
        GPIO.output(motor_in4, GPIO.LOW)
        logging.info("Moving Forward")
    except Exception as e:
        logging.error(f"Error moving forward: {e}")

def move_reverse():
    try:
        GPIO.output(motor_in1, GPIO.LOW)
        GPIO.output(motor_in2, GPIO.HIGH)
        GPIO.output(motor_in3, GPIO.LOW)
        GPIO.output(motor_in4, GPIO.HIGH)
        set_speed(reverse_speed, reverse_speed)
        logging.info("Moving Reverse")
    except Exception as e:
        logging.error(f"Error moving reverse: {e}")

def stop():
    try:
        set_speed(0, 0)
        GPIO.output(motor_in1, GPIO.LOW)
        GPIO.output(motor_in2, GPIO.LOW)
        GPIO.output(motor_in3, GPIO.LOW)
        GPIO.output(motor_in4, GPIO.LOW)
        logging.info("Stopping")
    except Exception as e:
        logging.error(f"Error stopping: {e}")

def pid_control(error):
    global integral, previous_error
    try:
        proportional = error
        integral += error
        derivative = error - previous_error
        control_signal = Kp * proportional + Ki * integral + Kd * derivative
        previous_error = error
        return control_signal
    except Exception as e:
        logging.error(f"Error in PID control: {e}")
        return 0

# Define color ranges in HSV for detection
colors = {
    "Blue": ((90, 100, 100), (130, 255, 255), (255, 0, 0)),    # Blue range, BGR color for outline
    "Black": ((0, 0, 0), (180, 255, 50), (0, 0, 0)),           # Black range, BGR color for outline
    "Red": ((0, 100, 100), (10, 255, 255), (0, 0, 255)),       # Red range, BGR color for outline
    "Green": ((40, 100, 100), (80, 255, 255), (0, 255, 0)),    # Green range, BGR color for outline
    "Yellow": ((20, 100, 100), (40, 255, 255), (0, 255, 255))  # Yellow range, BGR color for outline
}

print("Press 'q' to exit the live feed.")

try:
    while True:
        # Capture frame
        try:
            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                logging.warning("Failed to capture frame")
                continue
        except Exception as e:
            logging.error(f"Error capturing frame: {e}")
            continue

        # Convert to HSV for color detection
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            display_frame = frame.copy()
        except Exception as e:
            logging.error(f"Error converting to HSV: {e}")
            continue

        detected_color = None
        largest_contour = None
        max_area = 0

        # Check each color
        for color_name, (lower, upper, outline_color) in colors.items():
            try:
                # Create mask for the color
                mask = cv2.inRange(hsv, lower, upper)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:  # This is safe for lists
                    # Get the largest contour for this color
                    contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour)
                    
                    if area > max_area:
                        max_area = area
                        largest_contour = contour
                        detected_color = (color_name, outline_color)
            except Exception as e:
                logging.error(f"Error processing color {color_name}: {e}")
                continue

        if largest_contour and detected_color:
            try:
                # Get bounding box for the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Draw bounding box and label
                color_name, outline_color = detected_color
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), outline_color, 2)
                cv2.putText(display_frame, f"{color_name} Line", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, outline_color, 2)

                # Line center and frame center
                line_center = x + w // 2
                frame_center = display_frame.shape[1] // 2

                # Calculate error
                error = line_center - frame_center

                # Calculate control signal using PID
                control_signal = pid_control(error)

                # Adjust motor speeds based on control signal
                right_speed = base_speed - control_signal
                left_speed = base_speed + control_signal

                # Set motor speeds and move forward
                set_speed(left_speed, right_speed)
                move_forward()

                # Display direction text
                direction = f"Error: {error:.2f}, Control: {control_signal:.2f}"
                cv2.putText(display_frame, direction, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                logging.error(f"Error processing contour: {e}")
                continue
        else:
            # If no line is detected, reverse until a line is found
            try:
                move_reverse()
                direction = "No line detected - Reversing"
                cv2.putText(display_frame, direction, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e:
                logging.error(f"Error in reverse mode: {e}")
                continue

        # Show the processed RGB image with colored outlines
        if "DISPLAY" in os.environ:
            try:
                cv2.imshow("Line Detection (RGB with Colored Outlines)", display_frame)
            except Exception as e:
                logging.error(f"Error displaying frame: {e}")

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logging.info("Stopping robot due to keyboard interrupt")
except Exception as e:
    logging.error(f"Unexpected error in main loop: {e}")

# Cleanup
try:
    stop()
    pwm1.stop()  # Explicitly stop PWM
    pwm2.stop()  # Explicitly stop PWM
    cv2.destroyAllWindows()
    picam2.stop()
    GPIO.cleanup()
except Exception as e:
    logging.error(f"Error during cleanup: {e}")