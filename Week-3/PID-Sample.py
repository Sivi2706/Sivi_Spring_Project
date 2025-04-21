from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import os

# Initialize camera with tuning file
tuning_file = "/usr/share/libcamera/ipa/pisp/imx219.json"  # Adjust for your sensor and platform
picam2 = Picamera2()
if os.path.exists(tuning_file):
    picam2.load_tuning_file(tuning_file)
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

# Define motor driver GPIO pins
motor_in1 = 22  # Left motor forward
motor_in2 = 27  # Left motor backward
motor_in3 = 17  # Right motor forward
motor_in4 = 4   # Right motor backward
ENA = 13  # Left motor speed control
ENB = 12  # Right motor speed control

# Setup GPIO Mode
GPIO.setmode(GPIO.BCM)
GPIO.setup([motor_in1, motor_in2, motor_in3, motor_in4, ENA, ENB], GPIO.OUT)

# Setup PWM for speed control (frequency: 1kHz)
pwm1 = GPIO.PWM(ENA, 1000)
pwm2 = GPIO.PWM(ENB, 1000)
pwm1.start(0)
pwm2.start(0)

# Speed settings
base_speed = 55
max_speed = 100
reverse_speed = 50

# PID parameters
Kp = 0.5
Ki = 0
Kd = 0.5

# PID variables
integral = 0
previous_error = 0

# HSV ranges (calibrated based on Macbeth chart images)
color_ranges = {
    'red1': ([0, 100, 100], [10, 255, 255]),
    'red2': ([160, 100, 100], [179, 255, 255]),
    'blue': ([110, 100, 100], [130, 255, 255]),
    'green': ([45, 100, 100], [75, 255, 255]),
    'yellow': ([25, 100, 100], [35, 255, 255]),
    'cyan': ([85, 100, 100], [95, 255, 255]),
    'magenta': ([145, 100, 100], [155, 255, 255]),
}

def set_speed(left_speed, right_speed):
    left_speed = max(0, min(100, left_speed))
    right_speed = max(0, min(100, right_speed))
    pwm1.ChangeDutyCycle(left_speed)
    pwm2.ChangeDutyCycle(right_speed)

def move_forward():
    GPIO.output(motor_in1, GPIO.HIGH)
    GPIO.output(motor_in2, GPIO.LOW)
    GPIO.output(motor_in3, GPIO.HIGH)
    GPIO.output(motor_in4, GPIO.LOW)
    print("Moving Forward")
    return "Moving Forward"

def move_reverse():
    GPIO.output(motor_in1, GPIO.LOW)
    GPIO.output(motor_in2, GPIO.HIGH)
    GPIO.output(motor_in3, GPIO.LOW)
    GPIO.output(motor_in4, GPIO.HIGH)
    set_speed(reverse_speed, reverse_speed)
    print("Moving Reverse")
    return "Moving Reverse"

def stop():
    set_speed(0, 0)
    GPIO.output(motor_in1, GPIO.LOW)
    GPIO.output(motor_in2, GPIO.LOW)
    GPIO.output(motor_in3, GPIO.LOW)
    GPIO.output(motor_in4, GPIO.LOW)
    print("Stopping")
    return "Stopped"

def pid_control(error):
    global integral, previous_error
    proportional = error
    integral += error
    derivative = error - previous_error
    control_signal = Kp * proportional + Ki * integral + Kd * derivative
    previous_error = error
    return control_signal

print("Press 'q' to exit the live feed.")

# Initialize error
error = 0

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Detect color and contour
        detected_color, largest_contour, mask = detect_color(frame, color_ranges, tuning_file)

        movement = "No line detected"
        outline_coords = "N/A"
        display_color = detected_color if detected_color else "None"

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            outline_coords = f"({x}, {y}, {w}, {h})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            line_center = x + w // 2
            frame_center = frame.shape[1] // 2
            error = line_center - frame_center
            control_signal = pid_control(error)
            
            right_speed = base_speed - control_signal
            left_speed = base_speed + control_signal
            set_speed(left_speed, right_speed)
            movement = move_forward()
        else:
            movement = move_reverse()

        # Display metadata
        metadata = [
            f"Color: {display_color}",
            f"Command: {movement}",
            f"Outline: {outline_coords}",
            f"Error: {error:.2f}"
        ]
        for i, text in enumerate(metadata):
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the processed image
        if "DISPLAY" in os.environ:
            cv2.imshow("Color Line Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping robot...")

except Exception as e:
    print("Error:", e)

# Cleanup
stop()
cv2.destroyAllWindows()
picam2.stop()
GPIO.cleanup()