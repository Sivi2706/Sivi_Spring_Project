from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import os

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

# Define motor driver GPIO pins
motor_in1 = 22  # Left motor forward
motor_in2 = 27  # Left motor backward
motor_in3 = 17  # Right motor forward
motor_in4 = 4   # Right motor backward

# PWM pins for speed control
ENA = 13  # Left motor speed control
ENB = 12  # Right motor speed control

# Setup GPIO Mode
GPIO.setmode(GPIO.BCM)
GPIO.setup([motor_in1, motor_in2, motor_in3, motor_in4, ENA, ENB], GPIO.OUT)

# Setup PWM for speed control (frequency: 1kHz)
pwm1 = GPIO.PWM(ENA, 1000)  # Left motor PWM
pwm2 = GPIO.PWM(ENB, 1000)  # Right motor PWM
pwm1.start(0)
pwm2.start(0)

# Speed settings
base_speed = 55  # Base speed for forward movement
max_speed = 100  # Maximum speed for turns
reverse_speed = 50  # Speed for reverse movement

# PID parameters
Kp = 0.5  # Proportional gain
Ki = 0    # Integral gain
Kd = 0.5  # Derivative gain

# PID variables
integral = 0
previous_error = 0

# HSV values corrected based on the RGB to HSV table
# OpenCV uses H: 0-179, S: 0-255, V: 0-255 (not the standard H: 0-360, S: 0-100%, V: 0-100%)
# Converting from standard to OpenCV: H * 0.5, S * 2.55, V * 2.55

# Define HSV ranges for line detection with correct OpenCV values
color_ranges = {
    # Red spans across 0 and 180 in the hue circle, so we need two ranges
    'red1': ([0, 100, 100], [10, 255, 255]),        # Lower red range
    'red2': ([160, 100, 100], [179, 255, 255]),     # Upper red range (not 170-180)
    'blue': ([110, 100, 100], [130, 255, 255]),     # Blue: ~240° in standard is ~120° in OpenCV
    'green': ([45, 100, 100], [75, 255, 255]),      # Green: ~120° in standard is ~60° in OpenCV
    'yellow': ([25, 100, 100], [35, 255, 255]),     # Yellow: ~60° in standard is ~30° in OpenCV
    'cyan': ([85, 100, 100], [95, 255, 255]),       # Cyan: ~180° in standard is ~90° in OpenCV
    'magenta': ([145, 100, 100], [155, 255, 255]),  # Magenta: ~300° in standard is ~150° in OpenCV
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

def detect_color(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_area = 0
    detected_color = None
    largest_contour = None
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        if contours:
            # Get the largest contour for this color
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                # Map red1 and red2 to 'red'
                detected_color = 'red' if color.startswith('red') else color
                largest_contour = contour

                # Debug: Calculate average HSV of the contour
                mask_temp = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_temp, [contour], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=mask_temp)[:3]
                print(f"Color: {detected_color}, Mean HSV: {mean_hsv}, Range: {lower} to {upper}")

    return detected_color, largest_contour, combined_mask

print("Press 'q' to exit the live feed.")

# Initialize error before the loop
error = 0

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Detect color and contour
        detected_color, largest_contour, mask = detect_color(frame)

        movement = "No line detected"
        outline_coords = "N/A"
        display_color = detected_color if detected_color else "None"

        if largest_contour is not None:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            outline_coords = f"({x}, {y}, {w}, {h})"

            # Draw outline on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Line center and frame center
            line_center = x + w // 2
            frame_center = frame.shape[1] // 2

            # Calculate error
            error = line_center - frame_center

            # Calculate control signal using PID
            control_signal = pid_control(error)

            # Adjust motor speeds based on control signal
            right_speed = base_speed - control_signal
            left_speed = base_speed + control_signal

            # Set motor speeds and move forward
            set_speed(left_speed, right_speed)
            movement = move_forward()

        else:
            # If no line is detected, reverse
            movement = move_reverse()

        # Display metadata on the frame
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

        # Press 'q' to exit
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