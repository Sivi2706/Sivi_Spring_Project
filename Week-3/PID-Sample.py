import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import os

# Define the detect_color function
def detect_color(frame, color_ranges, tuning_file=None):
    """
    Detect the dominant color in the frame using HSV and return the color and largest contour.
    Incorporates camera tuning and adaptive noise reduction.
    
    Args:
        frame: Input image frame from the camera.
        color_ranges: Dictionary of color names and their HSV ranges.
        tuning_file: Path to the camera tuning JSON file (optional).
    
    Returns:
        detected_color: Name of the detected color or None.
        largest_contour: Largest contour of the detected color or None.
        combined_mask: Combined mask of all detected colors.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    max_area = 0
    detected_color = None
    largest_contour = None
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Adaptive noise reduction parameters
    kernel = np.ones((5, 5), np.uint8)
    
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply adaptive morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Optional: Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        if contours:
            # Get the largest contour for this color
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                detected_color = 'red' if color.startswith('red') else color
                largest_contour = contour
                
                # Debug: Calculate average HSV of the contour
                mask_temp = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_temp, [contour], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=mask_temp)[:3]
                print(f"Color: {detected_color}, Mean HSV: {mean_hsv}, Range: {lower} to {upper}")
    
    return detected_color, largest_contour, combined_mask

# Initialize camera
tuning_file = "/usr/share/libcamera/ipa/vc4/ov5647.json"  # Use OV5647 tuning file if available
picam2 = Picamera2()
if os.path.exists(tuning_file):
    picam2.load_tuning_file(tuning_file)
    print(f"Loaded tuning file: {tuning_file}")
else:
    print(f"Warning: Tuning file {tuning_file} not found. Using default tuning.")
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

# HSV ranges (calibrated for OV5647, adjust based on your environment)
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

finally:
    # Proper cleanup to avoid PWM TypeError
    try:
        stop()
        pwm1.stop()
        pwm2.stop()
    except Exception as e:
        print(f"Error during PWM cleanup: {e}")
    try:
        cv2.destroyAllWindows()
        picam2.stop()
    except Exception as e:
        print(f"Error during camera cleanup: {e}")
    try:
        GPIO.cleanup()
    except Exception as e:
        print(f"Error during GPIO cleanup: {e}")