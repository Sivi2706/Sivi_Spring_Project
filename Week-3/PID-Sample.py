from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from PIL import Image
from rgb_matrix import RGB_Matrix

# Initialize RGB Matrix
rr = RGB_Matrix(0x74)

# Initialize camera
camera = PiCamera()
camera.resolution = (640, 480)  # Reduced resolution for performance
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)  # Camera warmup

# Define motor driver GPIO pins
motor_in1 = 22  # Left motor forward
motor_in2 = 27  # Left motor backward
motor_in3 = 17  # Right motor forward
motor_in4 = 4   # Right motor backward
ENA = 13  # Left motor speed control
ENB = 12  # Right motor speed control

# Setup GPIO
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
}

# Color to RGB mapping for RGB matrix display
color_to_rgb = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    None: (0, 0, 0)  # Black for no detection
}

def detect_color(frame, color_ranges):
    """
    Detect the dominant color in the frame using HSV and return the color and largest contour.
    
    Args:
        frame: Input image frame from the camera.
        color_ranges: Dictionary of color names and their HSV ranges.
    
    Returns:
        detected_color: Name of the detected color or None.
        largest_contour: Largest contour of the detected color or None.
        combined_mask: Combined mask of all detected colors.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    max_area = 0
    detected_color = None
    largest_contour = None
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    kernel = np.ones((5, 5), np.uint8)
    
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                detected_color = 'red' if color.startswith('red') else color
                largest_contour = contour
                
                mask_temp = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_temp, [contour], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=mask_temp)[:3]
                print(f"Color: {detected_color}, Mean HSV: {mean_hsv}, Range: {lower} to {upper}")
    
    return detected_color, largest_contour, combined_mask

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
    return "Moving Forward"

def move_reverse():
    GPIO.output(motor_in1, GPIO.LOW)
    GPIO.output(motor_in2, GPIO.HIGH)
    GPIO.output(motor_in3, GPIO.LOW)
    GPIO.output(motor_in4, GPIO.HIGH)
    set_speed(reverse_speed, reverse_speed)
    return "Moving Reverse"

def stop():
    set_speed(0, 0)
    GPIO.output(motor_in1, GPIO.LOW)
    GPIO.output(motor_in2, GPIO.LOW)
    GPIO.output(m emery_in3, GPIO.LOW)
    GPIO.output(motor_in4, GPIO.LOW)
    return "Stopped"

def pid_control(error):
    global integral, previous_error
    proportional = error
    integral += error
    derivative = error - previous_error
    control_signal = Kp * proportional + Ki * integral + Kd * derivative
    previous_error = error
    return control_signal

def display_on_matrix(detected_color):
    """
    Display the detected color on the 8x8 RGB matrix.
    """
    rgb = color_to_rgb.get(detected_color, (0, 0, 0))
    # Create an 8x8 image with the detected color
    matrix_img = Image.new('RGB', (8, 8), rgb)
    rr.image(list(matrix_img.getdata()))

print("Press 'q' to exit the live feed.")

error = 0

try:
    for frame in camera.capture_continuous(raw_capture, format="rgb", use_video_port=True):
        image = frame.array

        # Detect color and contour
        detected_color, largest_contour, mask = detect_color(image, color_ranges)

        movement = "No line detected"
        outline_coords = "N/A"
        display_color = detected_color if detected_color else "None"

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            outline_coords = f"({x}, {y}, {w}, {h})"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            line_center = x + w // 2
            frame_center = image.shape[1] // 2
            error = line_center - frame_center
            control_signal = pid_control(error)
            
            right_speed = base_speed - control_signal
            left_speed = base_speed + control_signal
            set_speed(left_speed, right_speed)
            movement = move_forward()
        else:
            movement = move_reverse()

        # Display on RGB matrix
        display_on_matrix(detected_color)

        # Display metadata on frame
        metadata = [
            f"Color: {display_color}",
            f"Command: {movement}",
            f"Outline: {outline_coords}",
            f"Error: {error:.2f}"
        ]
        for i, text in enumerate(metadata):
            cv2.putText(image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the processed image if display is available
        if "DISPLAY" in os.environ:
            cv2.imshow("Color Line Detection", image)

        raw_capture.truncate(0)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping robot...")

except Exception as e:
    print("Error:", e)

finally:
    try:
        stop()
        pwm1.stop()
        pwm2.stop()
    except Exception as e:
        print(f"Error during PWM cleanup: {e}")
    try:
        cv2.destroyAllWindows()
        camera.close()
    except Exception as e:
        print(f"Error during camera cleanup: {e}")
    try:
        GPIO.cleanup()
    except Exception as e:
        print(f"Error during GPIO cleanup: {e}")