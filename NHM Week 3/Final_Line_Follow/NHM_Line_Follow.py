import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import json
import os
from picamera2 import Picamera2

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
SERVO_PIN = 18            # Servo motor pin
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

# Threshold for turning
TURN_THRESHOLD = 100       # Error threshold for pivoting

# Recovery parameters
REVERSE_SPEED = 40         # Speed when reversing

# ROI parameters
USE_ROI = True             # Enable ROI for more focused line detection
ROI_HEIGHT = 150           # Height of the ROI from the bottom of the frame

# PWM settings
PWM_FREQ = 1000           # Motor PWM frequency
SERVO_FREQ = 50           # Servo PWM frequency
SERVO_NEUTRAL = 7.5       # Neutral position (90 degrees, 7.5% duty cycle)

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Calibration file
CALIBRATION_FILE = "color_calibration.json"

# Default color ranges (HSV format) in case calibration file is not found
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

# Function to load calibrated color ranges
def load_color_calibration():
    """Load calibrated color ranges from file or return defaults"""
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                loaded_ranges = json.load(f)
                print("Loaded calibrated color ranges from file.")
                return loaded_ranges
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            print("Using default color ranges.")
    else:
        print("No calibration file found. Using default color ranges.")
        print("Run the program to calibrate colors or create a calibration file.")
    
    return default_color_ranges

# Function to save calibrated color ranges
def save_color_calibration(color_ranges):
    """Save calibrated color ranges to file"""
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(color_ranges, f, indent=4)
        print(f"Saved calibrated color ranges to {CALIBRATION_FILE}")
    except Exception as e:
        print(f"Error saving calibration file: {e}")

# Function to get user's color priority choice
def get_color_choices():
    print("\nAvailable line colors to follow (priority order):")
    print("r = red (highest priority)")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black (lowest priority)")
    print("q = quit program")
    print("\nEnter colors in priority order (e.g., 'rb' for red then blue)")
    
    color_map = {
        'r': 'red',
        'b': 'blue',
        'g': 'green',
        'y': 'yellow',
        'k': 'black'
    }
    
    while True:
        choices = input("\nEnter color priorities (e.g., 'rbk'): ").lower()
        if choices == 'q':
            return None
            
        # Remove duplicates while preserving order
        seen = set()
        unique_choices = []
        for c in choices:
            if c in color_map and c not in seen:
                seen.add(c)
                unique_choices.append(c)
                
        if unique_choices:
            selected_colors = [color_map[c] for c in unique_choices]
            # Make sure black is always in the list (as lowest priority if not specified)
            if 'black' not in selected_colors:
                selected_colors.append('black')
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        else:
            print("Invalid choice. Please try again.")

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

    # Setup motor and encoder pins
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, SERVO_PIN], GPIO.OUT)
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Set up encoder interrupts
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    # Setup motor PWM
    left_pwm = GPIO.PWM(ENA, PWM_FREQ)  # Left motor
    right_pwm = GPIO.PWM(ENB, PWM_FREQ) # Right motor
    left_pwm.start(0)
    right_pwm.start(0)

    # Setup servo PWM
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    servo_pwm.start(SERVO_NEUTRAL)  # Start at neutral

    return left_pwm, right_pwm, servo_pwm

# Motor control functions
def turn_right(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    print("Turning Right")

def turn_left(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    print("Turning Left")

def move_forward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(BASE_SPEED)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    print("Moving Forward")

def move_backward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(REVERSE_SPEED)
    right_pwm.ChangeDutyCycle(REVERSE_SPEED)
    print("Moving Backward")

def stop_motors(left_pwm, right_pwm, servo_pwm):
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    servo_pwm.ChangeDutyCycle(SERVO_NEUTRAL)
    print("Stopped")

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera to warm up
    return picam2

# Function to calibrate a specific color
def calibrate_color(picam2, color_ranges, color_name):
    print(f"\nCalibrating {color_name} line detection...")
    print(f"Place the camera to view the {color_name} line and press 'c' to capture and calibrate.")
    print("Press 'q' to skip calibration for this color.")
    
    # Define initial broad HSV ranges for each color
    initial_ranges = {
        'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
        'blue': [([90, 100, 50], [130, 255, 255])],
        'green': [([35, 100, 50], [85, 255, 255])],
        'yellow': [([20, 100, 100], [40, 255, 255])],
        'black': [([0, 0, 0], [180, 100, 80])]
    }
    
    while True:
        frame = picam2.capture_array()
        
        # Draw ROI if enabled
        if USE_ROI:
            roi_y_start = FRAME_HEIGHT - ROI_HEIGHT
            cv2.rectangle(frame, (0, roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)
            roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        else:
            roi = frame
            
        cv2.putText(frame, f"Press 'c' to calibrate {color_name} line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"Skipped calibration for {color_name}.")
            cv2.destroyWindow("Calibration")
            return False
            
        if key == ord('c'):
            # Convert ROI to HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create a mask using initial broad range
            color_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            for lower, upper in initial_ranges[color_name]:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv_roi, lower, upper))
            
            # Find contours in the mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                color_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(color_contour) > MIN_CONTOUR_AREA:
                    # Create a mask from the contour
                    mask = np.zeros_like(color_mask)
                    cv2.drawContours(mask, [color_contour], -1, 255, -1)
                    
                    # Find the HSV range of pixels in the contour
                    roi_pixels = hsv_roi[mask == 255]
                    if len(roi_pixels) > 0:
                        h_min, s_min, v_min = np.min(roi_pixels, axis=0)
                        h_max, s_max, v_max = np.max(roi_pixels, axis=0)
                        
                        # Apply margins
                        h_min = max(0, h_min - 10)
                        s_min = max(0, s_min - 10)
                        v_min = max(0, v_min - 10)
                        h_max = min(179, h_max + 10)
                        s_max = min(255, s_max + 40)
                        v_max = min(255, v_max + 40)
                        
                        # Update color range
                        if color_name == 'red':
                            # Split into two ranges for red
                            color_ranges['red'] = [
                                ([0, s_min, v_min], [min(10, h_max), s_max, v_max]),
                                ([max(170, h_min), s_min, v_min], [180, s_max, v_max])
                            ]
                        else:
                            color_ranges[color_name] = [([h_min, s_min, v_min], [h_max, s_max, v_max])]
                        
                        print(f"Updated {color_name} line HSV range: ({h_min}, {s_min}, {v_min}) to ({h_max}, {s_max}, {v_max})")
                        
                        # Show the calibrated mask
                        calibrated_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                        for lower, upper in color_ranges[color_name]:
                            lower = np.array(lower, dtype=np.uint8)
                            upper = np.array(upper, dtype=np.uint8)
                            calibrated_mask = cv2.bitwise_or(calibrated_mask, cv2.inRange(hsv_roi, lower, upper))
                        
                        cv2.imshow(f"Calibrated {color_name.capitalize()} Line Mask", calibrated_mask)
                        cv2.waitKey(2000)
                        cv2.destroyWindow(f"Calibrated {color_name.capitalize()} Line Mask")
                        cv2.destroyWindow("Calibration")
                        return True
                    else:
                        print(f"Could not find {color_name} pixels in the contour. Please try again.")
                else:
                    print(f"{color_name.capitalize()} line contour too small. Please try again.")
            else:
                print(f"No {color_name} line detected. Please try again.")
                
# Refined line detection function
def detect_line(frame, color_priorities, color_ranges):
    # Apply ROI if enabled
    if USE_ROI:
        roi_y_start = FRAME_HEIGHT - ROI_HEIGHT
        roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Draw ROI border if enabled
    if USE_ROI:
        roi_y_start = FRAME_HEIGHT - ROI_HEIGHT
        cv2.rectangle(frame, (0, roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)

    best_contour = None
    best_color = None
    best_cx, best_cy = -1, -1
    max_area = 0

    # Dictionary to store all valid contours for each color
    valid_contours = {}

    # First, check for all available lines
    all_available_colors = []
    for color_name in color_priorities:
        color_ranges_for_color = color_ranges.get(color_name, [])
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in color_ranges_for_color:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Show the masks for debugging
        if color_name == 'black':
            cv2.imshow(f"{color_name} Mask", color_mask)
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

        if valid:
            valid_contours[color_name] = valid
            all_available_colors.append(color_name)

    # Prioritize: Find the best contour of the highest-priority available color
    for color_name in color_priorities:
        if color_name in valid_contours:
            largest_contour = max(valid_contours[color_name], key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > max_area:
                max_area = area
                best_contour = largest_contour
                best_color = color_name

    if best_contour is not None:
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            # Calculate center of contour
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Adjust coordinates if using ROI
            if USE_ROI:
                best_cx = cx
                best_cy = cy + (FRAME_HEIGHT - ROI_HEIGHT)
            else:
                best_cx = cx
                best_cy = cy

            # Draw contour and center point
            contour_color = (0, 255, 0) if best_color != 'black' else (128, 128, 128)
            
            if USE_ROI:
                # Draw on ROI portion
                cv2.drawContours(frame[FRAME_HEIGHT-ROI_HEIGHT:FRAME_HEIGHT, 0:FRAME_WIDTH], 
                                [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
            else:
                cv2.drawContours(frame, [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
                
            cv2.line(frame, (center_x, best_cy), (best_cx, best_cy), (255, 0, 0), 2)

            error = best_cx - center_x

            # Display information
            if USE_ROI:
                hsv_value = hsv[cy, cx]  # Use ROI coordinates
            else:
                hsv_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[best_cy, best_cx]
                
            h, s, v = hsv_value

            # Display all available colors
            available_colors_text = "Available: " + ", ".join(all_available_colors)
            cv2.putText(frame, available_colors_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Change text color based on the detected line color
            if best_color == 'red':
                text_color = (0, 0, 255)  # Red in BGR
            elif best_color == 'green':
                text_color = (0, 255, 0)  # Green
            elif best_color == 'blue':
                text_color = (255, 0, 0)  # Blue
            elif best_color == 'yellow':
                text_color = (0, 255, 255)  # Yellow
            elif best_color == 'black':
                text_color = (128, 128, 128)  # Gray
            else:
                text_color = (255, 255, 255)  # White
                
            cv2.putText(frame, f"{best_color.capitalize()} Line, Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(frame, f"HSV: ({h}, {s}, {v})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            return error, True, best_color, all_available_colors

    return 0, False, None, []

# Main function
def main():
    left_pwm, right_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    # Load calibrated color ranges
    color_ranges = load_color_calibration()
    
    # Calibrate all colors
    colors_to_calibrate = ['red', 'blue', 'green', 'yellow', 'black']
    print("\nStarting calibration for all colors...")
    for color in colors_to_calibrate:
        success = calibrate_color(picam2, color_ranges, color)
        if success:
            print(f"Calibration for {color} completed.")
        else:
            print(f"Using existing or default range for {color}.")
    
    # Save the updated color ranges
    save_color_calibration(color_ranges)
    
    # Get user's color priority choices
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    print("Line follower with color detection started. Press 'q' in the display window or Ctrl+C to stop.")
    
    # Flag to track if we're in recovery mode
    recovery_mode = False
    
    try:
        while True:
            frame = picam2.capture_array()
            error, line_found, detected_color, available_colors = detect_line(frame, color_priorities, color_ranges)
            
            cv2.imshow("Line Follower", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Allow recalibration during runtime for all colors
                for color in colors_to_calibrate:
                    success = calibrate_color(picam2, color_ranges, color)
                    if success:
                        print(f"Recalibration for {color} completed.")
                    else:
                        print(f"Skipped recalibration for {color}.")
                save_color_calibration(color_ranges)
            
            if line_found:
                # Line is detected (can be any color including black)
                recovery_mode = False
                if error > TURN_THRESHOLD:
                    turn_right(left_pwm, right_pwm)
                    print(f"Turning Right - {detected_color} line")
                elif error < -TURN_THRESHOLD:
                    turn_left(left_pwm, right_pwm)
                    print(f"Turning Left - {detected_color} line")
                else:
                    move_forward(left_pwm, right_pwm)
                    print(f"Moving Forward - {detected_color} line (Available: {', '.join(available_colors)})")
            else:
                # No line detected - only now do we go into recovery mode
                if not recovery_mode:
                    print("No line detected. Starting recovery...")
                    recovery_mode = True
                
                move_backward(left_pwm, right_pwm)
                print("Reversing to find line...")
                time.sleep(0.1)  # Small delay to avoid flooding the console
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(left_pwm, right_pwm, servo_pwm)
        left_pwm.stop()
        right_pwm.stop()
        servo_pwm.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()