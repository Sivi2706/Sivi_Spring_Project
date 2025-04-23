import numpy as np
import cv2
import json
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

# Define GPIO pins
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
MIN_CONTOUR_AREA = 800    # Minimum area for valid contours
FRAME_WIDTH = 640         # Camera frame width
FRAME_HEIGHT = 480        # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100      # Error threshold for pivoting
CENTER_THRESHOLD = 10     # For responsive turns

# Recovery parameters
REVERSE_SPEED = 40        # Speed when reversing

# ROI parameters
USE_ROI = True            # Enable ROI for more focused line detection
ROI_HEIGHT = 150          # Height of the ROI from the bottom of the frame

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

# Global variable for temporal smoothing of angles
prev_line_angle = 0
SMOOTHING_FACTOR = 0.7  # For exponential smoothing

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
        print("Run the color_calibration.py script first to create calibrated ranges.")
    
    return default_color_ranges

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
    
    return right_pwm, left_pwm

# Initialize camera with white balance adjustment
def setup_camera():
    picam2 = Picamera2()
    # Configure the camera to capture in BGR format
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    )
    picam2.configure(config)

    # Disable automatic white balance and set manual gains
    picam2.set_controls({
        "AwbEnable": 0,  # Disable AWB
        "ColourGains": (1.5, 2.0)  # (red_gain, blue_gain) - increase blue gain to correct blue colors
    })

    picam2.start()
    time.sleep(2)  # Allow camera to warm up
    return picam2

# Function to allow user to calibrate selected color detection parameters
def calibrate_color(picam2, color_ranges, color_name):
    print(f"\nCalibrating {color_name} line detection...")
    print(f"Place the camera to view the {color_name} line and press 'c' to capture and calibrate.")
    print("Press 'q' to skip calibration.")
    
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
            cv2.destroyWindow("Calibration")
            return
            
        if key == ord('c'):
            # Convert ROI to HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create a broad initial mask for the color
            if color_name == 'red':
                # Combine lower and upper red ranges
                mask1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv_roi, np.array([170, 100, 100]), np.array([180, 255, 255]))
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                # Use a broad range for other colors
                broad_ranges = {
                    'blue': ([90, 100, 50], [140, 255, 255]),
                    'green': ([30, 100, 50], [85, 255, 255]),
                    'yellow': ([20, 100, 100], [40, 255, 255]),
                    'black': ([0, 0, 0], [180, 100, 80])
                }
                lower, upper = broad_ranges.get(color_name, ([0, 0, 0], [180, 255, 255]))
                color_mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            
            # Find contours in the mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                    # Create a mask from the contour
                    mask = np.zeros_like(color_mask)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                    
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
                            # Split into lower and upper ranges
                            if h_max > 90:  # Upper red range
                                color_ranges['red'] = [([170, s_min, v_min], [180, s_max, v_max])]
                            else:  # Lower red range
                                color_ranges['red'] = [([h_min, s_min, v_min], [h_max, s_max, v_max])]
                        else:
                            color_ranges[color_name] = [([h_min, s_min, v_min], [h_max, s_max, v_max])]
                        
                        print(f"Updated {color_name} line HSV range: ({h_min}, {s_min}, {v_min}) to ({h_max}, {s_max}, {v_max})")
                        
                        # Show the calibrated mask
                        if color_name == 'red' and h_max > 90:
                            calibrated_mask = cv2.inRange(hsv_roi, np.array([170, s_min, v_min]), np.array([180, s_max, v_max]))
                        else:
                            calibrated_mask = cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                        cv2.imshow(f"Calibrated {color_name.capitalize()} Line Mask", calibrated_mask)
                        cv2.waitKey(2000)
                        cv2.destroyWindow(f"Calibrated {color_name.capitalize()} Line Mask")
                        break
                    else:
                        print(f"Could not find {color_name} pixels in the contour. Please try again.")
                else:
                    print(f"{color_name.capitalize()} line contour too small. Please try again.")
            else:
                print(f"No {color_name} line detected. Please try again.")
                
    cv2.destroyWindow("Calibration")

# Refined line detection function
def detect_line(frame, color_priorities, color_ranges, tight_turn=False):
    global prev_line_angle
    
    # Apply ROI if enabled - use dynamic ROI for tight turns
    if USE_ROI:
        if tight_turn:
            roi_height = int(FRAME_HEIGHT * 0.4)  # Reduce to 40% for tight turns
        else:
            roi_height = ROI_HEIGHT  # Default ROI height
        roi_y_start = FRAME_HEIGHT - roi_height
        roi = frame[roi_y_start:FRAME_HEIGHT, 0:FRAME_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Draw ROI border if enabled
    if USE_ROI:
        cv2.rectangle(frame, (0, roi_y_start), (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 0), 2)

    best_contour = None
    best_color = None
    best_cx, best_cy = -1, -1
    max_area = 0
    line_angle = 0

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
        
        # Show the masks for debugging for all colors
        cv2.imshow(f"{color_name.capitalize()} Mask", color_mask)
        
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
                best_cy = cy + (FRAME_HEIGHT - roi_height)
            else:
                best_cx = cx
                best_cy = cy

            # Draw contour and center point
            contour_color = (0, 255, 0) if best_color != 'black' else (128, 128, 128)
            
            if USE_ROI:
                # Draw on ROI portion
                cv2.drawContours(frame[FRAME_HEIGHT-roi_height:FRAME_HEIGHT, 0:FRAME_WIDTH], 
                                [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
            else:
                cv2.drawContours(frame, [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
                
            cv2.line(frame, (center_x, best_cy), (best_cx, best_cy), (255, 0, 0), 2)

            error = best_cx - center_x

            # Calculate line angle using segmented contour analysis
            epsilon = 0.01 * cv2.arcLength(best_contour, True)
            smoothed_contour = cv2.approxPolyDP(best_contour, epsilon, True)
            
            # Split contour into segments
            segment_size = len(smoothed_contour) // 3  # Split into 3 segments
            angles = []
            for i in range(0, len(smoothed_contour), segment_size):
                segment = smoothed_contour[i:i + segment_size]
                if len(segment) >= 5:
                    [vx, vy, x, y] = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01)
                    angle = np.degrees(np.arctan2(vy, vx))[0]
                    if angle < -45:
                        angle += 90
                    elif angle > 45:
                        angle -= 90
                    angles.append(angle)
            
            # Average the segment angles
            if angles:
                line_angle = np.mean(angles)
            else:
                # Fallback to single fitLine if segmentation fails
                if len(smoothed_contour) >= 5:
                    [vx, vy, x, y] = cv2.fitLine(smoothed_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    line_angle = np.degrees(np.arctan2(vy, vx))[0]
                    if line_angle < -45:
                        line_angle += 90
                    elif line_angle > 45:
                        line_angle -= 90
            
            # Apply temporal smoothing to the angle
            line_angle = SMOOTHING_FACTOR * prev_line_angle + (1 - SMOOTHING_FACTOR) * line_angle
            prev_line_angle = line_angle

            # Display information
            if USE_ROI:
                hsv_value = hsv[cy, cx]  # Use ROI coordinates
            else:
                hsv_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[best_cy, best_cx]
                
            h, s, v = hsv_value

            # Display all available colors
            available_colors_text = "Available: " + ", ".join(all_available_colors)
            
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

            return error, True, best_color, all_available_colors, line_angle, (best_cx, best_cy), available_colors_text, text_color, (h, s, v)

    return 0, False, None, [], 0, (-1, -1), "Available: None", (255, 255, 255), (0, 0, 0)

# Main function
def main():
    right_pwm, left_pwm = setup_gpio()
    picam2 = setup_camera()
    
    # Load calibrated color ranges
    color_ranges = load_color_calibration()
    
    # Get user's color priority choices
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    # Offer calibration for the highest-priority color
    if color_priorities[0] != 'black':  # Skip calibration for black if it's highest priority
        calibrate_color(picam2, color_ranges, color_priorities[0])
    else:
        calibrate_color(picam2, color_ranges, 'black')
    
    print("Line follower with color detection started. Press 'q' in the display window or Ctrl+C to stop.")
    
    # Create a larger display window
    cv2.namedWindow("Color Line Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color Line Detection", 800, 600)
    
    # Flag to track if we're in recovery mode
    recovery_mode = False
    tight_turn = False
    
    try:
        while True:
            frame = picam2.capture_array()
            if frame is not None:
                # Ensure the frame is in BGR format for OpenCV
                if frame.shape[-1] == 4:  # If frame has an alpha channel (RGBA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[-1] != 3:  # If frame is not in a recognizable format
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect line using the new function
            error, line_found, detected_color, all_available_colors, line_angle, (best_cx, best_cy), available_colors_text, text_color, (h, s, v) = detect_line(
                frame, color_priorities, color_ranges, tight_turn=tight_turn)
            
            # Determine if it's a tight turn (based on top angle)
            tight_turn = abs(line_angle) > 20  # Threshold for tight turn detection
            
            # Display information
            height, width = frame.shape[:2]
            
            # Draw fitted line if a line is detected
            if line_found:
                rows, cols = frame.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(np.array([[best_cx, best_cy]]), cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                cv2.line(frame, (cols-1, righty), (0, lefty), (255, 0, 255), 2)
                
                # Calculate servo angle
                if tight_turn:
                    servo_angle = 90 - (line_angle * 3)  # More aggressive adjustment
                else:
                    servo_angle = 90 - (line_angle * 2)  # Normal adjustment
                servo_angle = max(0, min(180, servo_angle))
            else:
                servo_angle = 90
            
            # Display error in large red text at top-left corner
            cv2.putText(frame, f"Error: {error}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Display additional information
            if line_found:
                detection_text = f"Detected: {detected_color}"
                command_text = "Command: "
                if error < -CENTER_THRESHOLD:
                    command_text += "Turn Left"
                elif error > CENTER_THRESHOLD:
                    command_text += "Turn Right"
                else:
                    command_text += "Move Forward"
                angle_text = f"Line Angle (Top): {line_angle:.2f}°"
                servo_text = f"Servo Angle: {servo_angle:.2f}°"
                
                cv2.putText(frame, f"{detected_color.capitalize()} Line", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                cv2.putText(frame, f"HSV: ({h}, {s}, {v})", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            else:
                detection_text = "Detected: None"
                command_text = "Command: No line detected"
                angle_text = f"Line Angle (Top): 0.00°"
                servo_text = f"Servo Angle: 90.00°"
            
            # Put info text in the bottom section with yellow color
            y_start = height - 120
            line_height = 25
            
            cv2.putText(frame, detection_text, (20, y_start), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, command_text, (20, y_start + line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, angle_text, (20, y_start + 2*line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, servo_text, (20, y_start + 3*line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display available colors
            cv2.putText(frame, available_colors_text, (width-200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Color Line Detection", frame)
            
            # Print detection status
            if line_found:
                recovery_mode = False
                if error > TURN_THRESHOLD:
                    print(f"Pivot Turning Right - {detected_color} line")
                elif error < -TURN_THRESHOLD:
                    print(f"Pivot Turning Left - {detected_color} line")
                else:
                    if tight_turn:
                        if line_angle > 0:
                            print(f"Tight Turn Right - {detected_color} line")
                        else:
                            print(f"Tight Turn Left - {detected_color} line")
                    else:
                        print(f"Moving Forward - {detected_color} line (Available: {', '.join(all_available_colors)})")
            else:
                if not recovery_mode:
                    print("No line detected. Starting recovery...")
                    recovery_mode = True
                print("Reversing to find line...")
                time.sleep(0.1)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Allow recalibration during runtime for the highest-priority color
                if color_priorities[0] != 'black':
                    calibrate_color(picam2, color_ranges, color_priorities[0])
                else:
                    calibrate_color(picam2, color_ranges, 'black')
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        cv2.destroyAllWindows()
        GPIO.cleanup()
        picam2.stop()
        print("Resources released")

if __name__ == "__main__":
    main()