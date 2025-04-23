import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import json
import os
from picamera2 import Picamera2

# Define GPIO pins for encoders
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Line detection parameters
MIN_CONTOUR_AREA = 800     # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# ROI parameters
USE_TOP_ROI = True         # Enable top ROI for line angle determination
TOP_ROI_HEIGHT = 150       # Height of the top ROI from the top of the frame

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Calibration file
CALIBRATION_FILE = "color_calibration.json"

# Default color ranges (HSV format)
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
            
        seen = set()
        unique_choices = []
        for c in choices:
            if c in color_map and c not in seen:
                seen.add(c)
                unique_choices.append(c)
                
        if unique_choices:
            selected_colors = [color_map[c] for c in unique_choices]
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

# GPIO Setup for encoders
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

# Function to allow user to calibrate black line detection parameters
def calibrate_black_line(picam2, color_ranges):
    print("\nCalibrating black line detection...")
    print("Place the camera to view the black line and press 'c' to capture and calibrate.")
    print("Press 'q' to skip calibration.")
    
    while True:
        frame = picam2.capture_array()
        
        if USE_TOP_ROI:
            roi_y_end = TOP_ROI_HEIGHT
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, roi_y_end), (255, 255, 0), 2)
            roi = frame[0:roi_y_end, 0:FRAME_WIDTH]
        else:
            roi = frame
            
        cv2.putText(frame, "Press 'c' to calibrate black line", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyWindow("Calibration")
            return
            
        if key == ord('c'):
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            black_mask = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 100, 80]))
            contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                black_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(black_contour) > MIN_CONTOUR_AREA:
                    mask = np.zeros_like(black_mask)
                    cv2.drawContours(mask, [black_contour], -1, 255, -1)
                    roi_pixels = hsv_roi[mask == 255]
                    if len(roi_pixels) > 0:
                        h_min, s_min, v_min = np.min(roi_pixels, axis=0)
                        h_max, s_max, v_max = np.max(roi_pixels, axis=0)
                        h_min = max(0, h_min - 10)
                        s_min = max(0, s_min - 10)
                        v_min = max(0, v_min - 10)
                        h_max = min(179, h_max + 10)
                        s_max = min(255, s_max + 40)
                        v_max = min(255, v_max + 40)
                        color_ranges['black'] = [([h_min, s_min, v_min], [h_max, s_max, v_max])]
                        print(f"Updated black line HSV range: ({h_min}, {s_min}, {v_min}) to ({h_max}, {s_max}, {v_max})")
                        calibrated_mask = cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                        cv2.imshow("Calibrated Black Line Mask", calibrated_mask)
                        cv2.waitKey(2000)
                        cv2.destroyWindow("Calibrated Black Line Mask")
                        break
                    else:
                        print("Could not find black pixels in the contour. Please try again.")
                else:
                    print("Black line contour too small. Please try again.")
            else:
                print("No black line detected. Please try again.")
                
    cv2.destroyWindow("Calibration")

# Refined line detection function with angle determination
# Refined line detection function with angle determination
def detect_line(frame, color_priorities, color_ranges):
    if USE_TOP_ROI:
        roi_y_end = TOP_ROI_HEIGHT
        roi = frame[0:roi_y_end, 0:FRAME_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    if USE_TOP_ROI:
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, roi_y_end), (255, 255, 0), 2)

    best_contour = None
    best_color = None
    best_cx, best_cy = -1, -1
    max_area = 0
    line_angle = 0

    valid_contours = {}
    all_available_colors = []

    for color_name in color_priorities:
        color_ranges_for_color = color_ranges.get(color_name, [])
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in color_ranges_for_color:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
        
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        if color_name == 'black':
            cv2.imshow(f"{color_name} Mask", color_mask)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

        if valid:
            valid_contours[color_name] = valid
            all_available_colors.append(color_name)

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
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            if USE_TOP_ROI:
                best_cx = cx
                best_cy = cy
            else:
                best_cx = cx
                best_cy = cy

            contour_color = (0, 255, 0) if best_color != 'black' else (128, 128, 128)
            
            if USE_TOP_ROI:
                cv2.drawContours(frame[0:TOP_ROI_HEIGHT, 0:FRAME_WIDTH], 
                               [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
            else:
                cv2.drawContours(frame, [best_contour], -1, contour_color, 2)
                cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
                
            cv2.line(frame, (center_x, best_cy), (best_cx, best_cy), (255, 0, 0), 2)

            error = best_cx - center_x

            # Calculate line angle using contour points
            [vx, vy, x, y] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            line_angle = np.arctan2(vy, vx) * 180 / np.pi

            if USE_TOP_ROI:
                hsv_value = hsv[cy, cx]
            else:
                hsv_value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[best_cy, best_cx]
                
            h, s, v = hsv_value

            available_colors_text = "Available: " + ", ".join(all_available_colors)
            cv2.putText(frame, available_colors_text, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if best_color == 'red':
                text_color = (0, 0, 255)
            elif best_color == 'green':
                text_color = (0, 255, 0)
            elif best_color == 'blue':
                text_color = (255, 0, 0)
            elif best_color == 'yellow':
                text_color = (0, 255, 255)
            elif best_color == 'black':
                text_color = (128, 128, 128)
            else:
                text_color = (255, 255, 255)
                
            cv2.putText(frame, f"{best_color.capitalize()} Line, Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(frame, f"HSV: ({h}, {s}, {v})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Angle: {line_angle:.2f} deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            return error, True, best_color, all_available_colors

    return 0, False, None, []

# Main function
def main():
    setup_gpio()
    picam2 = setup_camera()
    
    color_ranges = load_color_calibration()
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    calibrate_black_line(picam2, color_ranges)
    
    print("Line detection with color and angle detection started. Press 'q' in the display window or Ctrl+C to stop.")
    
    try:
        while True:
            frame = picam2.capture_array()
            error, line_found, detected_color, available_colors = detect_line(frame, color_priorities, color_ranges)
            
            cv2.imshow("Line Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                calibrate_black_line(picam2, color_ranges)
            
            if line_found:
                print(f"Detected {detected_color} line (Available: {', '.join(available_colors)})")
            else:
                print("No line detected.")
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()