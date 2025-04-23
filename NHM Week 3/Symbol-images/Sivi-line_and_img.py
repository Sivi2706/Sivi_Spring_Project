import cv2
import numpy as np
import os
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
SHAPE_ROI_WIDTH = 200      # Width of the ROI for shape detection (centered)
ASPECT_RATIO_TOLERANCE = 0.2  # Tolerance for aspect ratio matching
DEFAULT_SHAPE_ROI_HEIGHT = FRAME_HEIGHT // 2  # Default height if no line detected

# Variables for encoder counts
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

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Allow camera to warm up
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

def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    left_pwm.ChangeDutyCycle(BASE_SPEED)

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Load calibrated color ranges
def load_color_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                loaded_ranges = json.load(f)
                print("Loaded calibrated color ranges from file.")
                return loaded_ranges
        except Exception as e:
            print(f"Error loading calibration file: {e}")
    print("Using default color ranges.")
    return default_color_ranges

# Get user's color priority choice
def get_color_choices():
    print("\nAvailable line colors to follow (priority order):")
    print("r = red (highest priority)")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black (lowest priority)")
    print("q = quit program")
    
    color_map = {'r': 'red', 'b': 'blue', 'g': 'green', 'y': 'yellow', 'k': 'black'}
    while True:
        choices = input("\nEnter color priorities (e.g., 'rbk'): ").lower()
        if choices == 'q':
            return None
        seen = set()
        unique_choices = [c for c in choices if c in color_map and c not in seen and not seen.add(c)]
        if unique_choices:
            selected_colors = [color_map[c] for c in unique_choices]
            if 'black' not in selected_colors:
                selected_colors.append('black')
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        print("Invalid choice. Please try again.")

# Preprocess and store reference images and shapes
def preprocess_reference_data():
    reference_data = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in os.listdir(script_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(script_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Preprocess image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(thresh, 30, 200)
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # Get largest contour
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < 500:
                continue
                
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = float(w) / h if h != 0 else float('inf')
            
            # Create mask for color histogram
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [main_contour], -1, 255, -1)
            
            # Calculate color histogram in HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            
            # Store contour, histogram, and aspect ratio
            reference_data[filename] = {
                'contour': main_contour,
                'color_hist': hist,
                'image': img,
                'aspect_ratio': aspect_ratio
            }
    
    return reference_data

# Compare contours using Hu Moments
def compare_contours(c1, c2):
    if c1 is None or c2 is None:
        return float('inf')
    m1 = cv2.moments(c1)
    m2 = cv2.moments(c2)
    if m1['m00'] == 0 or m2['m00'] == 0:
        return float('inf')
    hu1 = cv2.HuMoments(m1)
    hu2 = cv2.HuMoments(m2)
    return cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)

# Compare color histograms
def compare_color_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

# Check if contour is complete (not touching frame edges)
def is_contour_complete(contour, frame_width, frame_height, margin=5):
    x, y, w, h = cv2.boundingRect(contour)
    return (x > margin and
            y > margin and
            x + w < frame_width - margin and
            y + h < frame_height - margin)

# Line Detection Function (No ROI)
# Line Detection Function (No ROI)
def detect_line(frame, color_priorities, color_ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Preprocess for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Line Threshold", thresh)
    
    best_contour = None
    best_color = None
    best_cx, best_cy = -1, -1
    max_area = 0
    valid_contours = {}
    all_available_colors = []
    line_y_top, line_y_bottom = 0, FRAME_HEIGHT  # Default values if no line detected
    
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
    
    # Create an image to show all line contours
    line_contour_display = np.zeros_like(frame)
    for color_name in valid_contours:
        for cnt in valid_contours[color_name]:
            color = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'black': (128, 128, 128)
            }.get(color_name, (255, 255, 255))
            cv2.drawContours(line_contour_display, [cnt], -1, color, 2)
    cv2.imshow("Line Contours", line_contour_display)
    
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
            best_cx = int(M["m10"] / M["m00"])
            best_cy = int(M["m01"] / M["m00"])
            # Get bounding box for line to determine vertical extent
            x, y, w, h = cv2.boundingRect(best_contour)
            line_y_top = max(0, y - 10)  # Add small buffer above
            line_y_bottom = min(FRAME_HEIGHT, y + h + 10)  # Add small buffer below
            contour_color = (0, 255, 0) if best_color != 'black' else (128, 128, 128)
            cv2.drawContours(frame, [best_contour], -1, contour_color, 2)
            cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
            cv2.line(frame, (center_x, best_cy), (best_cx, best_cy), (255, 0, 0), 2)
            error = best_cx - center_x
            hsv_value = hsv[best_cy, best_cx]
            h, s, v = hsv_value
            available_colors_text = "Available: " + ", ".join(all_available_colors)
            cv2.putText(frame, available_colors_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_color = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'black': (128, 128, 128)
            }.get(best_color, (255, 255, 255))
            cv2.putText(frame, f"{best_color.capitalize()} Line, Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(frame, f"HSV: ({h}, {s}, {v})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Show best line contour separately
            best_line_contour_display = np.zeros_like(frame)
            cv2.drawContours(best_line_contour_display, [best_contour], -1, contour_color, 2)
            cv2.imshow("Best Line Contour", best_line_contour_display)
            
            return error, True, best_color, all_available_colors, frame, line_y_top, line_y_bottom, thresh
    return 0, False, None, [], frame, line_y_top, line_y_bottom, thresh


# Combined Detection and Line Following with Contour and Color Validation
# Combined Detection and Line Following with Contour and Color Validation
# Combined Detection and Line Following with Contour and Color Validation
# Combined Detection and Line Following with Contour and Color Validation

# Combined Detection and Line Following with Contour and Color Validation
def detect_images_shapes_and_line(frame, prev_detections, reference_data, color_priorities, color_ranges, right_pwm, left_pwm, pause_state, max_len=5):
    # Line detection
    error, line_found, detected_color, available_colors, _, line_y_top, line_y_bottom, thresh = detect_line(frame, color_priorities, color_ranges)
    
    detected_name = None
    label = "Detected: None"
    
    # Only perform shape/image detection if line is found
    if line_found:
        # Create a mask to exclude the line region from the threshold image
        shape_thresh = thresh.copy()
        shape_thresh[line_y_top:line_y_bottom, :] = 0  # Set the line region to black
        
        # Draw ROI visualization (excluding the line region)
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, line_y_top), (0, 255, 255), 2)
        cv2.rectangle(frame, (0, line_y_bottom), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 255), 2)
        
        # Find contours in the modified threshold image
        contours, _ = cv2.findContours(shape_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an image to show all shape contours (full frame, no quadrants)
        shape_contour_display = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) >= 500:
                cv2.drawContours(shape_contour_display, [contour], -1, (0, 255, 0), 2)
        cv2.imshow("Shape Contours", shape_contour_display)
        
        best_match = None
        best_score = float('inf')
        best_contour = None
        
        # Compare contours with reference data
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
                
            # Check if contour is complete
            if not is_contour_complete(contour, FRAME_WIDTH, FRAME_HEIGHT):
                continue
                
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else float('inf')
            
            for name, data in reference_data.items():
                # Check aspect ratio
                ref_aspect_ratio = data['aspect_ratio']
                if abs(aspect_ratio - ref_aspect_ratio) / ref_aspect_ratio > ASPECT_RATIO_TOLERANCE:
                    continue
                
                score = compare_contours(contour, data['contour'])
                if score < best_score and score < 0.5:  # Threshold for contour similarity
                    best_score = score
                    best_match = name
                    best_contour = contour
        
        # Show best contour even if color validation fails (for debugging)
        if best_contour is not None:
            best_shape_contour_display = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.drawContours(best_shape_contour_display, [best_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Best Shape Contour", best_shape_contour_display)
        
        # Validate color if we have a contour match
        if best_match and best_contour is not None:
            # Create mask for current contour using the full frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [best_contour], -1, 255, -1)
            
            # Calculate color histogram for current contour
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            
            # Compare with reference histogram
            color_score = compare_color_histograms(hist, reference_data[best_match]['color_hist'])
            
            # Validate match based on color similarity (increased threshold)
            if color_score < 100:  # Increased threshold for color histogram similarity
                detected_name = best_match
                # Draw contour on the frame
                cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), 2)
                # Calculate bounding box for text
                x, y, w, h = cv2.boundingRect(best_contour)
                cv2.putText(frame, detected_name, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                label = f"Detected: {detected_name}"
                print(f"Confirmed {detected_name} with contour score: {best_score:.3f}, color score: {color_score:.3f}")
            else:
                print(f"Color validation failed for {best_match}. Score: {color_score:.3f}, Contour score: {best_score:.3f}")
                detected_name = None
                label = "Detected: None"
    
    # Update detection history
    prev_detections.append(detected_name)
    if len(prev_detections) > max_len:
        prev_detections.popleft()
    
    valid_detections = [d for d in prev_detections if d is not None]
    
    # Pause logic
    should_pause = False
    if valid_detections and not pause_state['active']:
        confirmed_name = max(set(valid_detections), key=valid_detections.count)
        should_pause = True
        pause_state['active'] = True
        pause_state['start_time'] = time.time()
        pause_state['detected_object'] = confirmed_name
        stop_motors(right_pwm, left_pwm)
        print(f"Confirmed detection: {confirmed_name}. Pausing for 5 seconds.")
        
        cv2.rectangle(frame, (5, 45), (400, 85), (0, 0, 0), -1)
        cv2.putText(frame, "Paused: 5.0s remaining", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Handle pause state
    if pause_state['active']:
        elapsed = time.time() - pause_state['start_time']
        remaining = max(5.0 - elapsed, 0)
        
        cv2.rectangle(frame, (5, 45), (400, 85), (0, 0, 0), -1)
        cv2.putText(frame, f"Paused: {remaining:.1f}s remaining", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if remaining <= 0:
            pause_state['active'] = False
            print(f"Resume line following after {pause_state['detected_object']} detection")
    
    # Line following
    if not pause_state['active']:
        if line_found:
            if error > TURN_THRESHOLD:
                pivot_turn_right(right_pwm, left_pwm)
                print(f"Pivot Turning Right - {detected_color} line")
            elif error < -TURN_THRESHOLD:
                pivot_turn_left(right_pwm, left_pwm)
                print(f"Pivot Turning Left - {detected_color} line")
            else:
                move_forward(right_pwm, left_pwm)
                print(f"Moving Forward - {detected_color} line (Available: {', '.join(available_colors)})")
        else:
            move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            print("Reversing to find line...")
    
    cv2.rectangle(frame, (5, 5), (400, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, detected_name, error, line_found, detected_color, available_colors, pause_state

# Main function
def main():
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return
    
    right_pwm, left_pwm = setup_gpio()
    color_ranges = load_color_calibration()
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    reference_data = preprocess_reference_data()
    prev_detections = deque()
    
    pause_state = {'active': False, 'start_time': 0, 'detected_object': None}
    
    print("Combined image/shape detection and line follower started. Press 'q' to stop.")
    try:
        while True:
            frame = picam2.capture_array()
            
            if len(frame.shape) == 2:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                display_frame = frame.copy()
            
            output_frame, detected_name, error, line_found, detected_color, available_colors, pause_state = detect_images_shapes_and_line(
                display_frame, prev_detections, reference_data, color_priorities, color_ranges, right_pwm, left_pwm, pause_state)
            
            cv2.imshow("Combined Detection and Line Follower", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()