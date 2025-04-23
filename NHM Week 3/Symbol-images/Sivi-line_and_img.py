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

# Shape detection parameters
MIN_SHAPE_AREA = 500       # Minimum area for shape contours
CONTOUR_MARGIN = 5         # Margin from frame edges for complete contours

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

# Load calibrated color ranges for line detection
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

# Get user's color priority choice for line following
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

# Check if contour is complete (not touching frame edges)
def is_contour_complete(contour, frame_width, frame_height, margin=CONTOUR_MARGIN):
    x, y, w, h = cv2.boundingRect(contour)
    return (x > margin and
            y > margin and
            x + w < frame_width - margin and
            y + h < frame_height - margin)

# Preprocess and store reference images
def preprocess_reference_images():
    reference_shapes = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("\nPreprocessing reference images...")
    
    for filename in os.listdir(script_dir):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(script_dir, filename)
            print(f"Processing {filename}...", end=' ')
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                print("Failed to load image")
                continue
                
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("No contours found")
                continue
                
            # Get largest contour
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < MIN_SHAPE_AREA:
                print("Contour too small")
                continue
                
            # Check if contour is complete
            if not is_contour_complete(main_contour, img.shape[1], img.shape[0]):
                print("Contour touches edges")
                continue
                
            # Calculate shape features
            shape_type = classify_shape(main_contour)
            if shape_type is None:
                print("Unknown shape")
                continue
                
            # Store contour data
            reference_shapes[filename] = {
                'contour': main_contour,
                'shape': shape_type,
                'aspect_ratio': cv2.boundingRect(main_contour)[2] / cv2.boundingRect(main_contour)[3],
                'area': cv2.contourArea(main_contour)
            }
            print(f"Stored as {shape_type} (AR: {reference_shapes[filename]['aspect_ratio']:.2f})")
    
    print(f"\nPreprocessing complete. Loaded {len(reference_shapes)} reference shapes.")
    return reference_shapes

# Classify shape based on contour approximation
def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    num_sides = len(approx)
    
    if num_sides == 3:
        return "triangle"
    elif num_sides == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if 0.9 <= aspect_ratio <= 1.1:
            return "square"
        else:
            return "rectangle"
    elif num_sides == 5:
        return "pentagon"
    elif num_sides >= 6:
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        if area / circle_area > 0.8:
            return "circle"
    return None

# Match detected contour with reference shapes
def match_contour(contour, reference_shapes):
    best_match = None
    best_score = float('inf')
    
    for name, ref_data in reference_shapes.items():
        # Compare using shape matching score
        score = cv2.matchShapes(contour, ref_data['contour'], cv2.CONTOURS_MATCH_I1, 0)
        
        # Compare aspect ratios
        x, y, w, h = cv2.boundingRect(contour)
        current_ar = w/h
        ar_diff = abs(current_ar - ref_data['aspect_ratio'])
        
        # Combined score (weighted)
        combined_score = score * 0.7 + ar_diff * 0.3
        
        if combined_score < best_score and combined_score < 0.5:  # Threshold
            best_score = combined_score
            best_match = (name, ref_data['shape'])
    
    return best_match, best_score

# Line Detection Function
def detect_line(frame, color_priorities, color_ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Preprocess for contour detection (Line Threshold)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
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
            
            return error, True, best_color, all_available_colors, frame, line_y_top, line_y_bottom, thresh
    return 0, False, None, [], frame, line_y_top, line_y_bottom, thresh

# Combined Detection and Line Following with Shape Detection
def detect_images_shapes_and_line(frame, prev_detections, reference_shapes, color_priorities, color_ranges, right_pwm, left_pwm, pause_state, max_len=5):
    # Line detection (includes preprocessing for Line Threshold)
    error, line_found, detected_color, available_colors, _, line_y_top, line_y_bottom, thresh = detect_line(frame, color_priorities, color_ranges)
    
    detected_name = None
    label = "Detected: None"
    
    # Convert thresh to 3-channel image for color overlays
    thresh_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Only perform shape detection if line is found
    if line_found:
        # Create a copy of the threshold image and exclude the line region
        shape_thresh = thresh.copy()
        shape_thresh[line_y_top:line_y_bottom, :] = 0  # Set the line region to black
        
        # Draw ROI visualization (excluding the line region) on the live feed
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, line_y_top), (0, 255, 255), 2)
        cv2.rectangle(frame, (0, line_y_bottom), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 255), 2)
        
        # Find contours in the modified threshold image
        contours, _ = cv2.findContours(shape_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_match = None
        best_score = float('inf')
        
        # Process each contour
        for contour in contours:
            if cv2.contourArea(contour) < MIN_SHAPE_AREA:
                continue
                
            # Check if contour is complete
            if not is_contour_complete(contour, FRAME_WIDTH, FRAME_HEIGHT):
                continue
                
            # Match with reference shapes
            match, score = match_contour(contour, reference_shapes)
            if match and score < best_score:
                best_contour = contour
                best_match = match
                best_score = score
        
        # If a valid shape is detected, overlay it on both the RGB frame and Line Threshold
        if best_contour is not None:
            filename, shape_type = best_match
            # Draw the contour on the RGB frame
            cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), 2)
            # Calculate bounding box for text on RGB frame
            x, y, w, h = cv2.boundingRect(best_contour)
            detected_name = f"{shape_type.capitalize()}"
            cv2.putText(frame, detected_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            label = f"Detected: {detected_name} ({os.path.splitext(filename)[0]})"
            
            # Draw the contour and label on the Line Threshold display
            cv2.drawContours(thresh_display, [best_contour], -1, (0, 255, 0), 2)
            cv2.putText(thresh_display, detected_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            print(f"Detected {detected_name} from {filename} (Score: {best_score:.3f})")
    
    # Show the updated Line Threshold window with shape detection
    cv2.imshow("Line Threshold", thresh_display)
    
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
    # Preprocess reference images first
    reference_shapes = preprocess_reference_images()
    if not reference_shapes:
        print("No valid reference images found. Please add PNG images to the script directory.")
        return
    
    # Initialize hardware
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
    
    prev_detections = deque()
    pause_state = {'active': False, 'start_time': 0, 'detected_object': None}
    
    print("\nCombined shape detection and line follower started. Press 'q' to stop.")
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
                display_frame, prev_detections, reference_shapes, color_priorities, color_ranges, right_pwm, left_pwm, pause_state)
            
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