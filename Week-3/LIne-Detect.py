import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5       # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5      # Duty cycle for 180 degrees
SERVO_FREQ = 50            # 50Hz frequency for servo

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60            # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 800     # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100        # Error threshold for pivoting

# Recovery parameters
REVERSE_DURATION = 0.5     # Seconds to reverse
REVERSE_SPEED = 40         # Speed when reversing

# Updated scanning angles: center at 90, right at 45, left at 135.
SCAN_ANGLES = [90, 45, 135]
SCAN_TIME_PER_ANGLE = 0.5   # Seconds to wait per scan angle

# Define all available color ranges (HSV format)
all_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),
        ([114, 167, 154], [134, 247, 234])
    ],
    'blue': [
        ([6, 167, 60], [26, 255, 95])
    ],
    'green': [
        ([31, 180, 110], [51, 255, 190])
    ],
    'yellow': [
        ([84, 155, 189], [104, 235, 255])
    ],
    'black': [
        ([0, 0, 0], [179, 78, 50])
    ]
}

# Variables to store encoder counts
right_counter = 0
left_counter = 0

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
    
    # Set up PWM for servo
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

# Function to set servo angle
def set_servo_angle(servo_pwm, angle):
    # Constrain angle
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Allow time for movement
    servo_pwm.ChangeDutyCycle(0)

def map_line_angle_to_servo_angle(line_angle):
    """
    Map line angle (-45° to +45°) to servo angle (180° to 0°)
    -45° (left) -> 180°, 0° (straight) -> 90°, +45° (right) -> 0°
    """
    # Normalize line angle to [-45, 45]
    normalized_angle = max(-45, min(45, line_angle))
    # Linear mapping: -45° -> 180°, 0° -> 90°, +45° -> 0°
    servo_angle = 90 - (normalized_angle * 4)  # Scale: 45° -> 180°
    return max(0, min(180, servo_angle))

# Turn function based on scanned angle
def turn_with_scanned_angle(scanned_angle, servo_pwm, right_pwm, left_pwm):
    # Calculate turn time: assume 45° turn takes 1 second
    turn_time = abs(scanned_angle - 90) / 45.0
    if scanned_angle > 90:
        print(f"Detected angle {scanned_angle}: Pivoting LEFT for {turn_time:.2f} seconds")
        # For left pivot: right wheel forward, left wheel backward
        GPIO.output(IN1, GPIO.LOW)    # Left backward
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)    # Right forward
        GPIO.output(IN4, GPIO.HIGH)
        right_pwm.ChangeDutyCycle(TURN_SPEED)
        left_pwm.ChangeDutyCycle(TURN_SPEED)
    elif scanned_angle < 90:
        print(f"Detected angle {scanned_angle}: Pivoting RIGHT for {turn_time:.2f} seconds")
        # For right pivot: left wheel forward, right wheel backward
        GPIO.output(IN1, GPIO.HIGH)   # Left forward
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)   # Right backward
        GPIO.output(IN4, GPIO.LOW)
        right_pwm.ChangeDutyCycle(TURN_SPEED)
        left_pwm.ChangeDutyCycle(TURN_SPEED)
    else:
        print("Detected angle 90: No pivot required.")
        return

    time.sleep(turn_time)
    stop_motors(right_pwm, left_pwm)
    # Reset the servo to center
    print("Resetting servo to 90 degrees")
    set_servo_angle(servo_pwm, 90)

# Motor control functions
def pivot_turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)    # Right forward
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

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

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
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        else:
            print("Invalid choice. Please try again.")

def detect_priority_color(frame, color_names, roi_type='bottom'):
    """
    Detect colors in priority order within specified ROI
    Bottom ROI (30%) for motor control, Top ROI (70%) for servo
    Returns contour, color, and angle
    """
    height, width = frame.shape[:2]
    if roi_type == 'bottom':
        roi_height = int(height * 0.3)  # Bottom 30%
        roi = frame[height - roi_height:height, :]
        y_offset = height - roi_height
    else:  # top
        roi_height = int(height * 0.7)  # Top 70%
        roi = frame[0:roi_height, :]
        y_offset = 0
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    MIN_AREA = 800  # Minimum area threshold
    
    # Edge detection to enhance contours
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to make them more prominent
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Check for intersections (multiple contours)
    intersection = False
    
    for color_name in color_names:
        color_ranges = all_color_ranges.get(color_name, [])
        
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # Combine color mask with edge detection for better results
            enhanced_mask = cv2.bitwise_and(mask, dilated_edges)
            if cv2.countNonZero(enhanced_mask) < 50:  # If not enough overlap, use original mask
                enhanced_mask = mask
            
            # Apply morphological operations
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel)
            enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
            enhanced_mask = cv2.GaussianBlur(enhanced_mask, (5, 5), 0)
            
            contours, _ = cv2.findContours(enhanced_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for intersections - valid contours that are not too close to each other
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
            if len(valid_contours) >= 2:
                intersection = True
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_AREA:
                    # Adjust contour coordinates to original frame
                    adjusted_contour = largest_contour.copy()
                    adjusted_contour[:, :, 1] += y_offset
                    
                    # Get angle for steering
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]
                    if angle < -45:
                        angle += 90
                    
                    return adjusted_contour, color_name, angle, intersection
    
    return None, None, 0, intersection

def create_display_frame(frame, color_priority, contour_bottom, color_name_bottom, line_angle_bottom, 
                         contour_top, color_name_top, line_angle_top, error):
    """
    Create a display frame with all visual elements for monitoring
    """
    display_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Define ROIs
    top_roi_height = int(height * 0.7)
    bottom_roi_height = int(height * 0.3)
    
    # Draw ROIs on the frame
    # Bottom ROI (red rectangle)
    cv2.rectangle(display_frame, (0, height - bottom_roi_height), (width, height), (0, 0, 255), 2)
    
    # Top ROI (blue rectangle)
    cv2.rectangle(display_frame, (0, 0), (width, top_roi_height), (255, 0, 0), 2)
    
    # Draw center line (red vertical line)
    center_x = width // 2
    cv2.line(display_frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
    
    # Draw horizontal yellow dividing line
    cv2.line(display_frame, (0, top_roi_height), (width, top_roi_height), (0, 255, 255), 2)
    
    movement = "No line detected"
    current_color = "None"
    servo_angle = 90
    
    # Color to BGR mapping
    color_map = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'black': (0, 0, 0)
    }
    
    # Process bottom ROI detection
    if contour_bottom is not None:
        # Draw contour outline in green
        cv2.drawContours(display_frame, [contour_bottom], -1, (0, 255, 0), 2)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour_bottom)
        current_color = color_name_bottom
        
        # Calculate error (distance from center)
        line_center = x + w // 2
        frame_center = width // 2
        error = line_center - frame_center
        
        # Draw a blue dot at contour center point with line to center
        cv2.circle(display_frame, (line_center, y + h//2), 5, (255, 0, 0), -1)
        cv2.line(display_frame, (frame_center, y + h//2), (line_center, y + h//2), (255, 0, 0), 2)
        
        # Determine movement direction
        CENTER_THRESHOLD = 20
        if error < -CENTER_THRESHOLD:
            movement = "Turn Left"
        elif error > CENTER_THRESHOLD:
            movement = "Turn Right"
        else:
            movement = "Move Forward"
        
        # Label bottom ROI contour
        cv2.putText(display_frame, f"{color_name_bottom}", (x, y - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[color_name_bottom], 2)
    
    # Process top ROI detection
    if contour_top is not None:
        # Draw contour outline in green
        cv2.drawContours(display_frame, [contour_top], -1, (0, 255, 0), 2)
        
        # Get the center of the contour
        M = cv2.moments(contour_top)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Label top ROI contour
            cv2.putText(display_frame, f"{color_name_top}", (cx, cy - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[color_name_top], 2)
        
        # Calculate servo angle based on top ROI
        servo_angle = map_line_angle_to_servo_angle(line_angle_top)
    
    # Display error in large red text at top-left corner
    cv2.putText(display_frame, f"Error: {error}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Display information at the bottom in yellow text
    detection_text = f"Detected: {current_color}"
    command_text = f"Command: {movement}"
    angle_text = f"Line Angle (Top): {line_angle_top:.2f}°"
    servo_text = f"Servo Angle: {servo_angle:.2f}°"
    
    # Put info text in the bottom section with yellow color
    y_start = height - 120
    line_height = 25
    
    cv2.putText(display_frame, detection_text, (20, y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, command_text, (20, y_start + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, angle_text, (20, y_start + 2*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, servo_text, (20, y_start + 3*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return display_frame, error, movement, servo_angle

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    print("===== Color Line Detection and Following =====")
    
    # Get color priorities from user
    color_priority = get_color_choices()
    if not color_priority:
        return
    
    # Center the servo initially
    set_servo_angle(servo_pwm, 90)
    
    # State variables
    state = "NORMAL"
    reverse_start_time = 0
    current_scan_index = 0
    scan_start_time = 0
    detected_scan_angle = None
    error = 0
    
    print(f"\nColor priority: {' > '.join(color_priority)}")
    print("Line follower started. Press 'q' to quit or 'c' to change colors")

    try:
        # Create a larger display window
        cv2.namedWindow("Color Line Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color Line Detection", 800, 600)
        
        while True:
            frame = picam2.capture_array()
            
            # Bottom ROI for motor control (30% of height)
            contour_bottom, color_name_bottom, line_angle_bottom, intersection_bottom = detect_priority_color(
                frame, color_priority, roi_type='bottom')
            
            # Top ROI for angle visualization (70% of height)
            contour_top, color_name_top, line_angle_top, intersection_top = detect_priority_color(
                frame, color_priority, roi_type='top')
            
            # Create display frame with all visual elements
            display_frame, error, movement, servo_angle = create_display_frame(
                frame, color_priority, contour_bottom, color_name_bottom, line_angle_bottom,
                contour_top, color_name_top, line_angle_top, error)
            
            # Show the frame
            cv2.imshow("Color Line Detection", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                new_priority = get_color_choices()
                if new_priority:
                    color_priority = new_priority
                    print(f"New priority: {' > '.join(color_priority)}")
            
            # ROBOT STATE MACHINE
            if state == "NORMAL":
                # Set servo angle based on line angle from top ROI
                if contour_top is not None:
                    target_angle = map_line_angle_to_servo_angle(line_angle_top)
                    set_servo_angle(servo_pwm, target_angle)
                
                # Control motors based on bottom ROI
                if contour_bottom is not None:
                    # Handle intersections
                    if intersection_bottom:
                        print("Intersection detected. Continuing straight.")
                        move_forward(right_pwm, left_pwm)
                    # Handle line following
                    elif abs(error) > TURN_THRESHOLD:
                        if error > 0:
                            pivot_turn_right(right_pwm, left_pwm)
                            print("Pivot Turning Right")
                        else:
                            pivot_turn_left(right_pwm, left_pwm)
                            print("Pivot Turning Left")
                    else:
                        move_forward(right_pwm, left_pwm)
                        print("Moving Forward")
                else:
                    print("Line lost. Reversing...")
                    state = "REVERSING"
                    reverse_start_time = time.time()
                    move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            
            elif state == "REVERSING":
                if time.time() - reverse_start_time >= REVERSE_DURATION:
                    stop_motors(right_pwm, left_pwm)
                    print("Beginning scan for line...")
                    state = "SCANNING"
                    current_scan_index = 0
                    # Set servo to first scan angle
                    set_servo_angle(servo_pwm, SCAN_ANGLES[current_scan_index])
                    scan_start_time = time.time()
            
            elif state == "SCANNING":
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    frame = picam2.capture_array()
                    contour_scan, color_name_scan, _, intersection_scan = detect_priority_color(
                        frame, color_priority, roi_type='bottom')
                    
                    # Check if an intersection is detected
                    if intersection_scan:
                        print("Intersection detected. Centering servo to 90° and adjusting.")
                        set_servo_angle(servo_pwm, 90)
                        state = "NORMAL"
                    elif contour_scan is not None:
                        detected_scan_angle = SCAN_ANGLES[current_scan_index]
                        print(f"Line detected during scan at servo angle: {detected_scan_angle}")
                        state = "TURNING"
                    else:
                        current_scan_index += 1
                        if current_scan_index < len(SCAN_ANGLES):
                            set_servo_angle(servo_pwm, SCAN_ANGLES[current_scan_index])
                            scan_start_time = time.time()
                        else:
                            print("No line found during scan. Reversing again...")
                            state = "REVERSING"
                            move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                            reverse_start_time = time.time()
            
            elif state == "TURNING":
                if detected_scan_angle is not None:
                    turn_with_scanned_angle(detected_scan_angle, servo_pwm, right_pwm, left_pwm)
                state = "NORMAL"
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle(servo_pwm, 90)
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()