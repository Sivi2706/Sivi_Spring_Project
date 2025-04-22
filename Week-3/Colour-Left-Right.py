import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import os
import time

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Define motor driver GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# Setup PWM for speed control (frequency: 1kHz)
right_pwm = GPIO.PWM(ENA, 1000)  # Right motor
left_pwm = GPIO.PWM(ENB, 1000)   # Left motor
right_pwm.start(0)
left_pwm.start(0)

# Speed settings
base_speed = 45
turn_speed = 70
reverse_speed = 35
search_speed = 25  # New setting for searching mode

# Threshold for turning (pixels from center)
turn_threshold = 100

# Define all available color ranges (HSV format)
all_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),    # Lower red range
        ([114, 167, 154], [134, 247, 234])  # Upper red range
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

# Line detection parameters
MIN_AREA = 400  # Minimum contour area to consider valid detection
MIN_WIDTH = 10  # Minimum width for a valid line

# Recovery behavior parameters
max_search_time = 3.0  # Maximum time to search for a line before giving up (seconds)
line_memory = {'x': None, 'y': None, 'w': None, 'h': None, 'color': None, 'last_seen': 0}

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
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        else:
            print("Invalid choice. Please try again.")

def detect_priority_color(frame, color_names):
    """
    Detect colors in priority order, returning the first qualified detection
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    
    best_contour = None
    best_color = None
    best_area = 0
    
    for color_name in color_names:
        color_ranges = all_color_ranges.get(color_name, [])
        
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area and width
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if area > MIN_AREA and w > MIN_WIDTH:
                    valid_contours.append(contour)
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > best_area:
                    best_contour = largest_contour
                    best_color = color_name
                    best_area = area
                    
                    # For highest priority color, return immediately
                    if color_name == color_names[0]:
                        return best_contour, best_color
    
    return best_contour, best_color

def set_speed(left_speed, right_speed):
    left_speed = max(0, min(100, left_speed))
    right_speed = max(0, min(100, right_speed))
    left_pwm.ChangeDutyCycle(left_speed)
    right_pwm.ChangeDutyCycle(right_speed)

def move_forward():
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)    # Right forward
    GPIO.output(IN4, GPIO.HIGH)
    set_speed(base_speed, base_speed)
    print("Moving Forward")
    return "Moving Forward"

def move_reverse(duration=None):
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    set_speed(reverse_speed, reverse_speed)
    print("Moving Reverse")
    
    if duration:
        time.sleep(duration)
        stop()
        
    return "Moving Reverse"

def turn_right(speed=None):
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    if speed:
        set_speed(speed, speed)
    else:
        set_speed(turn_speed, turn_speed)
    print("Turning Right")
    return "Turning Right"

def turn_left(speed=None):
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)    # Right forward
    GPIO.output(IN4, GPIO.HIGH)
    if speed:
        set_speed(speed, speed)
    else:
        set_speed(turn_speed, turn_speed)
    print("Turning Left")
    return "Turning Left"

def spin_search(direction='right', speed=None):
    """Perform a spinning search for the line"""
    if direction == 'right':
        return turn_right(speed or search_speed)
    else:
        return turn_left(speed or search_speed)

def stop():
    set_speed(0, 0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    print("Stopping")
    return "Stopped"

def main():
    print("===== Priority Color Line Follower with Improved Error Handling =====")
    
    # Get color priorities
    color_priority = get_color_choices()
    if not color_priority:
        return
    
    print(f"\nColor priority: {' > '.join(color_priority)}")
    print("Press 'q' to quit or 'c' to change colors")
    
    # Initialize variables for line tracking
    line_lost_time = 0
    search_direction = 'right'  # Initial search direction
    consecutive_detections = 0  # Counter for consecutive successful detections
    search_mode = False
    last_valid_position = None
    
    try:
        while True:
            frame = picam2.capture_array()
            current_time = time.time()
            
            # Detect with priority
            contour, color_name = detect_priority_color(frame, color_priority)
            
            movement = "No line detected"
            outline_coords = "N/A"
            current_color = "None"
            
            if contour is not None:
                # Line detected
                x, y, w, h = cv2.boundingRect(contour)
                outline_coords = f"({x}, {y}, {w}, {h})"
                current_color = color_name
                
                # Update line memory
                line_memory['x'] = x
                line_memory['y'] = y
                line_memory['w'] = w
                line_memory['h'] = h
                line_memory['color'] = color_name
                line_memory['last_seen'] = current_time
                
                # Reset line lost counter and increase consecutive detections
                line_lost_time = 0
                consecutive_detections += 1
                search_mode = False
                
                # Draw with color-specific visualization
                color_map = {
                    'red': (0, 0, 255),
                    'blue': (255, 0, 0),
                    'green': (0, 255, 0),
                    'yellow': (0, 255, 255),
                    'black': (0, 0, 0)
                }
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[color_name], 2)
                
                # Simple left/right control
                line_center = x + w//2
                frame_center = frame.shape[1]//2
                error = line_center - frame_center
                
                # Store last valid position for search
                last_valid_position = {'center': line_center, 'frame_center': frame_center}
                
                if error > turn_threshold:
                    movement = turn_right()
                    # If line is to the right, we'll search right first if we lose it
                    search_direction = 'right'
                elif error < -turn_threshold:
                    movement = turn_left()
                    # If line is to the left, we'll search left first if we lose it
                    search_direction = 'left'
                else:
                    movement = move_forward()
                    
                # After 5 consecutive detections, we're confident we're on track
                if consecutive_detections >= 5:
                    consecutive_detections = 5  # Cap at 5 to prevent overflow
            
            else:
                # Line not detected - implement recovery behavior
                current_color = "None"
                consecutive_detections = max(0, consecutive_detections - 1)  # Decrease confidence
                
                # If we just lost the line
                if line_lost_time == 0:
                    line_lost_time = current_time
                    
                search_time = current_time - line_lost_time
                
                # Choose recovery strategy based on situation
                if consecutive_detections >= 3:
                    # We were confidently following a line, briefly continue forward
                    movement = move_forward()
                    search_mode = False
                
                elif search_time < max_search_time:
                    # Enter search mode - spin in the last known direction of the line
                    search_mode = True
                    
                    # Determine search direction based on last known position
                    if last_valid_position:
                        if last_valid_position['center'] > last_valid_position['frame_center']:
                            search_direction = 'right'
                        else:
                            search_direction = 'left'
                    
                    movement = f"Searching {search_direction}"
                    spin_search(search_direction)
                
                else:
                    # If we've been searching for too long, try the opposite direction
                    if search_time > max_search_time * 2:
                        # If we've searched both directions and found nothing, back up a bit
                        movement = "Recovery: backing up"
                        move_reverse(0.5)  # Back up briefly
                        
                        # Alternate search direction
                        search_direction = 'left' if search_direction == 'right' else 'right'
                        line_lost_time = current_time  # Reset search timer
                    else:
                        # Continue searching in current direction
                        movement = f"Extended search {search_direction}"
                        spin_search(search_direction)
            
            # Display
            priority_text = f"Priority: {'>'.join(color_priority)}"
            detection_text = f"Detected: {current_color}"
            status_text = f"Status: {'Searching' if search_mode else 'Tracking'}"
            
            cv2.putText(frame, priority_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Command: {movement}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if "DISPLAY" in os.environ:
                cv2.imshow("Priority Color Follower", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                stop()
                new_priority = get_color_choices()
                if new_priority:
                    color_priority = new_priority
                    print(f"New priority: {' > '.join(color_priority)}")
    
    except KeyboardInterrupt:
        print("Stopping robot...")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        stop()
        left_pwm.stop()
        right_pwm.stop()
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()