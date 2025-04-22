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

# Define motor driver GPIO pins (updated from second code)
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

# Speed settings (keeping original values)
base_speed = 45
turn_speed = 70
reverse_speed = 35

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

# Define color priority order (highest to lowest)
COLOR_PRIORITY = ['red', 'blue', 'green', 'yellow', 'black']

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
    
    # Minimum contour area to consider a valid detection
    MIN_AREA = 500  
    
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
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_AREA:
                    return largest_contour, color_name
    
    return None, None

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

def move_reverse():
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    set_speed(reverse_speed, reverse_speed)
    print("Moving Reverse")
    return "Moving Reverse"

def turn_right():
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    set_speed(turn_speed, turn_speed)
    print("Turning Right")
    return "Turning Right"

def turn_left():
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)    # Right forward
    GPIO.output(IN4, GPIO.HIGH)
    set_speed(turn_speed, turn_speed)
    print("Turning Left")
    return "Turning Left"

def stop():
    set_speed(0, 0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    print("Stopping")
    return "Stopped"

def main():
    print("===== Priority Color Line Follower =====")
    
    # Get color priorities
    color_priority = get_color_choices()
    if not color_priority:
        return
    
    print(f"\nColor priority: {' > '.join(color_priority)}")
    print("Press 'q' to quit or 'c' to change colors")
    
    try:
        while True:
            frame = picam2.capture_array()
            
            # Detect with priority
            contour, color_name = detect_priority_color(frame, color_priority)
            
            movement = "No line detected"
            outline_coords = "N/A"
            current_color = "None"
            
            if contour is not None:
                x, y, w, h = cv2.boundingRect(contour)
                outline_coords = f"({x}, {y}, {w}, {h})"
                current_color = color_name
                
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
                
                if error > turn_threshold:
                    movement = turn_right()
                elif error < -turn_threshold:
                    movement = turn_left()
                else:
                    movement = move_forward()
            else:
                movement = move_reverse()
            
            # Display
            priority_text = f"Priority: {'>'.join(color_priority)}"
            detection_text = f"Detected: {current_color}"
            cv2.putText(frame, priority_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Command: {movement}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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