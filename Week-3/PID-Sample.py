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

# Define all available color ranges
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

# Function to get user's color choice
def get_color_choice():
    print("\nAvailable line colors to follow:")
    print("r = red")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black")
    print("q = quit program")
    
    while True:
        choice = input("\nEnter line color (r/b/g/y/k): ").lower()
        if choice == 'q':
            return None
        elif choice == 'r':
            return 'red'
        elif choice == 'b':
            return 'blue'
        elif choice == 'g':
            return 'green'
        elif choice == 'y':
            return 'yellow'
        elif choice == 'k':
            return 'black'
        else:
            print("Invalid choice. Please try again.")

def detect_color(frame, color_name):
    """
    Detect the specified color in the frame using HSV
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    max_area = 0
    largest_contour = None
    mean_hsv = None
    
    kernel = np.ones((5, 5), np.uint8)
    
    # Get all ranges for the specified color
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
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
                
                mask_temp = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_temp, [contour], -1, 255, -1)
                mean_hsv = cv2.mean(hsv, mask=mask_temp)[:3]
    
    return largest_contour, mean_hsv

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

def main():
    print("===== Color Line Follower =====")
    
    # Get user's color choice
    color_name = get_color_choice()
    if color_name is None:
        return
    
    print(f"\nFollowing {color_name} line. Press 'q' to exit or change color.")
    print("Place the robot on the line to begin...")
    
    error = 0
    
    try:
        while True:
            frame = picam2.capture_array()

            largest_contour, mean_hsv = detect_color(frame, color_name)

            movement = "No line detected"
            outline_coords = "N/A"

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
            color_code = f"Tracking: {color_name}"
            metadata = [
                color_code,
                f"Command: {movement}",
                f"Outline: {outline_coords}",
                f"Error: {error:.2f}"
            ]
            
            for i, text in enumerate(metadata):
                cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if "DISPLAY" in os.environ:
                cv2.imshow(f"Color Line Detection - Following {color_name}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Allow changing color while running
                stop()
                color_name = get_color_choice()
                if color_name is None:
                    break
                print(f"\nNow following {color_name} line...")

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
            picam2.stop()
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
        try:
            GPIO.cleanup()
        except Exception as e:
            print(f"Error during GPIO cleanup: {e}")

if _name_ == "_main_":
    main()