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
ServoMotor = 18           # Servo motor PWM pin (kept but not used)

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Line following parameters
BASE_SPEED = 45           # Base motor speed (0-100)
TURN_SPEED = 60           # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 1000    # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100       # Error threshold for pivoting

# Recovery parameters
REVERSE_SPEED = 40         # Speed when reversing

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Define all available color ranges (HSV format)
all_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),    # Lower red range
        ([170, 167, 154], [180, 247, 234])  # Upper red range
    ],
    'blue': [
        ([100, 167, 60], [130, 255, 95])    # Adjusted blue range
    ],
    'green': [
        ([40, 180, 110], [75, 255, 190])    # Adjusted green range
    ],
    'yellow': [
        ([80, 100, 100], [100, 255, 255])  # Adjusted to detect cyan (instead of yellow)
    ],

    'black': [
        ([0, 0, 0], [179, 78, 50])          # Original black range
    ]
}

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

# Refined line detection function
def detect_line(frame, color_priorities):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)

    best_contour = None
    best_color = None
    best_cx, best_cy = -1, -1
    max_area = 0

    # Dictionary to store all valid contours for each color
    valid_contours = {}

    for color_name in color_priorities:
        color_ranges = all_color_ranges.get(color_name, [])
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter by area
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

        if valid:
            valid_contours[color_name] = valid

    # Prioritize: Find the best contour of the highest-priority available color
    for color_name in color_priorities:
        if color_name in valid_contours:
            for contour in valid_contours[color_name]:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    best_contour = contour
                    best_color = color_name
            break  # Found the highest-priority color available

    if best_contour is not None:
        M = cv2.moments(best_contour)
        if M["m00"] != 0:
            best_cx = int(M["m10"] / M["m00"])
            best_cy = int(M["m01"] / M["m00"])

            hsv_value = hsv[best_cy, best_cx]
            h, s, v = hsv_value

            contour_color = (0, 255, 0) if best_color != 'black' else (128, 128, 128)
            cv2.drawContours(frame, [best_contour], -1, contour_color, 2)
            cv2.circle(frame, (best_cx, best_cy), 5, (255, 0, 0), -1)
            cv2.line(frame, (center_x, best_cy), (best_cx, best_cy), (255, 0, 0), 2)

            error = best_cx - center_x

            cv2.putText(frame, f"{best_color.capitalize()} Line, Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, contour_color, 2)
            cv2.putText(frame, f"HSV: ({h}, {s}, {v})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return error, True, best_color

    return 0, False, None


# Main function
def main():
    right_pwm, left_pwm = setup_gpio()
    picam2 = setup_camera()
    
    # Get user's color priority choices
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    print("Line follower with color detection started. Press 'q' in the display window or Ctrl+C to stop.")
    
    try:
        while True:
            frame = picam2.capture_array()
            error, line_found, detected_color = detect_line(frame, color_priorities)
            
            cv2.imshow("Line Follower", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if line_found and detected_color in color_priorities:
                if error > TURN_THRESHOLD:
                    pivot_turn_right(right_pwm, left_pwm)
                    print(f"Pivot Turning Right - {detected_color} line")
                elif error < -TURN_THRESHOLD:
                    pivot_turn_left(right_pwm, left_pwm)
                    print(f"Pivot Turning Left - {detected_color} line")
                else:
                    move_forward(right_pwm, left_pwm)
                    print(f"Moving Forward - {detected_color} line")
            else:
                print("No prioritized color line detected. Reversing...")
                move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                
                while True:
                    frame = picam2.capture_array()
                    _, line_found, detected_color = detect_line(frame, color_priorities)
                    if line_found and detected_color in color_priorities:
                        stop_motors(right_pwm, left_pwm)
                        break
                    time.sleep(0.1)

                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()