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
MIN_CONTOUR_AREA = 1000    # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100        # Error threshold for pivoting

# Recovery parameters
REVERSE_SPEED = 40         # Speed when reversing

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Define all available color ranges (HSV format) - Adjusted for better recognition
all_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),    # Lower red range
        ([170, 167, 154], [180, 247, 234])  # Upper red range
    ],
    'blue': [
        ([90, 130, 60], [140, 255, 255])    # Widened blue range to catch light blue
    ],
    'green': [
        ([40, 180, 110], [75, 255, 190])    # Adjusted green range
    ],
    'yellow': [
        ([20, 155, 189], [35, 235, 255])    # Adjusted yellow range
    ],
    'black': [
        ([0, 0, 0], [179, 78, 50])          # Original black range
    ]
}

# Default color priority (can be changed by user input)
COLOR_PRIORITY = ['red', 'blue', 'green', 'yellow', 'black']

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

# Improved line detection function that works better with contrast
def detect_line(frame, color_priorities):
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a copy for drawing
    display_frame = frame.copy()
    center_x = FRAME_WIDTH // 2
    cv2.line(display_frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    # Initial values
    best_contour = None
    best_color = None
    max_area = MIN_CONTOUR_AREA
    intersection = False
    
    # Check each color priority
    for color_name in color_priorities:
        color_ranges = all_color_ranges.get(color_name, [])
        
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # Create mask for this color
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphology to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.erode(color_mask, kernel, iterations=1)
            color_mask = cv2.dilate(color_mask, kernel, iterations=1)
            
            # Find contours in this color mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
            
            # Check for intersection
            if len(valid_contours) >= 2:
                intersection = True
            
            # Find the largest contour for this color
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                # If this is the highest priority color so far with a large enough contour
                if area > max_area:
                    max_area = area
                    best_contour = largest
                    best_color = color_name
    
    # Process the best contour if found
    if best_contour is not None:
        M = cv2.moments(best_contour)
        
        # Set contour color based on detected line color
        if best_color == 'red':
            contour_color = (0, 0, 255)  # BGR - Red
        elif best_color == 'blue':
            contour_color = (255, 0, 0)  # BGR - Blue
        elif best_color == 'green':
            contour_color = (0, 255, 0)  # BGR - Green
        elif best_color == 'yellow':
            contour_color = (0, 255, 255)  # BGR - Yellow
        elif best_color == 'black':
            contour_color = (128, 128, 128)  # BGR - Gray
        else:
            contour_color = (255, 255, 255)  # BGR - White
        
        # Draw the contour
        cv2.drawContours(display_frame, [best_contour], -1, contour_color, 2)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(display_frame, (cx, cy), 5, (255, 0, 0), -1)
            error = cx - center_x
            cv2.line(display_frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)
            cv2.putText(display_frame, f"{best_color.capitalize()} Line, Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, contour_color, 2)
            
            # Return the processed frame instead of modifying original
            return display_frame, error, True, intersection, best_color
    
    # If no line is found
    cv2.putText(display_frame, "No Line Detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return display_frame, 0, False, intersection, None

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    # Center the servo initially
    set_servo_angle(servo_pwm, 90)
    
    # Get user's color priority choices
    color_priorities = get_color_choices()
    if color_priorities is None:
        print("Program terminated by user.")
        GPIO.cleanup()
        return
    
    # State variables (simplified to just NORMAL and RECOVERY)
    state = "NORMAL"
    current_color = None  # Track which color line we're following

    print("Line follower with color detection started. Press 'q' in the display window or Ctrl+C to stop.")
    
    try:
        while True:
            frame = picam2.capture_array()
            # Get processed frame and detection results
            display_frame, error, line_found, intersection, detected_color = detect_line(frame, color_priorities)
            
            # If color has changed, announce it
            if line_found and detected_color and detected_color != current_color:
                current_color = detected_color
                print(f"Now following {current_color} line")
                
            cv2.imshow("Line Follower", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if state == "NORMAL":
                if line_found:
                    if error > TURN_THRESHOLD:
                        pivot_turn_right(right_pwm, left_pwm)
                        print(f"Pivot Turning Right - {current_color} line")
                    elif error < -TURN_THRESHOLD:
                        pivot_turn_left(right_pwm, left_pwm)
                        print(f"Pivot Turning Left - {current_color} line")
                    else:
                        move_forward(right_pwm, left_pwm)
                        print(f"Moving Forward - {current_color} line")
                else:
                    # Line lost - enter recovery mode
                    print("Line lost. Entering recovery mode (reversing)...")
                    state = "RECOVERY"
                    move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            
            elif state == "RECOVERY":
                # Keep reversing until a line is found
                if line_found:
                    print(f"{detected_color.capitalize()} line detected during recovery. Returning to normal mode.")
                    stop_motors(right_pwm, left_pwm)
                    state = "NORMAL"
                    current_color = detected_color
                # Continue reversing while no line is found (no time-based limit)
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle(servo_pwm, 90)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")
        
if __name__ == "__main__":
    main()