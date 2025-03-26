import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
ServoMotor = 18           # Servo motor PWM for the camera

# Constants
FRAME_WIDTH = 640         # Camera frame width
FRAME_HEIGHT = 480        # Camera frame height
MIN_CONTOUR_AREA = 1000   # Minimum area for valid contours
TURN_THRESHOLD = 80       # Error threshold for normal pivot turning

# Motor speeds
BASE_SPEED = 40           # Base motor speed (0-100)
TURN_SPEED = 50           # Speed for pivot turns (0-100)

# GPIO Setup & Motor Functions
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    # Setup motor control pins
    for pin in [IN1, IN2, IN3, IN4, ENA, ENB, ServoMotor]:
        GPIO.setup(pin, GPIO.OUT)
    # Set up PWM for motors
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    # Set up PWM for servo (if needed for camera adjustment)
    servo_pwm = GPIO.PWM(ServoMotor, 50)  # 50Hz for servo
    servo_pwm.start(0)
    return right_pwm, left_pwm, servo_pwm

def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    left_pwm.ChangeDutyCycle(BASE_SPEED)

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

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    for pin in [IN1, IN2, IN3, IN4]:
        GPIO.output(pin, GPIO.LOW)

def turn_left_90(right_pwm, left_pwm):
    print("Executing Turn Left 90")
    pivot_turn_left(right_pwm, left_pwm)
    time.sleep(1)  # Adjust duration for a 90° turn
    stop_motors(right_pwm, left_pwm)

def turn_right_90(right_pwm, left_pwm):
    print("Executing Turn Right 90")
    pivot_turn_right(right_pwm, left_pwm)
    time.sleep(1)  # Adjust duration for a 90° turn
    stop_motors(right_pwm, left_pwm)

# Camera setup
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

# Enhanced line detection function:
# It returns: error (offset), line_found (bool), and turn_command (string if a 90° turn command is detected).
def detect_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Detect dark areas (line candidates)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Draw center line for reference
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    turn_command = None  # Default, no command

    if contours:
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid_contours:
            # Draw the largest valid contour and compute its centroid for normal line following
            largest = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                error = cx - center_x
                cv2.line(frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)
                cv2.putText(frame, f"Error: {error}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                error = 0

            # If two or more valid contours exist, check for a turn command via Hough transform.
            if len(valid_contours) >= 2:
                lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
                if lines is not None and len(lines) >= 2:
                    # For simplicity, we use the first two lines detected.
                    line1 = lines[0][0]  # (x1, y1, x2, y2)
                    line2 = lines[1][0]
                    # Compute approximate intersection point of the two lines
                    def line_intersection(l1, l2):
                        x1, y1, x2, y2 = l1
                        x3, y3, x4, y4 = l2
                        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if denom == 0:
                            return None
                        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
                        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
                        return int(px), int(py)
                    
                    pt = line_intersection(line1, line2)
                    if pt is not None:
                        cv2.circle(frame, pt, 7, (0,255,255), -1)
                        # Use the intersection's x coordinate relative to center to decide turn command.
                        if pt[0] < center_x - 20:
                            turn_command = "turn left 90"
                        elif pt[0] > center_x + 20:
                            turn_command = "turn right 90"
                        cv2.putText(frame, f"Command: {turn_command}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            return error, True, turn_command
    # If no valid line found, return error 0, line_found False, and no command.
    return 0, False, None

# Main loop: display live feed and issue movement commands
def main():
    right_pwm, left_pwm, _ = setup_gpio()
    picam2 = setup_camera()
    
    print("Live feed with overlays started. Press 'q' to exit.")
    
    try:
        while True:
            frame = picam2.capture_array()
            error, line_found, turn_command = detect_line(frame)
            cv2.imshow("Live Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Movement decision:
            if turn_command is not None:
                # Execute turn commands if detected (priority over normal line following)
                if turn_command == "turn left 90":
                    turn_left_90(right_pwm, left_pwm)
                elif turn_command == "turn right 90":
                    turn_right_90(right_pwm, left_pwm)
            elif line_found:
                # If line is detected and no special turn command, use error to pivot
                if error > TURN_THRESHOLD:
                    pivot_turn_right(right_pwm, left_pwm)
                    print("Pivot Turning Right")
                elif error < -TURN_THRESHOLD:
                    pivot_turn_left(right_pwm, left_pwm)
                    print("Pivot Turning Left")
                else:
                    move_forward(right_pwm, left_pwm)
                    print("Moving Forward")
            else:
                # No line found; stop for safety.
                stop_motors(right_pwm, left_pwm)
                print("Line lost, stopping...")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("GPIO Cleaned Up")

if __name__ == "__main__":
    main()
