import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# --------------------- GPIO & Motor Pins ----------------------
IN1, IN2 = 22, 27         # Left motor control pins
IN3, IN4 = 17, 4          # Right motor control pins
ENA, ENB = 13, 12         # PWM pins for motors
ServoMotor = 18           # (Optional) Servo motor PWM for camera

# --------------------- Camera & Image Params ------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONTOUR_AREA = 1000   # Minimum area to consider a valid line contour

# --------------------- Movement Speeds ------------------------
BASE_SPEED = 40           # Base forward speed (0-100)
TURN_SPEED = 50           # Speed used for pivot turns (0-100)
TURN_THRESHOLD = 80       # Error threshold for normal pivot turning

# --------------------- Setup GPIO -----------------------------
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in [IN1, IN2, IN3, IN4, ENA, ENB, ServoMotor]:
        GPIO.setup(pin, GPIO.OUT)
    # Set up PWM for motors
    right_pwm = GPIO.PWM(ENA, 1000)  # 1 kHz
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    # (Optional) Set up PWM for servo
    servo_pwm = GPIO.PWM(ServoMotor, 50)  # 50Hz
    servo_pwm.start(0)
    return right_pwm, left_pwm, servo_pwm

# --------------------- Motor Control --------------------------
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

def turn_left_90(right_pwm, left_pwm, duration=1.0):
    """
    Executes a ~90° turn to the left by pivoting motors for `duration` seconds.
    Adjust duration to fine-tune the actual angle.
    """
    print("Executing Turn Left 90")
    pivot_turn_left(right_pwm, left_pwm)
    time.sleep(duration)
    stop_motors(right_pwm, left_pwm)

def turn_right_90(right_pwm, left_pwm, duration=1.0):
    """
    Executes a ~90° turn to the right by pivoting motors for `duration` seconds.
    Adjust duration to fine-tune the actual angle.
    """
    print("Executing Turn Right 90")
    pivot_turn_right(right_pwm, left_pwm)
    time.sleep(duration)
    stop_motors(right_pwm, left_pwm)

# --------------------- Camera Setup ---------------------------
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

# --------------------- Intersection Helper ---------------------
def line_intersection(l1, l2):
    """
    Computes the intersection (px, py) of two lines each given by (x1, y1, x2, y2).
    Returns None if lines are parallel or if denominator = 0.
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

# --------------------- Main Detection Function -----------------
def detect_line(frame):
    """
    Returns:
        error (int): Horizontal offset from center for normal line following
        line_found (bool): Whether a line is detected
        turn_command (str): "turn left 90", "turn right 90", or None
    """
    # Convert to HSV and threshold for dark (black) regions
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Draw a vertical line at the frame center
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)

    # Find contours to get the largest contour for line-following
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    turn_command = None
    error = 0
    line_found = False

    if contours:
        # Filter out small contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid_contours:
            # Largest contour for normal line following
            largest = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                error = cx - center_x
                line_found = True
                # Draw error line
                cv2.line(frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)
                cv2.putText(frame, f"Error: {error}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- Hough lines to detect potential 90° corner ---
    # (We always run it on the same mask; you can refine if needed)
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    if lines is not None and len(lines) >= 2:
        # Draw lines for debugging
        for i, l in enumerate(lines):
            x1, y1, x2, y2 = l[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"L{i}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Check all line pairs
        found_intersection = False
        for i in range(len(lines)):
            if found_intersection:
                break
            for j in range(i+1, len(lines)):
                l1 = lines[i][0]
                l2 = lines[j][0]
                pt = line_intersection(l1, l2)
                if pt is not None:
                    px, py = pt
                    # Check if intersection is within frame bounds
                    if 0 <= px < FRAME_WIDTH and 0 <= py < FRAME_HEIGHT:
                        # Mark the intersection
                        cv2.circle(frame, (px, py), 7, (0, 255, 255), -1)
                        cv2.putText(frame, f"({px},{py})", (px+5, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                        # Decide turn command if intersection is clearly left or right
                        if px < center_x - 20:
                            turn_command = "turn left 90"
                        elif px > center_x + 20:
                            turn_command = "turn right 90"
                        if turn_command:
                            cv2.putText(frame, turn_command, (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            found_intersection = True
                            break

    return error, line_found, turn_command

# --------------------- Main Loop ------------------------------
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
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

            # -- Movement Logic --
            if turn_command is not None:
                # Priority: if there's a special turn command, do it
                if turn_command == "turn left 90":
                    turn_left_90(right_pwm, left_pwm, duration=1.0)
                elif turn_command == "turn right 90":
                    turn_right_90(right_pwm, left_pwm, duration=1.0)

            elif line_found:
                # Normal line following using error
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
                # If line not found, stop or do any fallback
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
