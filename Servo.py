import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# --------------------- GPIO & Motor Pins ----------------------
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera

# --------------------- Robot / Movement Constants -------------
WHEEL_DIAMETER = 4.05
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER

SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5
SERVO_FREQ = 50

BASE_SPEED = 40            # Base motor speed (0-100)
TURN_SPEED = 50            # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 1000    # Min area for valid line contours
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

TURN_THRESHOLD = 80        # Error threshold for normal pivot turning
REVERSE_DURATION = 0.5     # Seconds to reverse
REVERSE_SPEED = 40         # Speed when reversing

# Scanning angles
SCAN_ANGLES = [90, 45, 135]
SCAN_TIME_PER_ANGLE = 0.5

# Encoder counters
right_counter = 0
left_counter = 0

# --------------------- Encoder Callbacks -----------------------
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

# --------------------- GPIO Setup ------------------------------
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Motor pins
    for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
        GPIO.setup(pin, GPIO.OUT)

    # Encoders
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)

    # PWM for motors
    right_pwm = GPIO.PWM(ENA, 1000)  # 1kHz
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)

    # Servo
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)

    return right_pwm, left_pwm, servo_pwm

# --------------------- Servo Control ---------------------------
def set_servo_angle_simple(servo_pwm, angle):
    angle = max(0, min(180, angle))  # Constrain
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

# --------------------- Motor Control ---------------------------
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
    for pin in [IN1, IN2, IN3, IN4]:
        GPIO.output(pin, GPIO.LOW)

def turn_with_scanned_angle(scanned_angle, servo_pwm, right_pwm, left_pwm):
    """
    Uses the servo scanning angle to decide how long to pivot turn.
    If scanned_angle > 90 => pivot left, < 90 => pivot right.
    """
    turn_time = abs(scanned_angle - 90) / 45.0  # 45° => 1s
    if scanned_angle > 90:
        print(f"Detected angle {scanned_angle}: Pivoting LEFT for {turn_time:.2f} s")
        pivot_turn_left(right_pwm, left_pwm)
    elif scanned_angle < 90:
        print(f"Detected angle {scanned_angle}: Pivoting RIGHT for {turn_time:.2f} s")
        pivot_turn_right(right_pwm, left_pwm)
    else:
        print("Detected angle 90: No pivot required.")
        return
    time.sleep(turn_time)
    stop_motors(right_pwm, left_pwm)
    print("Resetting servo to 90 degrees")
    set_servo_angle_simple(servo_pwm, 90)

# --------------------- Camera Setup ----------------------------
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

# --------------------- Corner Detection Helpers ----------------
def line_intersection(l1, l2):
    """
    Compute the intersection of two lines each given by (x1, y1, x2, y2).
    Returns (px, py) or None if parallel/out of range.
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return int(px), int(py)

def confirm_turn_direction_with_black_pixels(mask, px, py):
    """
    Look at a region around (px, py). Whichever side (left or right) has more
    black pixels indicates a left or right 90° corner.
    """
    region_size = 50  # half-size
    x1 = max(0, px - region_size)
    x2 = min(FRAME_WIDTH, px + region_size)
    y1 = max(0, py - region_size)
    y2 = min(FRAME_HEIGHT, py + region_size)
    if x2 <= x1 or y2 <= y1:
        return None

    submask = mask[y1:y2, x1:x2]
    mid_x = submask.shape[1] // 2
    left_region = submask[:, :mid_x]
    right_region = submask[:, mid_x:]
    left_black_count = cv2.countNonZero(left_region)
    right_black_count = cv2.countNonZero(right_region)

    if left_black_count > right_black_count:
        return "turn left 90"
    elif right_black_count > left_black_count:
        return "turn right 90"
    return None

# --------------------- Enhanced detect_line --------------------
def detect_line(frame):
    """
    Returns:
      error (int)          : horizontal offset for normal line following
      line_found (bool)    : whether we see a main line
      intersection (bool)  : if multiple large contours are found
      turn_command (str)   : "turn left 90", "turn right 90", or None
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)

    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)

    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    turn_command = None
    error = 0
    line_found = False
    intersection = False

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        if len(valid_contours) >= 2:
            intersection = True
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
                cv2.line(frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)
                cv2.putText(frame, f"Error: {error}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # --------- Hough lines for corner detection -----------
    lines = cv2.HoughLinesP(mask_black, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    if lines is not None and len(lines) >= 2:
        # Draw them for debugging
        for i, l in enumerate(lines):
            x1, y1, x2, y2 = l[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"L{i}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        found_corner = False
        for i in range(len(lines)):
            if found_corner:
                break
            for j in range(i+1, len(lines)):
                l1 = lines[i][0]
                l2 = lines[j][0]
                pt = line_intersection(l1, l2)
                if pt is not None:
                    px, py = pt
                    if 0 <= px < FRAME_WIDTH and 0 <= py < FRAME_HEIGHT:
                        cv2.circle(frame, (px, py), 5, (0,255,255), -1)
                        direction = confirm_turn_direction_with_black_pixels(mask_black, px, py)
                        if direction is not None:
                            turn_command = direction
                            cv2.putText(frame, turn_command, (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                            found_corner = True
                            break

    return error, line_found, intersection, turn_command

# --------------------- Main Program ---------------------------
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()

    # Center the servo
    set_servo_angle_simple(servo_pwm, 90)
    
    state = "NORMAL"
    reverse_start_time = 0
    current_scan_index = 0
    scan_start_time = 0
    detected_scan_angle = None

    print("Line follower started. Press 'q' in the display window or Ctrl+C to stop.")

    try:
        while True:
            frame = picam2.capture_array()
            error, line_found, intersection, turn_command = detect_line(frame)
            cv2.imshow("Line Follower", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if state == "NORMAL":
                if line_found:
                    # Normal line following
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
                    # Lost line => Reverse
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
                    set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                    scan_start_time = time.time()

            elif state == "SCANNING":
                # Wait for the servo angle to settle, then check for line
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    frame = picam2.capture_array()
                    error, line_found, intersection, turn_command = detect_line(frame)

                    # If a 90° corner is detected, turn immediately
                    if turn_command == "turn left 90":
                        print("Detected 90° left corner while scanning!")
                        pivot_turn_left(right_pwm, left_pwm)
                        time.sleep(1.0)  # Adjust for actual 90° turn
                        stop_motors(right_pwm, left_pwm)
                        set_servo_angle_simple(servo_pwm, 90)
                        state = "NORMAL"

                    elif turn_command == "turn right 90":
                        print("Detected 90° right corner while scanning!")
                        pivot_turn_right(right_pwm, left_pwm)
                        time.sleep(1.0)  # Adjust for actual 90° turn
                        stop_motors(right_pwm, left_pwm)
                        set_servo_angle_simple(servo_pwm, 90)
                        state = "NORMAL"

                    # Otherwise, if we found a normal line, pivot based on scanned angle
                    elif line_found:
                        detected_scan_angle = SCAN_ANGLES[current_scan_index]
                        print(f"Line detected during scan at servo angle: {detected_scan_angle}")
                        state = "TURNING"

                    # If not found, go to next scan angle
                    else:
                        current_scan_index += 1
                        if current_scan_index < len(SCAN_ANGLES):
                            set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
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
        set_servo_angle_simple(servo_pwm, 90)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()
