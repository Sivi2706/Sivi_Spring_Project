import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# GPIO and Motor Setup (similar to previous script)
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors

# Constants
BASE_SPEED = 40
TURN_SPEED = 50
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONTOUR_AREA = 1000

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Motor pin setup
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    
    right_pwm = GPIO.PWM(ENA, 1000)
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    
    return right_pwm, left_pwm

def pivot_turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    time.sleep(0.5)  # 90-degree turn duration
    stop_motors(right_pwm, left_pwm)

def pivot_turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)    # Left backward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)    # Right forward
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)
    time.sleep(0.5)  # 90-degree turn duration
    stop_motors(right_pwm, left_pwm)

def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    left_pwm.ChangeDutyCycle(BASE_SPEED)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

def detect_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Noise reduction
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter valid contours
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
    
    # Initialize return values
    turn_command = None
    
    # Check for intersection and turn detection
    if len(valid_contours) >= 2:
        # Get bounding rectangles for valid contours
        rects = [cv2.boundingRect(cnt) for cnt in valid_contours]
        
        # Check angle between rectangles
        if len(rects) >= 2:
            x1, y1, w1, h1 = rects[0]
            x2, y2, w2, h2 = rects[1]
            
            # Vertical and horizontal lines should have different orientations
            is_vertical_1 = w1 < h1
            is_vertical_2 = w2 < h2
            
            # Check if lines are approximately perpendicular
            if is_vertical_1 != is_vertical_2:
                if is_vertical_1:
                    # Check which is on the left or right of the image center
                    if x1 < FRAME_WIDTH/2 and x2 > FRAME_WIDTH/2:
                        turn_command = "turn_left_90"
                    elif x1 > FRAME_WIDTH/2 and x2 < FRAME_WIDTH/2:
                        turn_command = "turn_right_90"
                else:
                    # Check which is above or below the image center
                    if y1 < FRAME_HEIGHT/2 and y2 > FRAME_HEIGHT/2:
                        turn_command = "turn_left_90"
                    elif y1 > FRAME_HEIGHT/2 and y2 < FRAME_HEIGHT/2:
                        turn_command = "turn_right_90"
    
    return turn_command

def main():
    right_pwm, left_pwm = setup_gpio()
    picam2 = setup_camera()
    
    print("Enhanced Line Follower: Press 'q' to quit")
    
    try:
        while True:
            frame = picam2.capture_array()
            
            # Detect turn command
            turn_command = detect_line(frame)
            
            # Perform movement based on detected command
            if turn_command == "turn_left_90":
                print("Turning Left 90 Degrees")
                pivot_turn_left(right_pwm, left_pwm)
            elif turn_command == "turn_right_90":
                print("Turning Right 90 Degrees")
                pivot_turn_right(right_pwm, left_pwm)
            else:
                move_forward(right_pwm, left_pwm)
            
            # Display window
            cv2.imshow("Line Follower", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()