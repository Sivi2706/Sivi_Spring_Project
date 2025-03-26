import RPi.GPIO as GPIO
import time
import numpy as np
import math

# GPIO Pins
IN1, IN2 = 22, 27    # Left motor
IN3, IN4 = 17, 4     # Right motor
ENA, ENB = 13, 12    # PWM pins
encoderPinRight = 23  # Right encoder
encoderPinLeft = 24   # Left encoder
ServoMotor = 18       # Servo

# Constants
WHEEL_DIAMETER = 4.05  # cm
PULSES_PER_REV = 20
WHEEL_CIRCUM = math.pi * WHEEL_DIAMETER
WHEEL_BASE = 12.0      # cm
SERVO_MIN = 2.5        # 0°
SERVO_MAX = 12.5       # 180°
SERVO_FREQ = 50
TURN_RATIO = 1.08      # Calibration factor

# Global variables
right_counter = 0
left_counter = 0
current_angle = 90
right_dir = 1
left_dir = 1

def right_encoder(channel):
    global right_counter, right_dir
    right_counter += right_dir

def left_encoder(channel):
    global left_counter, left_dir
    left_counter += left_dir

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Motor setup
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
    GPIO.setup([encoderPinRight, encoderPinLeft], GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder)
    
    # PWM setup
    right_pwm = GPIO.PWM(ENA, 1000)
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    
    # Servo setup
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

def set_servo(pwm, angle):
    global current_angle
    angle = max(0, min(180, angle))
    duty = SERVO_MIN + (angle * (SERVO_MAX - SERVO_MIN) / 180)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    current_angle = angle

# Movement functions remain the same as previous version
# [Include move_forward, move_backward, turn_right, turn_left, stop_motors from previous code]

def calculate_rotation():
    right_dist = (right_counter / PULSES_PER_REV) * WHEEL_CIRCUM
    left_dist = (left_counter / PULSES_PER_REV) * WHEEL_CIRCUM
    return math.degrees((right_dist - left_dist) / WHEEL_BASE)

def execute_servo_turn(servo, right, left, target):
    # Move servo to target and back to center
    set_servo(servo, target)
    set_servo(servo, 90)
    
    # Calculate required turn
    angle_diff = 90 - target  # Positive = right turn, Negative = left turn
    if angle_diff == 0:
        print("Already centered")
        return
    
    direction = 'right' if angle_diff > 0 else 'left'
    degrees = abs(angle_diff)
    
    # Calculate required encoder counts
    arc_length = (degrees * math.pi * WHEEL_BASE) / 360
    counts_needed = int((arc_length / WHEEL_CIRCUM) * PULSES_PER_REV * TURN_RATIO)
    
    # Execute turn
    right_counter = left_counter = 0
    (turn_right if direction == 'right' else turn_left)(right, left, 45)
    
    try:
        while True:
            current = right_counter if direction == 'right' else left_counter
            if current >= counts_needed:
                break
            time.sleep(0.01)
    finally:
        stop_motors(right, left)
        print(f"Completed {degrees}° {direction} turn")
        print(f"Actual rotation: {calculate_rotation():.1f}°")

def manual_control():
    right_pwm, left_pwm, servo_pwm = setup()
    set_servo(servo_pwm, 90)
    
    print("Command Guide:")
    print("sv X - Set view angle (0-180)")
    print("f/b/r/l X - Move directions with speed")
    print("s - Stop, q - Quit")
    
    try:
        while True:
            cmd = input("\nCommand: ").lower().strip()
            
            if cmd == 'q':
                break
            elif cmd == 's':
                stop_motors(right_pwm, left_pwm)
            elif cmd.startswith('sv '):
                try:
                    angle = float(cmd.split()[1])
                    if 0 <= angle <= 180:
                        execute_servo_turn(servo_pwm, right_pwm, left_pwm, angle)
                    else:
                        print("Angle 0-180 only")
                except:
                    print("Invalid angle")
            elif cmd.split()[0] in ['f','b','r','l']:
                # Movement commands (same as previous)
                pass
                
    finally:
        stop_motors(right_pwm, left_pwm)
        servo_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    manual_control()