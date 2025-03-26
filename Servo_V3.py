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

def set_servo_angle(pwm, angle):
    global current_angle
    angle = max(0, min(180, angle))
    duty = SERVO_MIN + (angle * (SERVO_MAX - SERVO_MIN) / 180)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    current_angle = angle

def move_with_servo_turn(right_pwm, left_pwm, servo_pwm, angle, speed=45):
    global right_counter, left_counter
    
    # Reset encoder counters
    right_counter = 0
    left_counter = 0
    
    # Set servo to target angle
    set_servo_angle(servo_pwm, angle)
    
    # Calculate turn degrees
    angle_diff = 90 - angle  # Positive = right turn, Negative = left turn
    
    # Determine turn direction
    direction = 'right' if angle_diff > 0 else 'left'
    degrees = abs(angle_diff)
    
    # Calculate required encoder counts for turn
    arc_length = (degrees * math.pi * WHEEL_BASE) / 360
    counts_needed = int((arc_length / WHEEL_CIRCUM) * PULSES_PER_REV * TURN_RATIO)
    
    # Set motor directions and PWM
    if direction == 'right':
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)
    
    # Execute turn
    try:
        while True:
            current = right_counter if direction == 'right' else left_counter
            if current >= counts_needed:
                break
            time.sleep(0.01)
    finally:
        # Stop motors
        right_pwm.ChangeDutyCycle(0)
        left_pwm.ChangeDutyCycle(0)
        GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
        
        # Return servo to center
        set_servo_angle(servo_pwm, 90)
        
        print(f"Completed {degrees}° {direction} turn")

def manual_control():
    right_pwm, left_pwm, servo_pwm = setup()
    
    print("Command Guide:")
    print("sv X - Set view angle (0-180) and turn")
    print("s - Stop, q - Quit")
    
    try:
        while True:
            cmd = input("\nCommand: ").lower().strip()
            
            if cmd == 'q':
                break
            elif cmd == 's':
                right_pwm.ChangeDutyCycle(0)
                left_pwm.ChangeDutyCycle(0)
                GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
            elif cmd.startswith('sv '):
                try:
                    angle = float(cmd.split()[1])
                    if 0 <= angle <= 180:
                        move_with_servo_turn(right_pwm, left_pwm, servo_pwm, angle)
                    else:
                        print("Angle 0-180 only")
                except Exception as e:
                    print(f"Invalid command: {e}")
    finally:
        right_pwm.stop()
        left_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    manual_control()