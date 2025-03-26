import RPi.GPIO as GPIO
import time
import numpy as np
import math

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera

# Constants (to be calibrated)
WHEEL_DIAMETER = 4.05     # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm
WHEEL_BASE_DISTANCE = 12.0  # cm, distance between wheels
TURN_RATIO = 1.07         # Calibration factor for precise turns

# Servo motor parameters
SERVO_MIN_DUTY = 2.5      # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5     # Duty cycle for 180 degrees
SERVO_FREQ = 50           # 50Hz frequency for servo

# Global variables
right_counter = 0
left_counter = 0
current_servo_angle = 90  # Default servo position
right_direction = 1       # 1 for forward, -1 for backward
left_direction = 1        # 1 for forward, -1 for backward

# Encoder callback functions
def right_encoder_callback(channel):
    global right_counter, right_direction
    right_counter += right_direction

def left_encoder_callback(channel):
    global left_counter, left_direction
    left_counter += left_direction

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
    right_pwm = GPIO.PWM(ENA, 1000)
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
    global current_servo_angle
    angle = max(0, min(180, angle))
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    current_servo_angle = angle

# Movement functions
def move_forward(right_pwm, left_pwm, speed):
    global right_direction, left_direction
    right_direction = left_direction = 1
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def move_backward(right_pwm, left_pwm, speed):
    global right_direction, left_direction
    right_direction = left_direction = -1
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_right(right_pwm, left_pwm, speed):
    global right_direction, left_direction
    right_direction = 1
    left_direction = -1
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_left(right_pwm, left_pwm, speed):
    global right_direction, left_direction
    right_direction = -1
    left_direction = 1
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Calculate distance from encoder counts
def calculate_distance(encoder_count):
    return (encoder_count / PULSES_PER_REVOLUTION) * WHEEL_CIRCUMFERENCE

# Calculate rotation angle
def calculate_rotation_angle():
    right_dist = calculate_distance(right_counter)
    left_dist = calculate_distance(left_counter)
    return math.degrees((right_dist - left_dist) / WHEEL_BASE_DISTANCE)

# Print movement stats
def print_movement_stats():
    print(f"Right Pulses: {right_counter} | Left Pulses: {left_counter}")
    print(f"Right Distance: {calculate_distance(right_counter):.2f} cm")
    print(f"Left Distance: {calculate_distance(left_counter):.2f} cm")
    print(f"Rotation Angle: {calculate_rotation_angle():.2f}°")

# New precise turning function
def execute_precise_turn(servo_pwm, right_pwm, left_pwm, target_angle):
    global right_counter, left_counter
    
    # Show target angle
    turn_angle = target_angle - 90
    print(f"\nPreparing {abs(turn_angle)}° turn to {'right' if turn_angle > 0 else 'left'}")
    
    # 1. Turn servo to target angle
    set_servo_angle(servo_pwm, target_angle)
    time.sleep(0.5)
    
    # 2. Calculate required encoder counts for the turn
    wheel_distance = (abs(turn_angle) * math.pi * WHEEL_BASE_DISTANCE * TURN_RATIO) / 360.0
    required_counts = int((wheel_distance / WHEEL_CIRCUMFERENCE) * PULSES_PER_REVOLUTION)
    
    # 3. Execute the physical turn
    right_counter = left_counter = 0
    if turn_angle > 0:
        turn_right(right_pwm, left_pwm, 40)
    else:
        turn_left(right_pwm, left_pwm, 40)
    
    try:
        while True:
            current_counts = right_counter if turn_angle > 0 else left_counter
            if current_counts >= required_counts:
                break
            time.sleep(0.01)
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle(servo_pwm, 90)  # Return servo to center
        print(f"Turn completed: {calculate_rotation_angle():.2f}° rotation")
        print_movement_stats()

# Main control loop
def manual_control():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    set_servo_angle(servo_pwm, 90)
    
    print("\n==== Enhanced Robot Control ====")
    print("Commands:")
    print("  f <speed> - Move forward")
    print("  b <speed> - Move backward")
    print("  r <speed> - Turn right")
    print("  l <speed> - Turn left")
    print("  s - Stop motors")
    print("  sv <angle> - Set view angle (0-180) and turn")
    print("  t <time> - Set movement time")
    print("  q - Quit")
    
    movement_time = 1.0
    
    try:
        while True:
            cmd = input("\nCommand (f/b/r/l/s/sv/t/q): ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 's':
                stop_motors(right_pwm, left_pwm)
            elif cmd.startswith('t '):
                try:
                    movement_time = float(cmd.split()[1])
                    print(f"Movement time set to {movement_time:.1f}s")
                except:
                    print("Invalid time")
            elif cmd.startswith('sv '):
                try:
                    angle = float(cmd.split()[1])
                    if 0 <= angle <= 180:
                        execute_precise_turn(servo_pwm, right_pwm, left_pwm, angle)
                    else:
                        print("Angle must be 0-180")
                except:
                    print("Invalid angle")
            elif len(cmd.split()) == 2:
                cmd, speed = cmd.split()
                try:
                    speed = float(speed)
                    if 0 <= speed <= 100:
                        right_counter = left_counter = 0
                        if cmd == 'f':
                            move_forward(right_pwm, left_pwm, speed)
                        elif cmd == 'b':
                            move_backward(right_pwm, left_pwm, speed)
                        elif cmd == 'r':
                            turn_right(right_pwm, left_pwm, speed)
                        elif cmd == 'l':
                            turn_left(right_pwm, left_pwm, speed)
                        time.sleep(movement_time)
                        stop_motors(right_pwm, left_pwm)
                        print_movement_stats()
                except:
                    print("Invalid speed")
    finally:
        stop_motors(right_pwm, left_pwm)
        servo_pwm.stop()
        GPIO.cleanup()
        print("System shutdown")

if __name__ == "__main__":
    manual_control()