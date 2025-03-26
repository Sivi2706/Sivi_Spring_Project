import RPi.GPIO as GPIO
import time
import numpy as np
import math

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
encoderPinRight = 23      # Right encoder
encoderPinLeft =  24      # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera

# Constants (to be calibrated)
WHEEL_DIAMETER = 4.05  # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5     # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5    # Duty cycle for 180 degrees
SERVO_FREQ = 50          # 50Hz frequency for servo

# Robot geometry constants (to be measured precisely)
ROBOT_WIDTH = 20  # Distance between wheels in cm
ROBOT_LENGTH = 30  # Length of the robot in cm

# Variables to store encoder counts
right_counter = 0
left_counter = 0
initial_servo_angle = 90  # Default initial servo position
current_servo_angle = 90  # Track current servo angle

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
    global current_servo_angle
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Give servo time to move
    servo_pwm.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter
    current_servo_angle = angle

# Movement functions
def move_forward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_right(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_left(right_pwm, left_pwm, speed):
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

# Function to calculate distance
def calculate_distance(encoder_count):
    revolutions = encoder_count / PULSES_PER_REVOLUTION
    distance = revolutions * WHEEL_CIRCUMFERENCE
    return distance

# Rotation function maintaining servo angle
def rotate_to_servo_angle(right_pwm, left_pwm, servo_pwm):
    global right_counter, left_counter, current_servo_angle
    
    # Initial setup
    initial_right_counter = right_counter
    initial_left_counter = left_counter
    
    # Calculate rotation based on current servo angle
    if current_servo_angle > 90:
        # Turn right
        turn_right(right_pwm, left_pwm, 50)  # 50% speed
        rotation_direction = "right"
    elif current_servo_angle < 90:
        # Turn left
        turn_left(right_pwm, left_pwm, 50)  # 50% speed
        rotation_direction = "left"
    else:
        print("Servo already at forward position. No rotation needed.")
        return
    
    # Calculate rotation angle
    angle_difference = abs(current_servo_angle - 90)
    
    # Calculate encoder pulses needed for rotation
    circumference = math.pi * ROBOT_WIDTH
    rotation_portion = angle_difference / 360.0
    required_distance = circumference * rotation_portion
    
    # Estimate pulses needed (this will need precise calibration)
    estimated_pulses = int(required_distance / WHEEL_CIRCUMFERENCE * PULSES_PER_REVOLUTION)
    
    # Wait until we've approximately rotated the right amount
    start_time = time.time()
    while (abs(right_counter - initial_right_counter) < estimated_pulses and 
           abs(left_counter - initial_left_counter) < estimated_pulses):
        time.sleep(0.1)
        if time.time() - start_time > 5:  # Timeout after 5 seconds
            break
    
    # Stop motors
    stop_motors(right_pwm, left_pwm)
    
    # Print rotation stats
    print("Rotation Complete:")
    print(f"Current Servo Angle: {current_servo_angle}")
    print(f"Rotation Direction: {rotation_direction}")
    print(f"Right Encoder Pulses: {right_counter - initial_right_counter}")
    print(f"Left Encoder Pulses: {left_counter - initial_left_counter}")

# Realignment function
def realignment(right_pwm, left_pwm, servo_pwm, target_servo_angle):
    global right_counter, left_counter, initial_servo_angle, current_servo_angle
    
    # Initial setup
    initial_right_counter = right_counter
    initial_left_counter = left_counter
    initial_servo_angle = current_servo_angle
    
    # First, turn the servo to the target angle
    set_servo_angle(servo_pwm, target_servo_angle)
    
    # Calculate the angular displacement needed
    angle_difference = abs(target_servo_angle - initial_servo_angle)
    
    # Determine turn direction based on angle difference
    if target_servo_angle > initial_servo_angle:
        # Turn right
        turn_right(right_pwm, left_pwm, 50)  # 50% speed
    else:
        # Turn left
        turn_left(right_pwm, left_pwm, 50)  # 50% speed
    
    # Calculate encoder pulses needed for rotation
    circumference = math.pi * ROBOT_WIDTH
    rotation_portion = angle_difference / 360.0
    required_distance = circumference * rotation_portion
    
    # Estimate pulses needed (this will need precise calibration)
    estimated_pulses = int(required_distance / WHEEL_CIRCUMFERENCE * PULSES_PER_REVOLUTION)
    
    # Wait until we've approximately rotated the right amount
    start_time = time.time()
    while (abs(right_counter - initial_right_counter) < estimated_pulses and 
           abs(left_counter - initial_left_counter) < estimated_pulses):
        time.sleep(0.1)
        if time.time() - start_time > 5:  # Timeout after 5 seconds
            break
    
    # Stop motors
    stop_motors(right_pwm, left_pwm)
    
    # Return servo to initial position
    set_servo_angle(servo_pwm, initial_servo_angle)
    
    # Print movement stats for verification
    print("Realignment Complete:")
    print(f"Initial Servo Angle: {initial_servo_angle}")
    print(f"Target Servo Angle: {target_servo_angle}")
    print(f"Right Encoder Pulses: {right_counter - initial_right_counter}")
    print(f"Left Encoder Pulses: {left_counter - initial_left_counter}")

# Print movement stats
def print_movement_stats():
    global right_counter, left_counter
    
    # Calculate distances
    right_distance = calculate_distance(right_counter)
    left_distance = calculate_distance(left_counter)
    distance_difference = abs(right_distance - left_distance)
    
    print(f"Right Encoder Pulses: {right_counter}")
    print(f"Left Encoder Pulses: {left_counter}")
    print(f"Right Encoder Distance: {right_distance:.2f} cm")
    print(f"Left Encoder Distance: {left_distance:.2f} cm")
    print(f"Difference: {distance_difference:.2f} cm")

# Main loop for manual control
def manual_control():
    global right_counter, left_counter
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    
    print("\n==== Robot Movement Testing Program ====")
    print("Commands:")
    print("  f <speed> - Move forward")
    print("  b <speed> - Move backward")
    print("  r <speed> - Turn right")
    print("  l <speed> - Turn left")
    print("  s - Stop motors")
    print("  sv <angle> - Set servo angle (0-180)")
    print("  ra <angle> - Realign robot with servo")
    print("  re - Rotate to servo angle")
    print("  t <time> - Set movement time (seconds)")
    print("  q - Quit")
    
    movement_time = 1.0  # Default movement time in seconds
    
    try:
        while True:
            command = input("\nEnter command (f/b/r/l/s/sv/ra/re/t/q): ").strip().lower()
            
            if command == 'q':
                break
                
            if command == 's':
                stop_motors(right_pwm, left_pwm)
                print("Motors stopped.")
                continue
                
            if command.startswith('t '):
                try:
                    movement_time = float(command.split()[1])
                    print(f"Movement time set to {movement_time:.1f} seconds")
                except (ValueError, IndexError):
                    print("Invalid time value. Please use format 't <seconds>'")
                continue
                
            if command.startswith('sv '):
                try:
                    angle = float(command.split()[1])
                    set_servo_angle(servo_pwm, angle)
                    print(f"Servo set to {angle} degrees")
                except (ValueError, IndexError):
                    print("Invalid angle value. Please use format 'sv <angle>'")
                continue
                
            if command.startswith('ra '):
                try:
                    angle = float(command.split()[1])
                    realignment(right_pwm, left_pwm, servo_pwm, angle)
                    print(f"Realignment to {angle} degrees complete")
                except (ValueError, IndexError):
                    print("Invalid angle value. Please use format 'ra <angle>'")
                continue
            
            if command == 're':
                rotate_to_servo_angle(right_pwm, left_pwm, servo_pwm)
                continue
                
            # For movement commands
            try:
                cmd_parts = command.split()
                if len(cmd_parts) != 2:
                    print("Please use format '<command> <speed>'")
                    continue
                    
                cmd, speed = cmd_parts
                speed = float(speed)
                
                if not (0 <= speed <= 100):
                    print("Speed must be between 0 and 100")
                    continue
                
                # Reset encoder counts
                right_counter = 0
                left_counter = 0
                
                # Execute movement based on command
                if cmd == 'f':
                    print(f"Moving forward at {speed}% for {movement_time:.1f} seconds")
                    move_forward(right_pwm, left_pwm, speed)
                elif cmd == 'b':
                    print(f"Moving backward at {speed}% for {movement_time:.1f} seconds")
                    move_backward(right_pwm, left_pwm, speed)
                elif cmd == 'r':
                    print(f"Turning right at {speed}% for {movement_time:.1f} seconds")
                    turn_right(right_pwm, left_pwm, speed)
                elif cmd == 'l':
                    print(f"Turning left at {speed}% for {movement_time:.1f} seconds")
                    turn_left(right_pwm, left_pwm, speed)
                else:
                    print(f"Unknown command: {cmd}")
                    continue
                
                # Run for specified time
                time.sleep(movement_time)
                stop_motors(right_pwm, left_pwm)
                print("Motors stopped.")
                
                # Print movement statistics
                print_movement_stats()
                
            except (ValueError, IndexError):
                print("Invalid command format. Use '<command> <speed>'")
    
    except KeyboardInterrupt:
        print("\nManual control stopped by user.")
    finally:
        stop_motors(right_pwm, left_pwm)
        servo_pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up.")

# Run the manual control function
if __name__ == "__main__":
    manual_control()