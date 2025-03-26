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
WHEEL_BASE_DISTANCE = 12.0  # cm, distance between wheels (calibrate this)

# Servo motor parameters
SERVO_MIN_DUTY = 2.5      # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5     # Duty cycle for 180 degrees
SERVO_FREQ = 50           # 50Hz frequency for servo

# Global variables
right_counter = 0
left_counter = 0
current_servo_angle = 90  # Default servo position

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

# Function to calculate rotation angle from encoder counts
def calculate_rotation_angle():
    global right_counter, left_counter
    right_dist = calculate_distance(right_counter)
    left_dist = calculate_distance(left_counter)
    delta_dist = right_dist - left_dist
    rotation_rad = delta_dist / WHEEL_BASE_DISTANCE
    rotation_deg = math.degrees(rotation_rad)
    return rotation_deg

# Print movement stats
def print_movement_stats():
    global right_counter, left_counter
    
    right_distance = calculate_distance(right_counter)
    left_distance = calculate_distance(left_counter)
    rotation_angle = calculate_rotation_angle()
    
    print(f"Right Encoder Pulses: {right_counter}")
    print(f"Left Encoder Pulses: {left_counter}")
    print(f"Right Distance: {right_distance:.2f} cm")
    print(f"Left Distance: {left_distance:.2f} cm")
    print(f"Rotation Angle: {rotation_angle:.2f} degrees")

# Realignment function
def Realignment(servo_pwm, right_pwm, left_pwm, original_angle):
    global right_counter, left_counter, current_servo_angle
    
    target_angle = current_servo_angle  # Angle we're trying to align to
    required_rotation = original_angle - target_angle
    
    if required_rotation == 0:
        print("No realignment needed.")
        return
    
    # Determine direction and absolute rotation
    if required_rotation > 0:
        direction = 'right'
        turn_func = turn_right
    else:
        direction = 'left'
        required_rotation = -required_rotation
        turn_func = turn_left
    
    speed = 50  # Adjust speed as needed
    right_counter = 0
    left_counter = 0
    
    print(f"Realigning: Turning {direction} until servo returns to {original_angle}°")
    
    # Start turning
    turn_func(right_pwm, left_pwm, speed)
    
    try:
        while True:
            # Calculate current rotation
            current_rotation = calculate_rotation_angle()
            if direction == 'left':
                current_rotation = -current_rotation
            
            # Calculate how much we've turned toward the target
            progress = current_rotation / required_rotation
            
            # Calculate new servo angle (approaching original)
            new_servo_angle = target_angle + (original_angle - target_angle) * progress
            new_servo_angle = max(0, min(180, new_servo_angle))
            set_servo_angle(servo_pwm, new_servo_angle)
            
            # Check if we've completed the rotation
            if current_rotation >= required_rotation:
                break
            
            time.sleep(0.1)
            
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle(servo_pwm, original_angle)
        print("Realignment complete.")
        print_movement_stats()

# Main loop for manual control
def manual_control():
    global right_counter, left_counter, current_servo_angle
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    
    # Initialize servo to center position
    set_servo_angle(servo_pwm, current_servo_angle)
    
    print("\n==== Robot Movement Testing Program ====")
    print("Commands:")
    print("  f <speed> - Move forward")
    print("  b <speed> - Move backward")
    print("  r <speed> - Turn right")
    print("  l <speed> - Turn left")
    print("  s - Stop motors")
    print("  sv <angle> - Set servo angle and realign (0-180)")
    print("  t <time> - Set movement time (seconds)")
    print("  q - Quit")
    
    movement_time = 1.0  # Default movement time in seconds
    
    try:
        while True:
            command = input("\nEnter command (f/b/r/l/s/sv/t/q): ").strip().lower()
            
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
                    original_angle = current_servo_angle  # Save original angle
                    set_servo_angle(servo_pwm, angle)
                    Realignment(servo_pwm, right_pwm, left_pwm, original_angle)
                except (ValueError, IndexError):
                    print("Invalid angle value. Please use format 'sv <angle>'")
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