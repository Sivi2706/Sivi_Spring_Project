import RPi.GPIO as GPIO
import time
import numpy as np

# Define GPIO pins
IN1, IN2 = 22, 27         # Right motor control
IN3, IN4 = 17, 4            # Left motor control
ENA, ENB = 13, 12           # PWM pins for motors
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24        # Left encoder
ServoMotor =  18           # Servo motor PWM for the camera 

# Constants (to be calibrated)
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Encoder callback functions
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    # Only count the rising edge for the left encoder
    if GPIO.input(encoderPinLeft) == GPIO.HIGH:
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
    GPIO.add_event_detect(encoderPinLeft, GPIO.BOTH, callback=left_encoder_callback)  # Use BOTH for left encoder
    
    # Set up PWM
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    
    right_pwm.start(0)
    left_pwm.start(0)
    
    return right_pwm, left_pwm

# Movement functions
def move_forward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_right(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_left(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def pivot_right(right_pwm, left_pwm, speed):
    # Right wheel stopped, left wheel forward
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(speed)

def pivot_left(right_pwm, left_pwm, speed):
    # Left wheel stopped, right wheel forward
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(0)

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
    right_pwm, left_pwm = setup_gpio()
    
    print("\n==== Robot Movement Testing Program ====")
    print("Commands:")
    print("  f <speed> - Move forward")
    print("  b <speed> - Move backward")
    print("  r <speed> - Turn right (both motors)")
    print("  l <speed> - Turn left (both motors)")
    print("  pr <speed> - Pivot right (left motor only)")
    print("  pl <speed> - Pivot left (right motor only)")
    print("  s - Stop motors")
    print("  t <time> - Set movement time (seconds)")
    print("  q - Quit")
    
    movement_time = 1.0  # Default movement time in seconds
    
    try:
        while True:
            command = input("\nEnter command (f/b/r/l/pr/pl/s/t/q): ").strip().lower()
            
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
                elif cmd == 'pr':
                    print(f"Pivoting right at {speed}% for {movement_time:.1f} seconds")
                    pivot_right(right_pwm, left_pwm, speed)
                elif cmd == 'pl':
                    print(f"Pivoting left at {speed}% for {movement_time:.1f} seconds")
                    pivot_left(right_pwm, left_pwm, speed)
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
        GPIO.cleanup()
        print("GPIO cleaned up.")

# Run the manual control function
if __name__ == "__main__":
    