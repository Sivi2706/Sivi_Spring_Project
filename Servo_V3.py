import RPi.GPIO as GPIO
import time
import numpy as np

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera
BLACK_SENSOR_PIN = 25     # Pin for the black line sensor

# Constants (to be calibrated)
WHEEL_DIAMETER = 4.05  # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5     # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5    # Duty cycle for 180 degrees
SERVO_FREQ = 50          # 50Hz frequency for servo

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Encoder callback functions
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

# Function to detect black line (placeholder logic)
def detect_black_line():
    # This function should return True when the black line is detected.
    # For example, if your sensor outputs LOW when detecting black:
    return GPIO.input(BLACK_SENSOR_PIN) == GPIO.LOW

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
    
    # Black line sensor pin
    GPIO.setup(BLACK_SENSOR_PIN, GPIO.IN)
    
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

# Function to calculate distance from encoder counts
def calculate_distance(encoder_count):
    revolutions = encoder_count / PULSES_PER_REVOLUTION
    distance = revolutions * WHEEL_CIRCUMFERENCE
    return distance

# Print movement stats
def print_movement_stats():
    global right_counter, left_counter
    
    right_distance = calculate_distance(right_counter)
    left_distance = calculate_distance(left_counter)
    distance_difference = abs(right_distance - left_distance)
    
    print(f"Right Encoder Pulses: {right_counter}")
    print(f"Left Encoder Pulses: {left_counter}")
    print(f"Right Encoder Distance: {right_distance:.2f} cm")
    print(f"Left Encoder Distance: {left_distance:.2f} cm")
    print(f"Difference: {distance_difference:.2f} cm")

# Modified servo function that turns the car toward the specified angle and resets the servo to 90°
def set_servo_angle(servo_pwm, angle, right_pwm, left_pwm):
    # Constrain angle between 0 and 180
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    
    # Move servo to the specified angle
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Allow time for servo movement
    servo_pwm.ChangeDutyCycle(0)  # Prevent jitter

    # Calculate turn time based on deviation from center (90°)
    # Assumption: 45° turn takes 1 second.
    turn_time = abs(angle - 90) / 45.0

    # Determine turn direction using convention: 180 (left), 90 (center), 0 (right)
    if angle > 90:
        print(f"Turning left for {turn_time:.2f} seconds")
        turn_left(right_pwm, left_pwm, 50)
    elif angle < 90:
        print(f"Turning right for {turn_time:.2f} seconds")
        turn_right(right_pwm, left_pwm, 50)
    else:
        print("Angle is 90°, no turning required.")
        return

    time.sleep(turn_time)
    stop_motors(right_pwm, left_pwm)

    # Reset servo to 90°
    print("Returning servo to 90°")
    duty = SERVO_MIN_DUTY + (90 * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

# New scanning function:
# While reversing at low speed, the servo sweeps from 180° (left) to 0° (right).
# When the black line is detected via the sensor, the current servo angle is recorded.
# The car stops, then rotates using the servo tuning logic, and finally resumes forward movement.
def scan_for_line(servo_pwm, right_pwm, left_pwm):
    print("Initiating reverse and scan for black line...")
    # Start reversing slowly (adjust speed as necessary)
    move_backward(right_pwm, left_pwm, 30)
    
    detected_angle = None
    # Sweep from 180 (left) to 0 (right) in steps (e.g., every 5°)
    for angle in range(180, -1, -5):
        duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
        servo_pwm.ChangeDutyCycle(duty)
        time.sleep(0.2)  # Allow servo to move and sensor to stabilize
        servo_pwm.ChangeDutyCycle(0)
        
        # Check for black line detection
        if detect_black_line():
            detected_angle = angle
            print(f"Black line detected at angle {angle}°")
            break
    
    # Stop the reversing motion regardless of detection outcome
    stop_motors(right_pwm, left_pwm)
    
    if detected_angle is None:
        print("Black line not detected during scan. Resuming normal operation.")
        return
    else:
        # Use the servo tuning logic to rotate the car toward the detected angle.
        set_servo_angle(servo_pwm, detected_angle, right_pwm, left_pwm)
        
        # Resume forward movement as part of line following
        print("Resuming line following (moving forward)...")
        move_forward(right_pwm, left_pwm, 50)
        time.sleep(1)  # Adjust forward duration as needed
        stop_motors(right_pwm, left_pwm)

# Main loop for manual control (includes new 'scan' command)
def manual_control():
    global right_counter, left_counter
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    
    print("\n==== Robot Movement Testing Program ====")
    print("Commands:")
    print("  f <speed>   - Move forward")
    print("  b <speed>   - Move backward")
    print("  r <speed>   - Turn right")
    print("  l <speed>   - Turn left")
    print("  s           - Stop motors")
    print("  sv <angle>  - Set servo angle and turn car (0-180)")
    print("  scan        - Reverse, scan for black line, and adjust heading")
    print("  t <time>    - Set movement time (seconds)")
    print("  q           - Quit")
    
    movement_time = 1.0  # Default movement time (in seconds)
    
    try:
        while True:
            command = input("\nEnter command (f/b/r/l/s/sv/scan/t/q): ").strip().lower()
            
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
                    print("Invalid time value. Use format 't <seconds>'")
                continue
            
            if command.startswith('sv '):
                try:
                    angle = float(command.split()[1])
                    set_servo_angle(servo_pwm, angle, right_pwm, left_pwm)
                    print(f"Servo moved to {angle}°, car turned accordingly, and servo reset to 90°")
                except (ValueError, IndexError):
                    print("Invalid angle value. Use format 'sv <angle>'")
                continue
            
            if command == 'scan':
                # Execute the reverse/scan routine
                scan_for_line(servo_pwm, right_pwm, left_pwm)
                continue
            
            # For movement commands (f, b, r, l)
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
                
                time.sleep(movement_time)
                stop_motors(right_pwm, left_pwm)
                print("Motors stopped.")
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

if __name__ == "__main__":
    manual_control()
