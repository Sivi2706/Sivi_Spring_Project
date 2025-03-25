import RPi.GPIO as GPIO
import time
import numpy as np

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
encoderPinRight = 23       # Right encoder
encoderPinLeft = 24        # Left encoder
ServoMotor = 18            # Servo motor PWM

# Constants (to be calibrated)
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5       # Duty cycle for -90° (full left)
SERVO_MAX_DUTY = 12.5      # Duty cycle for +90° (full right)
SERVO_CENTER_DUTY = 7.5    # Duty cycle for 0° (center)
SERVO_FREQ = 50            # 50Hz frequency for servo

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
    set_servo_angle(servo_pwm, 0)  # Center servo at startup
    
    return right_pwm, left_pwm, servo_pwm

# Servo control function (-90° to +90°)
def set_servo_angle(servo_pwm, angle):
    angle = max(-90, min(90, angle))  # Constrain to -90° to +90°
    
    if angle == 0:
        duty = SERVO_CENTER_DUTY
    elif angle < 0:  # Left turn
        duty = SERVO_CENTER_DUTY + (angle * (SERVO_CENTER_DUTY - SERVO_MIN_DUTY) / 90.0)
    else:  # Right turn
        duty = SERVO_CENTER_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_CENTER_DUTY) / 90.0)
    
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Allow time to move
    servo_pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter
    print(f"Servo: {angle}° (Duty: {duty:.1f}%)")

# Movement functions
def move_forward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)
    print(f"Moving forward at {speed}%")

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)
    print(f"Moving backward at {speed}%")

def turn_right(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)
    print(f"Turning right at {speed}%")

def turn_left(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)
    print(f"Turning left at {speed}%")

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    print("Motors stopped")

# Calculate distance from encoder counts
def calculate_distance(encoder_count):
    revolutions = encoder_count / PULSES_PER_REVOLUTION
    return revolutions * WHEEL_CIRCUMFERENCE

# Print movement statistics
def print_movement_stats():
    right_distance = calculate_distance(right_counter)
    left_distance = calculate_distance(left_counter)
    print("\nMovement Statistics:")
    print(f"Right Encoder: {right_counter} pulses, {right_distance:.2f} cm")
    print(f"Left Encoder: {left_counter} pulses, {left_distance:.2f} cm")
    print(f"Difference: {abs(right_distance - left_distance):.2f} cm")

# Main control loop
def manual_control():
    global right_counter, left_counter
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    
    print("\n==== Robot Control Program ====")
    print("Commands:")
    print("  f <speed> - Move forward (0-100%)")
    print("  b <speed> - Move backward (0-100%)")
    print("  r <speed> - Turn right (0-100%)")
    print("  l <speed> - Turn left (0-100%)")
    print("  s - Stop motors")
    print("  sv <angle> - Set servo angle (-90° to +90°)")
    print("  t <seconds> - Set movement duration")
    print("  p - Print movement stats")
    print("  q - Quit program")
    
    movement_time = 1.0  # Default movement duration
    
    try:
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'q':
                break
                
            elif command == 's':
                stop_motors(right_pwm, left_pwm)
                
            elif command == 'p':
                print_movement_stats()
                
            elif command.startswith('t '):
                try:
                    movement_time = float(command.split()[1])
                    print(f"Movement duration set to {movement_time:.1f} seconds")
                except:
                    print("Invalid time value")
                    
            elif command.startswith('sv '):
                try:
                    angle = float(command.split()[1])
                    set_servo_angle(servo_pwm, angle)
                except:
                    print("Invalid angle value")
                    
            elif len(command.split()) == 2:  # Movement commands
                cmd, speed = command.split()
                try:
                    speed = float(speed)
                    if not 0 <= speed <= 100:
                        print("Speed must be 0-100")
                        continue
                        
                    # Reset encoders
                    right_counter = 0
                    left_counter = 0
                    
                    # Execute movement
                    if cmd == 'f':
                        move_forward(right_pwm, left_pwm, speed)
                    elif cmd == 'b':
                        move_backward(right_pwm, left_pwm, speed)
                    elif cmd == 'r':
                        turn_right(right_pwm, left_pwm, speed)
                    elif cmd == 'l':
                        turn_left(right_pwm, left_pwm, speed)
                    else:
                        print("Unknown command")
                        continue
                    
                    # Run for specified time
                    time.sleep(movement_time)
                    stop_motors(right_pwm, left_pwm)
                    print_movement_stats()
                    
                except ValueError:
                    print("Invalid speed value")
            else:
                print("Unknown command")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    finally:
        stop_motors(right_pwm, left_pwm)
        servo_pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up. Program ended.")

if __name__ == "__main__":
    manual_control()