import RPi.GPIO as GPIO
import time
import numpy as np

# Define GPIO pins
IN1, IN2 = 4, 17          # Right motor control
IN3, IN4 = 27, 22         # Left motor control
ENA, ENB = 18, 23         # PWM pins for motors
encoderPinRight = 24      # Right encoder
encoderPinLeft = 25       # Left encoder

# Constants (to be calibrated)
WHEEL_DIAMETER = 6.5  # cm
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
    
    # Set up PWM
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    
    right_pwm.start(0)
    left_pwm.start(0)
    
    return right_pwm, left_pwm

# Function to move forward
def move_forward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

# Function to stop motors
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

# Main loop for manual control
def manual_pwm_control():
    global right_counter, left_counter
    right_pwm, left_pwm = setup_gpio()
    
    try:
        while True:
            pwm_value = input("Enter PWM value (0-100) or 'q' to quit: ")
            
            if pwm_value.lower() == 'q':
                break
            
            try:
                pwm_value = float(pwm_value)
                if 0 <= pwm_value <= 100:
                    print(f"Running motors at PWM: {pwm_value}")
                    
                    # Reset encoder counts
                    right_counter = 0
                    left_counter = 0
                    
                    move_forward(right_pwm, left_pwm, pwm_value)
                    time.sleep(1)  # Run for 1 second
                    stop_motors(right_pwm, left_pwm)
                    
                    # Calculate distances
                    right_distance = calculate_distance(right_counter)
                    left_distance = calculate_distance(left_counter)
                    distance_difference = abs(right_distance - left_distance)
                    
                    print(f"Right Encoder Pulses: {right_counter}")
                    print(f"Left Encoder Pulses: {left_counter}")
                    print(f"Right Encoder Distance: {right_distance:.2f} cm")
                    print(f"Left Encoder Distance: {left_distance:.2f} cm")
                    print(f"Difference: {distance_difference:.2f} cm")
                    print("Motors stopped. Enter new PWM value.")
                else:
                    print("Please enter a value between 0 and 100.")
            except ValueError:
                print("Invalid input. Enter a number between 0 and 100.")
    
    except KeyboardInterrupt:
        print("\nManual control stopped by user.")
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up.")

# Run the manual control function
if __name__ == "__main__":
    manual_control()

    manual_pwm_control()

