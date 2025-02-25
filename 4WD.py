import RPi.GPIO as GPIO
import time
import math

# Define GPIO pins for the motor driver
IN1 = 3
IN2 = 4
IN3 = 17
IN4 = 27
ENA = 2
ENB = 22

# Define GPIO pins for the rotary encoder
encoderPinA = 2

# Set up the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set up the GPIO pins as outputs
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Set up PWM for duty cycle control
pwmA = GPIO.PWM(ENA, 100)  # Using 100Hz as base frequency
pwmB = GPIO.PWM(ENB, 100)  # Using 100Hz as base frequency
pwmA.start(0)  # Start with 0% duty cycle (stopped)
pwmB.start(0)  # Start with 0% duty cycle (stopped)

# Rotary encoder variables
pulses = 0
wheelCircumference = 0.05 * math.pi * 0.0325 * 100

def counter_update(channel):
    global pulses
    pulses += 1

def encoder_setup():
    GPIO.setup(encoderPinA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(encoderPinA, GPIO.RISING, callback=counter_update)

def get_moving_distance():
    global pulses
    distance = pulses * wheelCircumference
    return distance

def reset_distance():
    global pulses
    pulses = 0

def move_forward(duty_cycle):
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal forward duty cycle is between 45-55%")
    reset_distance()  # Reset distance counter before starting movement
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)  # Allow some time for movement
    print(f"Distance traveled forward: {get_moving_distance()} meters")

def move_backward(duty_cycle):
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal backward duty cycle is between 45-55%")
    reset_distance()  # Reset distance counter before starting movement
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)  # Allow some time for movement
    print(f"Distance traveled backward: {get_moving_distance()} meters")

def turn_left(duty_cycle):
    if not (85 <= duty_cycle <= 100):
        print("Warning: Optimal left turn duty cycle is between 85-100%")
    reset_distance()  # Reset distance counter before starting movement
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)  # Allow some time for movement
    print(f"Distance traveled during left turn: {get_moving_distance()} meters")

def turn_right(duty_cycle):
    if not (75 <= duty_cycle <= 85):
        print("Warning: Optimal right turn duty cycle is between 75-85%")
    reset_distance()  # Reset distance counter before starting movement
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)  # Allow some time for movement
    print(f"Distance traveled during right turn: {get_moving_distance()} meters")

def stop_motors():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

try:
    # Setup rotary encoder
    encoder_setup()

    # Example usage with optimal duty cycles from the data table
    print("Moving forward")
    move_forward(50)  # 50% duty cycle for forward movement
    time.sleep(2)  # Wait for 2 seconds before next command

    print("Moving backward")
    move_backward(50)  # 50% duty cycle for backward movement
    time.sleep(2)  # Wait for 2 seconds before next command

    print("Turning left")
    turn_left(90)  # 90% duty cycle for left turn
    time.sleep(2)  # Wait for 2 seconds before next command

    print("Turning right")
    turn_right(80)  # 80% duty cycle for right turn
    time.sleep(2)  # Wait for 2 seconds before next command

except KeyboardInterrupt:
    print("Program interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop the motors and clean up
    print("Stopping motors and cleaning up GPIO...")
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete.")