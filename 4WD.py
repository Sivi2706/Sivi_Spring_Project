import RPi.GPIO as GPIO
import time
import math

# Define GPIO pins for the motor driver
IN1 = 4
IN2 = 17
IN3 = 27
IN4 = 22
ENA = 18  # PWM pin for motor A
ENB = 23  # PWM pin for motor B

# Define GPIO pins for the rotary encoders
encoderPinRight = 24  # Right encoder
encoderPinLeft = 25   # Left encoder

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
pwmA = GPIO.PWM(ENA, 100)  # Using 100Hz as base frequency for motor A
pwmB = GPIO.PWM(ENB, 100)  # Using 100Hz as base frequency for motor B
pwmA.start(0)  # Start with 0% duty cycle (stopped)
pwmB.start(0)  # Start with 0% duty cycle (stopped)

# Rotary encoder variables
pulsesRight = 0
pulsesLeft = 0
wheelCircumference = 0.05 * math.pi * 0.0325 * 100  # Wheel circumference in meters

def counter_update_right(channel):
    global pulsesRight
    pulsesRight += 1

def counter_update_left(channel):
    global pulsesLeft
    pulsesLeft += 1

def encoder_setup():
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=counter_update_right)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=counter_update_left)

def get_moving_distance():
    global pulsesRight, pulsesLeft
    distanceRight = pulsesRight * wheelCircumference
    distanceLeft = pulsesLeft * wheelCircumference
    averageDistance = (distanceRight + distanceLeft) / 2.0
    return distanceRight, distanceLeft, averageDistance

def reset_distance():
    global pulsesRight, pulsesLeft
    pulsesRight = 0
    pulsesLeft = 0

def move_forward(duty_cycle):
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal forward duty cycle is between 45-55%")
    reset_distance()
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    rightDist, leftDist, avgDist = get_moving_distance()
    print(f"Forward - Right: {rightDist:.4f}m, Left: {leftDist:.4f}m, Avg: {avgDist:.4f}m")

def move_backward(duty_cycle):
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal backward duty cycle is between 45-55%")
    reset_distance()
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    rightDist, leftDist, avgDist = get_moving_distance()
    print(f"Backward - Right: {rightDist:.4f}m, Left: {leftDist:.4f}m, Avg: {avgDist:.4f}m")

def turn_left(duty_cycle):
    if not (85 <= duty_cycle <= 100):
        print("Warning: Optimal left turn duty cycle is between 85-100%")
    reset_distance()
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    rightDist, leftDist, avgDist = get_moving_distance()
    print(f"Left Turn - Right: {rightDist:.4f}m, Left: {leftDist:.4f}m, Avg: {avgDist:.4f}m")

def turn_right(duty_cycle):
    if not (75 <= duty_cycle <= 85):
        print("Warning: Optimal right turn duty cycle is between 75-85%")
    reset_distance()
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    time.sleep(1)
    rightDist, leftDist, avgDist = get_moving_distance()
    print(f"Right Turn - Right: {rightDist:.4f}m, Left: {leftDist:.4f}m, Avg: {avgDist:.4f}m")

def stop_motors():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

# PWM demonstration for motor speed control
def pwm_demo():
    print("Starting PWM demonstration for motor speed control...")
    for x in range(50):  # Increase duty cycle from 0% to 50%
        pwmA.ChangeDutyCycle(x)
        pwmB.ChangeDutyCycle(x)
        time.sleep(0.1)
    for x in range(50):  # Decrease duty cycle from 50% to 0%
        pwmA.ChangeDutyCycle(50 - x)
        pwmB.ChangeDutyCycle(50 - x)
        time.sleep(0.1)
    print("PWM demonstration complete.")

try:
    encoder_setup()
    print("Starting motor control with PWM...")

    # PWM demonstration
    pwm_demo()

    # Motor control with distance measurement
    print("Moving forward")
    move_forward(50)
    time.sleep(2)

    print("Moving backward")
    move_backward(50)
    time.sleep(2)

    print("Turning left")
    turn_left(90)
    time.sleep(2)

    print("Turning right")
    turn_right(80)
    time.sleep(2)

except KeyboardInterrupt:
    print("Program interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Stopping motors and cleaning up GPIO...")
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete.")