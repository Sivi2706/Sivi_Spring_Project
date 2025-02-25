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
wheelDiameter = 6.5  # Wheel diameter in centimeters
wheelCircumference = wheelDiameter * math.pi  # Wheel circumference in centimeters

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
    # Calculate distance for each wheel
    distanceRight = pulsesRight * wheelCircumference
    distanceLeft = pulsesLeft * wheelCircumference
    # Calculate average distance
    averageDistance = (distanceRight + distanceLeft) / 2.0
    return distanceRight, distanceLeft, averageDistance

def reset_distance():
    global pulsesRight, pulsesLeft
    pulsesRight = 0
    pulsesLeft = 0

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
    time.sleep(1)  # Move for 1 second
    stop_motors()  # Stop the motors after moving
    rightDist, leftDist, avgDist = get_moving_distance()
    print(f"Forward - Right: {rightDist:.4f}cm, Left: {leftDist:.4f}cm, Avg: {avgDist:.4f}cm")

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
    # Setup rotary encoders
    encoder_setup()

    # PWM demonstration
    pwm_demo()

    # Move forward for 1 second and print distances
    print("Moving forward for 1 second...")
    move_forward(50)  # Move forward with 50% duty cycle

except KeyboardInterrupt:
    print("Program interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Stop the motors and clean up GPIO
    print("Stopping motors and cleaning up GPIO...")
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete.")