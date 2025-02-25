import RPi.GPIO as GPIO
import time
import math

# Define GPIO pins for the motor driver
IN1 = 4
IN2 = 17
IN3 = 27
IN4 = 22
ENA = 18
ENB = 23

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
pwmA = GPIO.PWM(ENA, 100)  # Using 100Hz as base frequency
pwmB = GPIO.PWM(ENB, 100)  # Using 100Hz as base frequency
pwmA.start(0)  # Start with 0% duty cycle (stopped)
pwmB.start(0)  # Start with 0% duty cycle (stopped)

# Rotary encoder variables
pulsesRight = 0
pulsesLeft = 0
wheelCircumference = 0.05 * math.pi * 0.0325 * 100  # Example calculation, adjust based on actual wheel specs

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
    stop_motors()  # Stop the motors after moving for 1 second
    rightDist, leftDist, avgDist = get_moving_distance()
    print(f"Forward - Right: {rightDist:.4f}m, Left: {leftDist:.4f}m, Avg: {avgDist:.4f}m")

def stop_motors():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

try:
    encoder_setup()
    print("Moving forward for 1 second...")
    move_forward(50)  # This will move forward, stop after 1s, and print distances

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