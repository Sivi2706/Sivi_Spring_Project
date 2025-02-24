import RPi.GPIO as GPIO
import time
#Testing Raspberry pi 
# Define GPIO pins for the motor driver
IN1 = 3
IN2 = 4
IN3 = 17
IN4 = 27
ENA = 2
ENB = 22

# Set up the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set up the GPIO pins as outputs
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Set up PWM for speed control
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(100)  # Set speed to 100%
pwmB.start(100)  # Set speed to 100%

def move_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    GPIO.output(ENA, x)
    GPIO.output(ENB. y)

try:
    move_forwar(x, y)
    print("Moving forward")
    time.sleep(5)  # Move forward for 5 seconds

finally:
    # Stop the motors
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.stop()
    pwmB.stop()
GPIO.cleanup()
# Testting 2 