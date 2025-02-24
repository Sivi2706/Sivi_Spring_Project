import RPi.GPIO as GPIO
import time

# Define GPIO pins for motor control
IN1 = 5 # IN1
IN2 = 7  # IN2
IN3 = 11  # IN3
IN4 = 13  # IN4
ENA = 3  # Enable Pin for Motor A
ENB = 15  # Enable Pin for Motor B

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Setup PWM
pwm_a = GPIO.PWM(ENA, 100)  # Create PWM instance for Motor A at 100Hz
pwm_b = GPIO.PWM(ENB, 100)  # Create PWM instance for Motor B at 100Hz
pwm_a.start(0)  # Start PWM with 0% duty cycle
pwm_b.start(0)  # Start PWM with 0% duty cycle

def set_speed(speed):
    """
    Set speed for both motors
    speed: 0-100 (percentage of maximum speed)
    """
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

# Function to stop the car
def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(0)

# Function to move the car forward
def forward(speed=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(speed)

# Function to move the car backward
def backward(speed=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    set_speed(speed)

# Function to turn the car left
def left(speed=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(speed)

# Function to turn the car right
def right(speed=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    set_speed(speed)

# Main program
try:
    while True:
        command = input("Enter command (f: forward, b: backward, l: left, r: right, s: stop)\n"
                       "Optional: Add speed (0-100) after command (e.g., 'f 75'): ").strip().lower()
        
        parts = command.split()
        cmd = parts[0]
        speed = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 50
        
        # Ensure speed is within valid range
        speed = max(0, min(100, speed))
        
        if cmd == 'f':
            forward(speed)
        elif cmd == 'b':
            backward(speed)
        elif cmd == 'l':
            left(speed)
        elif cmd == 'r':
            right(speed)
        elif cmd == 's':
            stop()
        else:
            print("Invalid command")

except KeyboardInterrupt:
    print("Program stopped")

finally:
    # Clean up GPIO on exit
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()