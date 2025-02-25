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

# Robot physical specifications
wheelDiameter = 6.5  # Wheel diameter in centimeters
wheelCircumference = wheelDiameter * math.pi  # Wheel circumference in centimeters

# Added: Number of pulses per revolution - you need to adjust this value 
# based on your encoder specifications
PULSES_PER_REVOLUTION = 20  # Example value - replace with your encoder's actual PPR

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
    
    # Calculate revolutions for each wheel
    revolutionsRight = pulsesRight / PULSES_PER_REVOLUTION
    revolutionsLeft = pulsesLeft / PULSES_PER_REVOLUTION
    
    # Calculate distance for each wheel
    distanceRight = revolutionsRight * wheelCircumference
    distanceLeft = revolutionsLeft * wheelCircumference
    
    # Calculate average distance
    averageDistance = (distanceRight + distanceLeft) / 2.0
    
    return revolutionsRight, revolutionsLeft, distanceRight, distanceLeft, averageDistance

def reset_distance():
    global pulsesRight, pulsesLeft
    pulsesRight = 0
    pulsesLeft = 0

def move_forward(duty_cycle, duration=1.0):
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal forward duty cycle is between 45-55%")
    
    reset_distance()  # Reset distance counter before starting movement
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    
    # Live monitoring of distance during movement
    start_time = time.time()
    end_time = start_time + duration
    
    try:
        while time.time() < end_time:
            revRight, revLeft, distRight, distLeft, avgDist = get_moving_distance()
            print(f"\rLive: Rev R: {revRight:.2f}, Rev L: {revLeft:.2f} | Dist R: {distRight:.2f}cm, Dist L: {distLeft:.2f}cm, Avg: {avgDist:.2f}cm", end="")
            time.sleep(0.1)  # Update every 0.1 seconds
    except KeyboardInterrupt:
        pass
    finally:
        stop_motors()  # Stop the motors after moving
        revRight, revLeft, distRight, distLeft, avgDist = get_moving_distance()
        print(f"\nFinal: Rev R: {revRight:.2f}, Rev L: {revLeft:.2f} | Dist R: {distRight:.2f}cm, Dist L: {distLeft:.2f}cm, Avg: {avgDist:.2f}cm")

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

    # Move forward for 3 seconds and print distances with live updates
    print("Moving forward with live distance monitoring...")
    move_forward(50, 3.0)  # Move forward with 50% duty cycle for 3 seconds

    # Optional: Add more movement commands here if needed
    
except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Stop the motors and clean up GPIO
    print("Stopping motors and cleaning up GPIO...")
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete.")