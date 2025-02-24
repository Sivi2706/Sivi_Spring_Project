import RPi.GPIO as GPIO
import time

#===================================Affects of Duty Cycle code==================================

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

# Set up PWM for duty cycle control
# The frequency is less important here as we're focusing on duty cycle
pwmA = GPIO.PWM(ENA, 100)  # Using 100Hz as base frequency
pwmB = GPIO.PWM(ENB, 100)  # Using 100Hz as base frequency
pwmA.start(0)  # Start with 0% duty cycle (stopped)
pwmB.start(0)  # Start with 0% duty cycle (stopped)

def move_forward(duty_cycle):
    """
    Move forward with specified duty cycle (45-55% based on data table)
    """
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal forward duty cycle is between 45-55%")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)

def move_backward(duty_cycle):
    """
    Move backward with specified duty cycle (45-55% based on data table)
    """
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal backward duty cycle is between 45-55%")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)

def turn_left(duty_cycle):
    """
    Turn left with specified duty cycle (85-100% based on data table)
    """
    if not (85 <= duty_cycle <= 100):
        print("Warning: Optimal left turn duty cycle is between 85-100%")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)

def turn_right(duty_cycle):
    """
    Turn right with specified duty cycle (75-85% based on data table)
    """
    if not (75 <= duty_cycle <= 85):
        print("Warning: Optimal right turn duty cycle is between 75-85%")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)

def stop_motors():
    """
    Stop all motor movement
    """
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

try:
    # Example usage with optimal duty cycles from the data table
    print("Moving forward")
    move_forward(50)  # 50% duty cycle for forward movement
    time.sleep(5)

    print("Moving backward")
    move_backward(50)  # 50% duty cycle for backward movement
    time.sleep(5)

    print("Turning left")
    turn_left(90)  # 90% duty cycle for left turn
    time.sleep(5)

    print("Turning right")
    turn_right(80)  # 80% duty cycle for right turn
    time.sleep(5)

finally:
    # Stop the motors and clean up
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()

# user@raspberrypi:~/Documents/SiviV2/Sivi_Spring_Project $ /usr/bin/python /home/user/Documents/SiviV2/Sivi_Spring_Project/4WD.py
# Moving forward
# Moving backward
# Turning left
# Turning right
# Exception ignored in: <function PWM.__del__ at 0x7fb6042de0>
# Traceback (most recent call last):
#   File "/usr/lib/python3/dist-packages/RPi/GPIO/__init__.py", line 179, in __del__
#   File "/usr/lib/python3/dist-packages/RPi/GPIO/__init__.py", line 202, in stop
#   File "/usr/lib/python3/dist-packages/lgpio.py", line 1084, in tx_pwm
# TypeError: unsupported operand type(s) for &: 'NoneType' and 'int'
# Exception ignored in: <function PWM.__del__ at 0x7fb6042de0>
# Traceback (most recent call last):
#   File "/usr/lib/python3/dist-packages/RPi/GPIO/__init__.py", line 179, in __del__
#   File "/usr/lib/python3/dist-packages/RPi/GPIO/__init__.py", line 202, in stop
#   File "/usr/lib/python3/dist-packages/lgpio.py", line 1084, in tx_pwm
# TypeError: unsupported operand type(s) for &: 'NoneType' and 'int'