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
CLK_RIGHT = 24  # Right encoder CLK
DT_RIGHT = 25   # Right encoder DT
CLK_LEFT = 26   # Left encoder CLK
DT_LEFT = 19    # Left encoder DT

# Set up the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set up the GPIO pins as outputs for motor control
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Set up the GPIO pins for rotary encoders with pull-down resistors
GPIO.setup(CLK_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(DT_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(CLK_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(DT_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Set up PWM for duty cycle control
pwmA = GPIO.PWM(ENA, 100)  # Using 100Hz as base frequency for motor A
pwmB = GPIO.PWM(ENB, 100)  # Using 100Hz as base frequency for motor B
pwmA.start(0)  # Start with 0% duty cycle (stopped)
pwmB.start(0)  # Start with 0% duty cycle (stopped)

# Rotary encoder variables
counterRight = 0
counterLeft = 0
last_clk_right = GPIO.input(CLK_RIGHT)
last_clk_left = GPIO.input(CLK_LEFT)

# Robot physical specifications
wheelDiameter = 6.5  # Wheel diameter in centimeters
wheelCircumference = wheelDiameter * math.pi  # Wheel circumference in centimeters

# Number of pulses per revolution - set exactly to 20 as specified
PULSES_PER_REVOLUTION = 20

# Variables for display updating
last_display_time = 0
DISPLAY_INTERVAL = 0.05  # Update display every 50ms instead of on every change

def reset_counters():
    global counterRight, counterLeft
    counterRight = 0
    counterLeft = 0

def get_moving_distance():
    global counterRight, counterLeft
    
    # Calculate revolutions for each wheel
    revolutionsRight = counterRight / PULSES_PER_REVOLUTION
    revolutionsLeft = counterLeft / PULSES_PER_REVOLUTION
    
    # Calculate distance for each wheel
    distanceRight = revolutionsRight * wheelCircumference
    distanceLeft = revolutionsLeft * wheelCircumference
    
    # Calculate average distance
    averageDistance = (distanceRight + distanceLeft) / 2.0
    
    return revolutionsRight, revolutionsLeft, distanceRight, distanceLeft, averageDistance

def move_forward(duty_cycle, duration=1.0):
    if not (45 <= duty_cycle <= 55):
        print("Warning: Optimal forward duty cycle is between 45-55%")
    
    reset_counters()  # Reset counters before starting movement
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)
    
    # Live monitoring of distance during movement
    start_time = time.time()
    end_time = start_time + duration
    last_update_time = 0
    
    try:
        while time.time() < end_time:
            # Read encoders in a tight loop
            read_encoders()
            
            # Update display at specified intervals
            current_time = time.time()
            if current_time - last_update_time >= 0.05:  # Update display every 50ms
                revRight, revLeft, distRight, distLeft, avgDist = get_moving_distance()
                print(f"\rLive: Rev R: {revRight:.2f}, Rev L: {revLeft:.2f} | Dist R: {distRight:.2f}cm, Dist L: {distLeft:.2f}cm, Avg: {avgDist:.2f}cm", end="")
                last_update_time = current_time
            
            # Use a minimal delay - just enough to prevent CPU hogging
            time.sleep(0.001)  # 1ms delay
    except KeyboardInterrupt:
        pass
    finally:
        stop_motors()  # Stop the motors after moving
        revRight, revLeft, distRight, distLeft, avgDist = get_moving_distance()
        print(f"\nFinal: Rev R: {revRight:.2f}, Rev L: {revLeft:.2f} | Dist R: {distRight:.2f}cm, Dist L: {distLeft:.2f}cm, Avg: {avgDist:.2f}cm")

def read_encoders():
    """Function to read both encoders with maximum speed"""
    global counterRight, counterLeft, last_clk_right, last_clk_left
    
    # Read right encoder
    current_clk_right = GPIO.input(CLK_RIGHT)
    if current_clk_right != last_clk_right:
        current_dt_right = GPIO.input(DT_RIGHT)
        if current_dt_right != current_clk_right:
            counterRight += 1  # Clockwise
        else:
            counterRight -= 1  # Counter-clockwise
        last_clk_right = current_clk_right
    
    # Read left encoder
    current_clk_left = GPIO.input(CLK_LEFT)
    if current_clk_left != last_clk_left:
        current_dt_left = GPIO.input(DT_LEFT)
        if current_dt_left != current_clk_left:
            counterLeft += 1  # Clockwise
        else:
            counterLeft -= 1  # Counter-clockwise
        last_clk_left = current_clk_left

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
    # PWM demonstration
    pwm_demo()

    # Move forward for 1 second and print distances with live updates
    print("Moving forward with live distance monitoring...")
    move_forward(50, 1.0)  # Move forward with 50% duty cycle for 1 second
    
    # Main loop to keep reading encoder values with high-speed reading
    print("\nContinuing to monitor encoder counts. Press CTRL+C to exit.")
    last_display_time = time.time()
    
    while True:
        # Read encoders as fast as possible
        read_encoders()
        
        # Throttle the display updates to avoid overwhelming the terminal
        current_time = time.time()
        if current_time - last_display_time >= DISPLAY_INTERVAL:
            revRight, revLeft, distRight, distLeft, avgDist = get_moving_distance()
            print(f"\rRight count: {counterRight}, Left count: {counterLeft} | Rev R: {revRight:.2f}, Rev L: {revLeft:.2f} | Dist: {avgDist:.2f}cm", end="")
            last_display_time = current_time
        
        # Minimal sleep to prevent CPU hogging while maintaining high reading speed
        time.sleep(0.001)  # 1ms delay
    
except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Stop the motors and clean up GPIO
    print("\nStopping motors and cleaning up GPIO...")
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleanup complete.")