import RPi.GPIO as GPIO
import time

# Define GPIO pins (from line-following code)
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
SERVO_PIN = 18            # Servo motor pin

PWM_FREQ = 1000           # Motor PWM frequency
SERVO_FREQ = 50           # Servo PWM frequency
RUN_DURATION = 0.5        # Seconds
PWM_DUTY_CYCLE = 100      # Duty cycle (max speed)
SERVO_NEUTRAL = 7.5       # Neutral position (90 degrees, 7.5% duty cycle)

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup motor pins
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, SERVO_PIN], GPIO.OUT)

    # Setup motor PWM
    left_pwm = GPIO.PWM(ENA, PWM_FREQ)  # Left motor
    right_pwm = GPIO.PWM(ENB, PWM_FREQ) # Right motor
    left_pwm.start(0)
    right_pwm.start(0)

    # Setup servo PWM
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    servo_pwm.start(SERVO_NEUTRAL)  # Start at neutral

    return left_pwm, right_pwm, servo_pwm

def move_forward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Moving Forward")

def move_backward(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Moving Backward")

def turn_left(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Turning Left")

def turn_right(left_pwm, right_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    print("Turning Right")

def stop_motors(left_pwm, right_pwm, servo_pwm):
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    servo_pwm.ChangeDutyCycle(SERVO_NEUTRAL)
    print("Stopped")

def set_servo_angle(servo_pwm, angle):
    """
    Set servo to specified angle (0 to 180 degrees)
    Maps angle to duty cycle (2.5% to 12.5% for 0 to 180 degrees)
    """
    try:
        angle = float(angle)
        if 0 <= angle <= 180:
            # Linear mapping: 0° -> 2.5% (0.5ms), 180° -> 12.5% (2.5ms)
            duty = 2.5 + (angle / 180.0) * 10.0
            servo_pwm.ChangeDutyCycle(duty)
            print(f"Servo set to {angle} degrees")
            return True
        else:
            print("Angle must be between 0 and 180 degrees")
            return False
    except ValueError:
        print("Invalid angle. Use a number between 0 and 180")
        return False

def main():
    left_pwm, right_pwm, servo_pwm = setup_gpio()
    try:
        print("Commands: f=forward, b=backward, l=left, r=right, s=stop, s<angle>=servo (e.g., s30), q=quit")
        while True:
            cmd = input("Enter command: ").strip().lower()
            if cmd == "f":
                move_forward(left_pwm, right_pwm)
            elif cmd == "b":
                move_backward(left_pwm, right_pwm)
            elif cmd == "l":
                turn_left(left_pwm, right_pwm)
            elif cmd == "r":
                turn_right(left_pwm, right_pwm)
            elif cmd == "s":
                stop_motors(left_pwm, right_pwm, servo_pwm)
            elif cmd.startswith("s") and len(cmd) > 1:
                angle = cmd[1:]  # Extract angle after 's'
                set_servo_angle(servo_pwm, angle)
                continue  # Don't stop motors or sleep for servo command
            elif cmd == "q":
                break
            else:
                print("Invalid command. Try again.")
                continue
            time.sleep(RUN_DURATION)
            stop_motors(left_pwm, right_pwm, servo_pwm)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stop_motors(left_pwm, right_pwm, servo_pwm)
        left_pwm.stop()
        right_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("GPIO cleaned up")

if __name__ == "__main__":
    main()