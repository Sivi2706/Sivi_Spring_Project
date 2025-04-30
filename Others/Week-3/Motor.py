import RPi.GPIO as GPIO
import time

# Define GPIO pins
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors

PWM_FREQ = 1000           # PWM frequency
RUN_DURATION = 0.5        # Seconds
PWM_DUTY_CYCLE = 100      # Duty cycle (max speed)

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)

    right_pwm = GPIO.PWM(ENA, PWM_FREQ)
    left_pwm = GPIO.PWM(ENB, PWM_FREQ)
    right_pwm.start(0)
    left_pwm.start(0)

    return right_pwm, left_pwm

def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)

def move_backward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)

def pivot_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)

def pivot_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)
    left_pwm.ChangeDutyCycle(PWM_DUTY_CYCLE)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def main():
    right_pwm, left_pwm = setup_gpio()
    try:
        while True:
            cmd = input("Enter movement command (f=forward, b=backward, l=left, r=right, s=stop, q=quit): ").strip().lower()
            if cmd == "f":
                move_forward(right_pwm, left_pwm)
            elif cmd == "b":
                move_backward(right_pwm, left_pwm)
            elif cmd == "l":
                pivot_left(right_pwm, left_pwm)
            elif cmd == "r":
                pivot_right(right_pwm, left_pwm)
            elif cmd == "s":
                stop_motors(right_pwm, left_pwm)
            elif cmd == "q":
                break
            else:
                print("Invalid command. Try again.")
                continue
            time.sleep(RUN_DURATION)
            stop_motors(right_pwm, left_pwm)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        GPIO.cleanup()
        print("GPIO cleaned up")

if __name__ == "__main__":
    main()
