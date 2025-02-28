import RPi.GPIO as GPIO
import time

# Define GPIO pins
IN1, IN2 = 4, 17          # Right motor control
IN3, IN4 = 27, 22         # Left motor control
ENA, ENB = 18, 23         # PWM pins for motors

# GPIO Setup
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Motor pins setup
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    
    # Set up PWM
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    
    right_pwm.start(0)
    left_pwm.start(0)
    
    return right_pwm, left_pwm

# Movement functions
def move_forward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_right(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_left(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Main loop for manual control
def manual_control():
    right_pwm, left_pwm = setup_gpio()
    
    print("\n==== Robot Movement Testing Program ====")
    print("Commands:")
    print("  f <speed> - Move forward")
    print("  b <speed> - Move backward")
    print("  r <speed> - Turn right")
    print("  l <speed> - Turn left")
    print("  s - Stop motors")
    print("  t <time> - Set movement time (seconds)")
    print("  q - Quit")
    
    movement_time = 1.0  # Default movement time in seconds
    
    try:
        while True:
            command = input("\nEnter command (f/b/r/l/s/t/q): ").strip().lower()
            
            if command == 'q':
                break
                
            if command == 's':
                stop_motors(right_pwm, left_pwm)
                print("Motors stopped.")
                continue
                
            if command.startswith('t '):
                try:
                    movement_time = float(command.split()[1])
                    print(f"Movement time set to {movement_time:.1f} seconds")
                except (ValueError, IndexError):
                    print("Invalid time value. Please use format 't <seconds>'")
                continue
                
            # For movement commands
            try:
                cmd_parts = command.split()
                if len(cmd_parts) != 2:
                    print("Please use format '<command> <speed>'")
                    continue
                    
                cmd, speed = cmd_parts
                speed = float(speed)
                
                if not (0 <= speed <= 100):
                    print("Speed must be between 0 and 100")
                    continue
                
                # Execute movement based on command
                if cmd == 'f':
                    print(f"Moving forward at {speed}% for {movement_time:.1f} seconds")
                    move_forward(right_pwm, left_pwm, speed)
                elif cmd == 'b':
                    print(f"Moving backward at {speed}% for {movement_time:.1f} seconds")
                    move_backward(right_pwm, left_pwm, speed)
                elif cmd == 'r':
                    print(f"Turning right at {speed}% for {movement_time:.1f} seconds")
                    turn_right(right_pwm, left_pwm, speed)
                elif cmd == 'l':
                    print(f"Turning left at {speed}% for {movement_time:.1f} seconds")
                    turn_left(right_pwm, left_pwm, speed)
                else:
                    print(f"Unknown command: {cmd}")
                    continue
                
                # Run for specified time
                time.sleep(movement_time)
                stop_motors(right_pwm, left_pwm)
                print("Motors stopped.")
                
            except (ValueError, IndexError):
                print("Invalid command format. Use '<command> <speed>'")
    
    except KeyboardInterrupt:
        print("\nManual control stopped by user.")
    finally:
        stop_motors(right_pwm, left_pwm)
        GPIO.cleanup()
        print("GPIO cleaned up.")

# Run the manual control function
if __name__ == "__main__":
    manual_control()