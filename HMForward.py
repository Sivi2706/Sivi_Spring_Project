import RPi.GPIO as GPIO
import time
import math

# Define GPIO pins
IN1, IN2 = 4, 17          # Right motor control
IN3, IN4 = 27, 22         # Left motor control
ENA, ENB = 18, 23         # PWM pins for motors
encoderPinRight = 24      # Right encoder
encoderPinLeft = 25       # Left encoder

# Constants
WHEEL_DIAMETER = 6.5  # cm
PULSES_PER_REVOLUTION = 20  # Adjust based on your encoder
wheelCircumference = WHEEL_DIAMETER * math.pi  # cm

# Global counters
pulsesRight = 0
pulsesLeft = 0

# Setup
def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Setup motor pins
    for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
        GPIO.setup(pin, GPIO.OUT)
    
    # Setup encoder pins with pull-up resistors
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Setup encoder interrupts
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=counter_update_right)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=counter_update_left)
    
    # Setup PWM
    pwmA = GPIO.PWM(ENA, 100)  # Right motor, 100Hz default
    pwmB = GPIO.PWM(ENB, 100)  # Left motor, 100Hz default
    pwmA.start(0)
    pwmB.start(0)
    
    return pwmA, pwmB

# Encoder callbacks
def counter_update_right(channel):
    global pulsesRight
    pulsesRight += 1

def counter_update_left(channel):
    global pulsesLeft
    pulsesLeft += 1

# Reset counters
def reset_counters():
    global pulsesRight, pulsesLeft
    pulsesRight = pulsesLeft = 0

# Calculate distance
def calculate_distance():
    distanceRight = (pulsesRight / PULSES_PER_REVOLUTION) * wheelCircumference
    distanceLeft = (pulsesLeft / PULSES_PER_REVOLUTION) * wheelCircumference
    avgDistance = (distanceRight + distanceLeft) / 2.0
    return distanceRight, distanceLeft, avgDistance

# Motor control
def set_motors(right_duty, left_duty):
    # Forward direction
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    
    # Set speeds
    pwmA.ChangeDutyCycle(right_duty)
    pwmB.ChangeDutyCycle(left_duty)

def stop_motors():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

# Test distance for different PWM values
def test_pwm_distance(pwm_values, duration=1.0, right_bias=1.0):
    results = []
    
    for pwm in pwm_values:
        reset_counters()
        right_pwm = pwm * right_bias
        
        print(f"Testing PWM: {pwm}% (Right: {right_pwm:.1f}%, Left: {pwm}%)")
        set_motors(right_pwm, pwm)
        
        # Precise timing
        start_time = time.time()
        time.sleep(duration)
        actual_duration = time.time() - start_time
        
        stop_motors()
        
        # Calculate results
        right_dist, left_dist, avg_dist = calculate_distance()
        right_speed = right_dist / actual_duration
        left_speed = left_dist / actual_duration
        avg_speed = avg_dist / actual_duration
        
        results.append({
            'pwm': pwm,
            'right_pwm': right_pwm,
            'right_pulses': pulsesRight,
            'left_pulses': pulsesLeft,
            'right_dist': right_dist,
            'left_dist': left_dist,
            'avg_dist': avg_dist,
            'right_speed': right_speed,
            'left_speed': left_speed,
            'avg_speed': avg_speed
        })
        
        print(f"  Pulses: R={pulsesRight}, L={pulsesLeft}")
        print(f"  Distance: R={right_dist:.2f}cm, L={left_dist:.2f}cm, Avg={avg_dist:.2f}cm")
        print(f"  Speed: {avg_speed:.2f}cm/s\n")
        
        time.sleep(1)  # Pause between tests
    
    return results

# Test frequency effects
def test_frequency_effects(frequencies, pwm_value=50, duration=1.0):
    results = []
    original_freq = pwmA.frequency
    
    for freq in frequencies:
        reset_counters()
        
        # Change PWM frequency
        pwmA.ChangeFrequency(freq)
        pwmB.ChangeFrequency(freq)
        
        print(f"Testing Frequency: {freq}Hz at PWM={pwm_value}%")
        set_motors(pwm_value, pwm_value)
        
        # Precise timing
        start_time = time.time()
        time.sleep(duration)
        actual_duration = time.time() - start_time
        
        stop_motors()
        
        # Calculate results
        right_dist, left_dist, avg_dist = calculate_distance()
        avg_speed = avg_dist / actual_duration
        
        results.append({
            'frequency': freq,
            'pwm': pwm_value,
            'right_dist': right_dist,
            'left_dist': left_dist,
            'avg_dist': avg_dist,
            'speed': avg_speed
        })
        
        print(f"  Distance: R={right_dist:.2f}cm, L={left_dist:.2f}cm, Avg={avg_dist:.2f}cm")
        print(f"  Speed: {avg_speed:.2f}cm/s\n")
        
        time.sleep(1)  # Pause between tests
    
    # Reset to original frequency
    pwmA.ChangeFrequency(original_freq)
    pwmB.ChangeFrequency(original_freq)
    
    return results

# Find optimal bias to make robot drive straight
def calibrate_right_bias(base_pwm=50, duration=1.0):
    biases = [1.0, 0.95, 0.9, 0.85, 0.8]
    best_bias = 1.0
    min_diff = float('inf')
    
    print("Calibrating right motor bias...")
    
    for bias in biases:
        reset_counters()
        
        print(f"Testing bias: {bias}")
        set_motors(base_pwm * bias, base_pwm)
        
        time.sleep(duration)
        stop_motors()
        
        right_dist, left_dist, _ = calculate_distance()
        diff = abs(right_dist - left_dist)
        
        print(f"  R={right_dist:.2f}cm, L={left_dist:.2f}cm, Diff={diff:.2f}cm")
        
        if diff < min_diff:
            min_diff = diff
            best_bias = bias
        
        time.sleep(1)
    
    print(f"Best bias: {best_bias} (Difference: {min_diff:.2f}cm)")
    return best_bias

# Main test sequence
try:
    # Initialize
    pwmA, pwmB = setup()
    
    # Find optimal bias to make the robot drive straight
    right_bias = calibrate_right_bias()
    
    # Test 1: Measure distance at different PWM values
    print("\n=== Testing Distance at Different PWM Values ===")
    pwm_results = test_pwm_distance([30, 40, 50, 60, 70], duration=1.0, right_bias=right_bias)
    
    # Test 2: Compare different PWM frequencies
    print("\n=== Testing Different PWM Frequencies ===")
    freq_results = test_frequency_effects([50, 100, 200, 500, 1000], pwm_value=50)
    
    # Summary
    print("\n=== DISTANCE-PWM SUMMARY ===")
    print("PWM Value | Avg Distance (cm) | Speed (cm/s)")
    print("-" * 45)
    for r in pwm_results:
        print(f"{r['pwm']:8} | {r['avg_dist']:16.2f} | {r['avg_speed']:10.2f}")
    
    print("\n=== FREQUENCY SUMMARY ===")
    print("Frequency | Avg Distance (cm) | Speed (cm/s)")
    print("-" * 45)
    for r in freq_results:
        print(f"{r['frequency']:9} | {r['avg_dist']:16.2f} | {r['speed']:10.2f}")

except KeyboardInterrupt:
    print("\nProgram interrupted")
except Exception as e:
    print(f"Error: {e}")
finally:
    stop_motors()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("Cleanup complete")