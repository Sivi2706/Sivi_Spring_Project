from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import os

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

# Define motor driver GPIO pins
motor_in1 = 22  # Left motor forward
motor_in2 = 27  # Left motor backward
motor_in3 = 17  # Right motor forward
motor_in4 = 4   # Right motor backward

# PWM pins for speed control
ENA = 13  # Left motor speed control
ENB = 12  # Right motor speed control

# Setup GPIO Mode
GPIO.setmode(GPIO.BCM)
GPIO.setup([motor_in1, motor_in2, motor_in3, motor_in4, ENA, ENB], GPIO.OUT)

# Setup PWM for speed control (frequency: 1kHz)
pwm1 = GPIO.PWM(ENA, 1000)  # Left motor PWM
pwm2 = GPIO.PWM(ENB, 1000)  # Right motor PWM
pwm1.start(0)
pwm2.start(0)

# Speed settings
base_speed = 55   # Base speed for forward movement
max_speed = 100   # Maximum speed for turns
reverse_speed = 50  # Speed for reverse movement

# PID parameters
Kp = 0.5  # Proportional gain
Ki = 0    # Integral gain
Kd = 0.5  # Derivative gain

# PID variables
integral = 0
previous_error = 0

# Color ranges in HSV (for better color detection)
color_ranges = {
    'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 50])},
    'red': {'lower': np.array([0, 120, 70]), 'upper': np.array([10, 255, 255])},
    'red2': {'lower': np.array([170, 120, 70]), 'upper': np.array([180, 255, 255])},  # Red wraps around 180
    'green': {'lower': np.array([40, 40, 40]), 'upper': np.array([80, 255, 255])},
    'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
    'blue': {'lower': np.array([90, 60, 60]), 'upper': np.array([120, 255, 255])}
}

current_color = None

def set_speed(left_speed, right_speed):
    left_speed = max(0, min(100, left_speed))
    right_speed = max(0, min(100, right_speed))
    pwm1.ChangeDutyCycle(left_speed)
    pwm2.ChangeDutyCycle(right_speed)

def move_forward():
    GPIO.output(motor_in1, GPIO.HIGH)
    GPIO.output(motor_in2, GPIO.LOW)
    GPIO.output(motor_in3, GPIO.HIGH)
    GPIO.output(motor_in4, GPIO.LOW)
    return "Moving Forward"

def move_reverse():
    GPIO.output(motor_in1, GPIO.LOW)
    GPIO.output(motor_in2, GPIO.HIGH)
    GPIO.output(motor_in3, GPIO.LOW)
    GPIO.output(motor_in4, GPIO.HIGH)
    set_speed(reverse_speed, reverse_speed)
    return "Moving Reverse"

def stop():
    set_speed(0, 0)
    GPIO.output(motor_in1, GPIO.LOW)
    GPIO.output(motor_in2, GPIO.LOW)
    GPIO.output(motor_in3, GPIO.LOW)
    GPIO.output(motor_in4, GPIO.LOW)
    return "Stopping"

def pid_control(error):
    global integral, previous_error
    proportional = error
    integral += error
    derivative = error - previous_error
    control_signal = Kp * proportional + Ki * integral + Kd * derivative
    previous_error = error
    return control_signal

def detect_color_dominant(hsv_img):
    # Find the dominant color in the image
    color_pixels = {}
    
    for color_name, color_range in color_ranges.items():
        if color_name == 'red':
            # Combine both red ranges
            mask1 = cv2.inRange(hsv_img, color_ranges['red']['lower'], color_ranges['red']['upper'])
            mask2 = cv2.inRange(hsv_img, color_ranges['red2']['lower'], color_ranges['red2']['upper'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_img, color_range['lower'], color_range['upper'])
        
        color_pixels[color_name] = cv2.countNonZero(mask)
    
    # Remove red2 from consideration (it's part of red)
    if 'red2' in color_pixels:
        del color_pixels['red2']
    
    # Get the color with most pixels detected
    dominant_color = max(color_pixels, key=color_pixels.get)
    
    # Only return if we have significant detection (at least 100 pixels)
    if color_pixels[dominant_color] > 100:
        return dominant_color
    return None

print("Press 'q' to exit the live feed.")

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect dominant color in the frame
        detected_color = detect_color_dominant(hsv)
        if detected_color:
            current_color = detected_color
        
        # Create mask for the current color
        if current_color:
            if current_color == 'red':
                # Handle red which is at both ends of HSV spectrum
                mask1 = cv2.inRange(hsv, color_ranges['red']['lower'], color_ranges['red']['upper'])
                mask2 = cv2.inRange(hsv, color_ranges['red2']['lower'], color_ranges['red2']['upper'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_ranges[current_color]['lower'], color_ranges[current_color]['upper'])
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the masked image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            action_text = ""
            direction_text = ""
            
            if contours:
                # Get the largest contour (assuming it is the line)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Draw bounding box on the original image
                color_display = (0, 0, 0)  # default black
                if current_color == 'red':
                    color_display = (0, 0, 255)  # red in BGR
                elif current_color == 'green':
                    color_display = (0, 255, 0)  # green
                elif current_color == 'blue':
                    color_display = (255, 0, 0)  # blue
                elif current_color == 'yellow':
                    color_display = (0, 255, 255)  # yellow
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_display, 2)
                cv2.putText(frame, f"Tracking: {current_color}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_display, 2)
                
                # Line center and frame center
                line_center = x + w // 2
                frame_center = frame.shape[1] // 2
                
                # Calculate error
                error = line_center - frame_center
                
                # Calculate control signal using PID
                control_signal = pid_control(error)
                
                # Adjust motor speeds based on control signal
                right_speed = base_speed - control_signal
                left_speed = base_speed + control_signal
                
                # Set motor speeds and move forward
                set_speed(left_speed, right_speed)
                action_text = move_forward()
                direction_text = f"Error: {error:.2f}, Control: {control_signal:.2f}, L: {left_speed:.1f}, R: {right_speed:.1f}"
            else:
                # If no line is detected, reverse until a line is found
                action_text = move_reverse()
                direction_text = "No line detected - Reversing"
            
            # Display metadata on the frame
            cv2.putText(frame, action_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, direction_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Tracking Color: {current_color}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the processed image
        if "DISPLAY" in os.environ:
            cv2.imshow("Line Following Robot - Color Detection", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping robot...")

except Exception as e:
    print("Error:", e)

# Cleanup
stop()
cv2.destroyAllWindows()
picam2.stop()
GPIO.cleanup()