import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Define all available color ranges (HSV format)
all_color_ranges = {
    'red': [
        ([0, 167, 154], [10, 247, 234]),
        ([114, 167, 154], [134, 247, 234])
    ],
    'blue': [
        ([6, 167, 60], [26, 255, 95])
    ],
    'green': [
        ([31, 180, 110], [51, 255, 190])
    ],
    'yellow': [
        ([84, 155, 189], [104, 235, 255])
    ],
    'black': [
        ([0, 0, 0], [179, 78, 50])
    ]
}

def get_color_choices():
    print("\nAvailable line colors to follow (priority order):")
    print("r = red (highest priority)")
    print("b = blue")
    print("g = green")
    print("y = yellow")
    print("k = black (lowest priority)")
    print("q = quit program")
    print("\nEnter colors in priority order (e.g., 'rb' for red then blue)")
    
    color_map = {
        'r': 'red',
        'b': 'blue',
        'g': 'green',
        'y': 'yellow',
        'k': 'black'
    }
    
    while True:
        choices = input("\nEnter color priorities (e.g., 'rbk'): ").lower()
        if choices == 'q':
            return None
        
        seen = set()
        unique_choices = []
        for c in choices:
            if c in color_map and c not in seen:
                seen.add(c)
                unique_choices.append(c)
        
        if unique_choices:
            selected_colors = [color_map[c] for c in unique_choices]
            print(f"Priority order: {' > '.join(selected_colors)}")
            return selected_colors
        else:
            print("Invalid choice. Please try again.")

def detect_priority_color(frame, color_names, roi_type='bottom'):
    """
    Detect colors in priority order within specified ROI
    Bottom ROI (30%) for motor control, Top ROI (70%) for servo
    Returns contour, color, and angle
    """
    height, width = frame.shape[:2]
    if roi_type == 'bottom':
        roi_height = int(height * 0.3)  # Bottom 30%
        roi = frame[height - roi_height:height, :]
        y_offset = height - roi_height
    else:  # top
        roi_height = int(height * 0.7)  # Top 70%
        roi = frame[0:roi_height, :]
        y_offset = 0
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    MIN_AREA = 1000  # Increased to reduce noise
    
    for color_name in color_names:
        color_ranges = all_color_ranges.get(color_name, [])
        
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_AREA:
                    # Adjust contour coordinates to original frame
                    adjusted_contour = largest_contour.copy()
                    adjusted_contour[:, :, 1] += y_offset
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]
                    if angle < -45:
                        angle += 90
                    return adjusted_contour, color_name, angle
    
    return None, None, 0

def map_line_angle_to_servo_angle(line_angle):
    """
    Map line angle (-45° to +45°) to servo angle (180° to 0°)
    -45° (left) -> 180°, 0° (straight) -> 90°, +45° (right) -> 0°
    """
    # Normalize line angle to [-45, 45]
    normalized_angle = max(-45, min(45, line_angle))
    # Linear mapping: -45° -> 180°, 0° -> 90°, +45° -> 0°
    servo_angle = 90 - (normalized_angle * 4)  # Scale: 45° -> 180°
    return max(0, min(180, servo_angle))

def main():
    # Initialize camera with larger resolution
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": (640, 480)})  # Larger resolution
    picam2.configure(config)
    picam2.start()
    
    print("===== Color Line Detection Visualization =====")
    
    color_priority = get_color_choices()
    if not color_priority:
        return
    
    print(f"\nColor priority: {' > '.join(color_priority)}")
    print("Press 'q' to quit or 'c' to change colors")
    
    try:
        # Create a larger display window
        cv2.namedWindow("Color Line Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color Line Detection", 800, 600)  # Set initial window size
        
        while True:
            frame = picam2.capture_array()
            display_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Define ROIs
            top_roi_height = int(height * 0.7)
            bottom_roi_height = int(height * 0.3)
            
            # Bottom ROI for detection (30% of height)
            contour_bottom, color_name_bottom, line_angle_bottom = detect_priority_color(frame, color_priority, roi_type='bottom')
            
            # Top ROI for angle visualization (70% of height)
            contour_top, color_name_top, line_angle_top = detect_priority_color(frame, color_priority, roi_type='top')
            
            # Draw ROIs on the frame
            # Bottom ROI (red rectangle)
            cv2.rectangle(display_frame, (0, height - bottom_roi_height), (width, height), (0, 0, 255), 2)
            
            # Top ROI (blue rectangle)
            cv2.rectangle(display_frame, (0, 0), (width, top_roi_height), (255, 0, 0), 2)
            
            # Draw center line (red vertical line)
            center_x = width // 2
            cv2.line(display_frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
            
            movement = "No line detected"
            outline_coords = "N/A"
            current_color = "None"
            error = 0
            servo_angle = 90
            
            # Color to BGR mapping
            color_map = {
                'red': (0, 0, 255),
                'blue': (255, 0, 0),
                'green': (0, 255, 0),
                'yellow': (0, 255, 255),
                'black': (0, 0, 0)
            }
            
            # Process bottom ROI detection
            if contour_bottom is not None:
                # Draw contour outline in green
                cv2.drawContours(display_frame, [contour_bottom], -1, (0, 255, 0), 2)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour_bottom)
                outline_coords = f"({x}, {y}, {w}, {h})"
                current_color = color_name_bottom
                
                # Calculate error (distance from center)
                line_center = x + w // 2
                frame_center = width // 2
                error = line_center - frame_center
                
                # Draw a blue dot at contour center point with line to center
                cv2.circle(display_frame, (line_center, y + h//2), 5, (255, 0, 0), -1)
                cv2.line(display_frame, (frame_center, y + h//2), (line_center, y + h//2), (255, 0, 0), 2)
                
                # Determine movement direction
                CENTER_THRESHOLD = 20
                if error < -CENTER_THRESHOLD:
                    movement = "Turn Left"
                elif error > CENTER_THRESHOLD:
                    movement = "Turn Right"
                else:
                    movement = "Move Forward"
                
                # Label bottom ROI contour
                cv2.putText(display_frame, f"{color_name_bottom}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[color_name_bottom], 2)
            
            # Process top ROI detection
            if contour_top is not None:
                # Draw contour outline in green
                cv2.drawContours(display_frame, [contour_top], -1, (0, 255, 0), 2)
                
                # Get the center of the contour
                M = cv2.moments(contour_top)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Label top ROI contour
                    cv2.putText(display_frame, f"{color_name_top}", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[color_name_top], 2)
                
                # Calculate servo angle based on top ROI
                servo_angle = map_line_angle_to_servo_angle(line_angle_top)
            
            # Display error in red text at top-left corner
            cv2.putText(display_frame, f"Error: {error}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Display all information
            priority_text = f"Priority: {'>'.join(color_priority)}"
            detection_text = f"Detected: {current_color}"
            command_text = f"Command: {movement}"
            error_text = f"Error: {error}"
            angle_text = f"Line Angle (Top): {line_angle_top:.2f}°"
            servo_text = f"Servo Angle: {servo_angle:.2f}°"
            
            # Put additional info at the bottom of the frame
            font_scale = 0.6
            thickness = 2
            y_start = height - 140
            line_height = 25
            
            cv2.putText(display_frame, priority_text, (10, y_start), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            cv2.putText(display_frame, detection_text, (10, y_start + line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            cv2.putText(display_frame, command_text, (10, y_start + 2*line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            cv2.putText(display_frame, angle_text, (10, y_start + 3*line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            cv2.putText(display_frame, servo_text, (10, y_start + 4*line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            
            # Show the frame
            cv2.imshow("Color Line Detection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                new_priority = get_color_choices()
                if new_priority:
                    color_priority = new_priority
                    print(f"New priority: {' > '.join(color_priority)}")
    
    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()