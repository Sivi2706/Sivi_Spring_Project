import numpy as np
import cv2
from picamera2 import Picamera2

# Line following parameters
MIN_CONTOUR_AREA = 800     # Minimum area for valid contours
FRAME_WIDTH = 640          # Camera frame width
FRAME_HEIGHT = 480         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 100        # Error threshold for pivoting

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

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

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
    Enhanced color detection with better angle calculation
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
    
    # Different minimum area thresholds for top and bottom
    MIN_AREA = 800 if roi_type == 'bottom' else 500
    
    # Edge detection for better contour finding
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    intersection = False
    
    for color_name in color_names:
        color_ranges = all_color_ranges.get(color_name, [])
        
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Combine color mask with edges
            combined = cv2.bitwise_and(mask, edges)
            if cv2.countNonZero(combined) < 50:  # Fallback to color mask if few edges
                combined = mask
            
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for intersections (multiple valid contours)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
            if len(valid_contours) >= 2:
                intersection = True
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Adjust contour coordinates to original frame
                adjusted_contour = largest_contour.copy()
                adjusted_contour[:, :, 1] += y_offset
                
                # Calculate line angle using fitLine for more accuracy
                if len(largest_contour) >= 5:  # Need at least 5 points for fitLine
                    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    line_angle = np.degrees(np.arctan2(vy, vx))[0]
                    
                    # Adjust angle to be between -90 and 90 degrees
                    if line_angle < -45:
                        line_angle += 90
                    elif line_angle > 45:
                        line_angle -= 90
                else:
                    # Fallback to minAreaRect if not enough points
                    rect = cv2.minAreaRect(largest_contour)
                    line_angle = rect[2]
                    if line_angle < -45:
                        line_angle += 90
                
                return adjusted_contour, color_name, line_angle, intersection
    
    return None, None, 0, intersection

def create_display_frame(frame, color_priority, contour_bottom, color_name_bottom, line_angle_bottom, 
                         contour_top, color_name_top, line_angle_top, error):
    """
    Enhanced display frame with detailed top ROI visualization
    """
    display_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw ROIs
    top_roi_height = int(height * 0.7)
    bottom_roi_height = int(height * 0.3)
    
    # Bottom ROI (red rectangle)
    cv2.rectangle(display_frame, (0, height - bottom_roi_height), (width, height), (0, 0, 255), 2)
    
    # Top ROI (blue rectangle)
    cv2.rectangle(display_frame, (0, 0), (width, top_roi_height), (255, 0, 0), 2)
    
    # Draw center line (red vertical line)
    center_x = width // 2
    cv2.line(display_frame, (center_x, 0), (center_x, height), (0, 0, 255), 1)
    
    # Draw horizontal yellow dividing line
    cv2.line(display_frame, (0, top_roi_height), (width, top_roi_height), (0, 255, 255), 2)
    
    movement = "No line detected"
    current_color = "None"
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
        
        # Draw fitted line for top ROI (magenta)
        rows, cols = frame.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(contour_top, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(display_frame, (cols-1, righty), (0, lefty), (255, 0, 255), 2)
        
        # Display angle information
        angle_text = f"Top Angle: {line_angle_top:.1f}°"
        cv2.putText(display_frame, angle_text, (width-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Calculate and display servo angle
        servo_angle = 90 - (line_angle_top * 2)  # Simplified mapping
        servo_angle = max(0, min(180, servo_angle))
        servo_text = f"Servo: {servo_angle:.1f}°"
        cv2.putText(display_frame, servo_text, (width-200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Get the center of the contour
        M = cv2.moments(contour_top)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Label top ROI contour
            cv2.putText(display_frame, f"{color_name_top}", (cx, cy - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[color_name_top], 2)
    
    # Display error in large red text at top-left corner
    cv2.putText(display_frame, f"Error: {error}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Display information at the bottom in yellow text
    detection_text = f"Detected: {current_color}"
    command_text = f"Command: {movement}"
    angle_text = f"Line Angle (Top): {line_angle_top:.2f}°"
    servo_text = f"Servo Angle: {servo_angle:.2f}°"
    
    # Put info text in the bottom section with yellow color
    y_start = height - 120
    line_height = 25
    
    cv2.putText(display_frame, detection_text, (20, y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, command_text, (20, y_start + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, angle_text, (20, y_start + 2*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, servo_text, (20, y_start + 3*line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return display_frame, error, movement, servo_angle

# Main function
def main():
    picam2 = setup_camera()
    
    print("===== Color Line Detection =====")
    
    # Get color priorities from user
    color_priority = get_color_choices()
    if not color_priority:
        return
    
    print(f"\nColor priority: {' > '.join(color_priority)}")
    print("Color detection started. Press 'q' to quit or 'c' to change colors")

    try:
        # Create a larger display window
        cv2.namedWindow("Color Line Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color Line Detection", 800, 600)
        
        while True:
            frame = picam2.capture_array()
            
            # Bottom ROI for motor control (30% of height)
            contour_bottom, color_name_bottom, line_angle_bottom, intersection_bottom = detect_priority_color(
                frame, color_priority, roi_type='bottom')
            
            # Top ROI for angle visualization (70% of height)
            contour_top, color_name_top, line_angle_top, intersection_top = detect_priority_color(
                frame, color_priority, roi_type='top')
            
            # Create display frame with all visual elements
            display_frame, error, movement, servo_angle = create_display_frame(
                frame, color_priority, contour_bottom, color_name_bottom, line_angle_bottom,
                contour_top, color_name_top, line_angle_top, error)
            
            # Show the frame
            cv2.imshow("Color Line Detection", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                new_priority = get_color_choices()
                if new_priority:
                    color_priority = new_priority
                    print(f"New priority: {' > '.join(color_priority)}")
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("Resources released")

if __name__ == "__main__":
    main()