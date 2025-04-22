import cv2
import numpy as np
from picamera2 import Picamera2
import os
import time
import uuid

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

# Define color priority order
COLOR_PRIORITY = ['red', 'blue', 'green', 'yellow', 'black']

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
    Bottom ROI (30%) for line detection, Top ROI (30%) for angle
    Returns contour, color, and angle
    """
    height, width = frame.shape[:2]
    if roi_type == 'bottom':
        roi_height = int(height * 0.3)  # Bottom 30%
        roi = frame[height - roi_height:height, :]
        y_offset = height - roi_height
    else:  # top
        roi_height = int(height * 0.3)  # Top 30%
        roi = frame[0:roi_height, :]
        y_offset = 0
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    MIN_AREA = 1000  # Minimum contour area to reduce noise
    
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
                    largest_contour[:, :, 1] += y_offset
                    rect = cv2.minAreaRect(largest_contour)
                    angle = rect[2]
                    if angle < -45:
                        angle += 90
                    return largest_contour, color_name, angle
    
    return None, None, 0

def main():
    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": (640, 480)})  # Increased resolution
    picam2.configure(config)
    picam2.start()
    
    # Set larger display window size
    WINDOW_NAME = "Color Line Display"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 960)  # Increased window size
    
    print("===== Color Line Display =====")
    
    color_priority = get_color_choices()
    if not color_priority:
        return
    
    print(f"\nColor priority: {' > '.join(color_priority)}")
    print("Press 'q' to quit or 'c' to change colors")
    
    try:
        while True:
            frame = picam2.capture_array()
            
            # Bottom ROI for line detection
            contour_bottom, color_name_bottom, line_angle_bottom = detect_priority_color(frame, color_priority, roi_type='bottom')
            
            # Top ROI for angle detection
            contour_top, color_name_top, line_angle_top = detect_priority_color(frame, color_priority, roi_type='top')
            
            # Initialize display variables
            current_color = "None"
            outline_coords = "N/A"
            error = 0
            
            # Draw ROIs
            height, width = frame.shape[:2]
            bottom_roi_height = int(height * 0.3)
            top_roi_height = int(height * 0.3)
            
            # Draw bottom ROI rectangle
            cv2.rectangle(frame, (0, height - bottom_roi_height), (width, height), (255, 255, 255), 1)
            cv2.putText(frame, "Bottom ROI", (10, height - bottom_roi_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw top ROI rectangle
            cv2.rectangle(frame, (0, 0), (width, top_roi_height), (255, 255, 255), 1)
            cv2.putText(frame, "Top ROI", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if contour_bottom is not None:
                current_color = color_name_bottom
                x, y, w, h = cv2.boundingRect(contour_bottom)
                outline_coords = f"({x}, {y}, {w}, {h})"
                
                color_map = {
                    'red': (0, 0, 255),
                    'blue': (255, 0, 0),
                    'green': (0, 255, 0),
                    'yellow': (0, 255, 255),
                    'black': (0, 0, 0)
                }
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[color_name_bottom], 2)
                
                # Calculate error for display
                line_center = x + w // 2
                frame_center = width // 2
                error = line_center - frame_center
            
            # Prepare metadata text
            priority_text = f"Priority: {'>'.join(color_priority)}"
            detection_text = f"Detected: {current_color}"
            coords_text = f"Coords: {outline_coords}"
            error_text = f"Error: {error:.2f}"
            angle_text = f"Top ROI Angle: {line_angle_top:.2f}Â°"
            
            # Display metadata
            y_offset = 60
            cv2.putText(frame, priority_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            cv2.putText(frame, detection_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            cv2.putText(frame, coords_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            cv2.putText(frame, error_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            cv2.putText(frame, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if "DISPLAY" in os.environ:
                cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                new_priority = get_color_choices()
                if new_priority:
                    color_priority = new_priority
                    print(f"New priority: {' > '.join(color_priority)}")
    
    except KeyboardInterrupt:
        print("Stopping display...")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()