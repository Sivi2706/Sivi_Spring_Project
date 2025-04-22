def detect_priority_color(frame, color_names, roi_type='bottom'):
    """
    Detect colors in priority order within specified ROI
    Bottom ROI (30%) for motor control, Top ROI (70%) for servo
    Returns contour, color, angle, and intersection status
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
    cv2.rectangle(display_frame, (0, 0), (width, top_roi_height), (255, 0, 0), 2)
    cv2.rectangle(display_frame, (0, height - int(height*0.3)), (width, height), (0, 0, 255), 2)
    
    # Draw center line
    center_x = width // 2
    cv2.line(display_frame, (center_x, 0), (center_x, height), (0, 0, 255), 1)
    
    # Process top ROI visualization
    if contour_top is not None:
        # Draw contour
        cv2.drawContours(display_frame, [contour_top], -1, (0, 255, 0), 2)
        
        # Draw fitted line for top ROI
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
        servo_angle = map_line_angle_to_servo_angle(line_angle_top)
        servo_text = f"Servo: {servo_angle:.1f}°"
        cv2.putText(display_frame, servo_text, (width-200, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Rest of the display logic remains the same...
    # ... [previous display code] ...
    
    return display_frame, error, movement, servo_angle