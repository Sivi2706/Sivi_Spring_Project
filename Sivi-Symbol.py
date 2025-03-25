import numpy as np
import cv2
import os

def detect_circular_arrows(roi):
    """
    Detect arrows within a circular region
    """
    # Convert to grayscale and resize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    
    # Apply edge detection
    edges = cv2.Canny(resized, 50, 150)
    
    # Detect circular shape
    circles = cv2.HoughCircles(
        resized, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=40, 
        maxRadius=60
    )
    
    # Check for arrow-like features
    if circles is not None:
        # Check for directional arrow within the circle
        up_arrow = np.sum(edges[20:40, 40:60]) > 100
        down_arrow = np.sum(edges[60:80, 40:60]) > 100
        
        if up_arrow:
            return "Circular Up Arrow"
        elif down_arrow:
            return "Circular Down Arrow"
    
    return None

def detect_rectangular_arrows(roi):
    """
    Detect arrows within a rectangular region
    """
    # Convert to grayscale and resize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    
    # Apply edge detection
    edges = cv2.Canny(resized, 50, 150)
    
    # Check for arrow-like features
    up_arrow = check_directional_arrow(edges, direction='up')
    down_arrow = check_directional_arrow(edges, direction='down')
    left_arrow = check_directional_arrow(edges, direction='left')
    right_arrow = check_directional_arrow(edges, direction='right')
    
    if up_arrow:
        return "Rectangular Up Arrow"
    elif down_arrow:
        return "Rectangular Down Arrow"
    elif left_arrow:
        return "Rectangular Left Arrow"
    elif right_arrow:
        return "Rectangular Right Arrow"
    
    return None

def detect_special_symbols(roi):
    """
    Detect special symbols with unique characteristics
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    
    # Detect stop sign (red circular shape with white bar)
    stop_sign = detect_stop_sign(resized)
    if stop_sign:
        return stop_sign
    
    # Detect traffic light
    traffic_light = detect_traffic_light(resized)
    if traffic_light:
        return traffic_light
    
    # Detect hand/stop gesture
    hand_stop = detect_hand_stop(resized)
    if hand_stop:
        return hand_stop
    
    # Detect facial recognition icon
    face_recognition = detect_face_recognition(resized)
    if face_recognition:
        return face_recognition
    
    return None

def detect_stop_sign(gray):
    """
    Detect stop sign characteristics
    """
    # Look for circular shape with internal white bar
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check for circular shape
    edges = cv2.Canny(gray, 50, 150)
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=40, 
        maxRadius=60
    )
    
    if circles is not None:
        # Check for white horizontal bar in middle
        mid_row = thresh[50, :]
        white_pixels = np.sum(mid_row == 255)
        
        if white_pixels > 50:
            return "Stop Sign"
    
    return None

def detect_traffic_light(gray):
    """
    Detect traffic light characteristics
    """
    # Look for vertical strip with multiple color regions
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check for vertical strip
    vertical_strip = thresh[:, 45:55]
    
    # Look for multiple color-like regions
    color_regions = np.sum(vertical_strip == 255, axis=0)
    
    if len(color_regions) > 2:
        return "Traffic Light"
    
    return None

def detect_hand_stop(gray):
    """
    Detect hand stop gesture
    """
    # Look for hand-like shape
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check for palm-like region
    palm_region = thresh[50:80, 30:70]
    
    # Look for distinct hand shape characteristics
    if np.sum(palm_region == 255) > 1000:
        return "Hand Stop"
    
    return None

def detect_face_recognition(gray):
    """
    Detect face recognition icon characteristics
    """
    # Look for head-like shape with detection markers
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check for head-like region
    head_region = thresh[20:50, 30:70]
    
    # Look for detection marker characteristics
    if np.sum(head_region == 255) > 500:
        return "Face Recognition"
    
    return None

def check_directional_arrow(edges, direction='up'):
    """
    Check for arrow-like features in a specific direction
    """
    h, w = edges.shape
    
    if direction == 'up':
        check_region = edges[:h//3, w//4:3*w//4]
    elif direction == 'down':
        check_region = edges[2*h//3:, w//4:3*w//4]
    elif direction == 'left':
        check_region = edges[h//4:3*h//4, :w//3]
    elif direction == 'right':
        check_region = edges[h//4:3*h//4, 2*w//3:]
    
    return np.sum(check_region > 0) > 100

def detect_shapes_and_symbols(frame):
    """
    Detect shapes and symbols with advanced recognition
    """
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        if cv2.contourArea(c) > 1000:  # Ignore small contours
            # Approximate the contour
            epsilon = 0.04 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            
            x, y, w, h = cv2.boundingRect(approx)
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Determine basic shape
            shape = "Unknown"
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Quadrilateral"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            else:
                shape = "Circle/Irregular"
            
            # Advanced symbol recognition
            symbol = None
            
            # Check for circular arrows
            if shape == "Circle/Irregular":
                symbol = detect_circular_arrows(roi)
            
            # Check for rectangular arrows
            if shape == "Quadrilateral":
                symbol = detect_rectangular_arrows(roi)
            
            # Check for special symbols
            if not symbol:
                symbol = detect_special_symbols(roi)
            
            # Draw contour and label
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Combine shape and symbol information
            label = f"{shape}: {symbol}" if symbol else shape
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def main():
    # Load images from the folder
    symbol_folder = "Symbol-images"
    
    # Process test images
    test_images = [f for f in os.listdir(symbol_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for image_name in test_images:
        # Read the image
        image_path = os.path.join(symbol_folder, image_name)
        frame = cv2.imread(image_path)
        
        if frame is not None:
            # Process the image
            output = detect_shapes_and_symbols(frame)
            
            # Show the processed image
            cv2.imshow(f"Processed: {image_name}", output)
    
    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()