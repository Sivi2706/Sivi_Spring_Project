import numpy as np
import cv2
import os

def analyze_shape_interior(roi):
    """
    Analyze the interior of a quadrilateral for symbols
    """
    # Convert to grayscale and resize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    
    # Apply thresholding
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    # Detect interior features
    
    # 1. Arrow Detection
    def detect_arrow():
        # Check for arrow-like features in different orientations
        directions = {
            'up': binary[:30, 35:65],
            'down': binary[70:, 35:65],
            'left': binary[35:65, :30],
            'right': binary[35:65, 70:]
        }
        
        for direction, region in directions.items():
            if np.sum(region == 255) > 50:
                return f"{direction.capitalize()} Arrow"
        return None
    
    # 2. Traffic Light Detection
    def detect_traffic_light():
        # Look for vertical strip with multiple color-like regions
        vertical_strip = binary[:, 45:55]
        color_regions = np.sum(vertical_strip == 255, axis=0)
        
        if len(color_regions) > 2:
            return "Traffic Light"
        return None
    
    # 3. Stop Sign Detection
    def detect_stop_sign():
        # Check for circular shape with white bar
        circle_mask = np.zeros_like(binary)
        cv2.circle(circle_mask, (50, 50), 40, 255, -1)
        
        # Check overlap of circle and binary image
        masked = cv2.bitwise_and(binary, circle_mask)
        
        # Check for white bar in middle
        mid_row = masked[50, :]
        white_pixels = np.sum(mid_row == 255)
        
        if white_pixels > 30:
            return "Stop Sign"
        return None
    
    # 4. Hand Stop Detection
    def detect_hand_stop():
        # Look for palm-like region
        palm_region = binary[50:80, 30:70]
        
        # Check for distinct hand shape
        if np.sum(palm_region == 255) > 800:
            return "Hand Stop"
        return None
    
    # 5. Face Recognition Detection
    def detect_face_recognition():
        # Look for head-like shape with detection markers
        head_region = binary[20:50, 30:70]
        
        if np.sum(head_region == 255) > 400:
            return "Face Recognition"
        return None
    
    # Run detection methods
    detections = [
        detect_arrow(),
        detect_traffic_light(),
        detect_stop_sign(),
        detect_hand_stop(),
        detect_face_recognition()
    ]
    
    # Return first non-None detection
    for detection in detections:
        if detection:
            return detection
    
    return None

def detect_shapes_and_symbols(frame):
    """
    Detect shapes and analyze their interior
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
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Special focus on quadrilaterals
            symbol = None
            if shape == "Quadrilateral":
                symbol = analyze_shape_interior(roi)
            
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