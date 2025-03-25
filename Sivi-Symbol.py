import numpy as np
import cv2
import os

def detect_color_contours(frame):
    """
    Detect contours based on color contrast
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Color ranges to detect
    color_ranges = [
        ((0, 0, 200), (180, 30, 255), "White"),
        ((90, 50, 50), (130, 255, 255), "Blue"),
        ((0, 100, 100), (10, 255, 255), "Red"),
        ((160, 100, 100), (180, 255, 255), "Red"),
        ((40, 50, 50), (80, 255, 255), "Green")
    ]
    
    # Final frame to draw on
    output = frame.copy()
    
    # Process each color range
    for lower, upper, color_name in color_ranges:
        # Create color mask
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Apply edge detection on color mask
        edges = cv2.Canny(mask, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for c in contours:
            # Filter out small contours
            if cv2.contourArea(c) > 500:
                # Approximate the contour
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Determine shape
                shape = "Unknown"
                if len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    shape = "Quadrilateral"
                elif len(approx) == 5:
                    shape = "Pentagon"
                elif len(approx) >= 6:
                    shape = "Circle/Irregular"
                
                # Extract ROI
                roi = frame[y:y+h, x:x+w]
                
                # Detect symbol in ROI
                symbol = detect_symbol(roi)
                
                # Draw contour
                cv2.drawContours(output, [c], -1, (0, 255, 0), 2)
                
                # Prepare label
                label = f"{color_name} {shape}"
                if symbol:
                    label += f": {symbol}"
                
                # Put text
                cv2.putText(output, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return output

def detect_symbol(roi):
    """
    Detect symbols within a region of interest
    """
    # Convert to grayscale and resize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(resized, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Arrow detection
    def detect_directional_arrow():
        height, width = binary.shape
        
        # Check different arrow regions
        regions = {
            'Up': binary[:height//3, width//3:2*width//3],
            'Down': binary[2*height//3:, width//3:2*width//3],
            'Left': binary[height//3:2*height//3, :width//3],
            'Right': binary[height//3:2*height//3, 2*width//3:]
        }
        
        for direction, region in regions.items():
            if np.sum(region == 255) > region.size * 0.3:
                return f"{direction} Arrow"
        return None
    
    # Traffic light detection
    def detect_traffic_light():
        height, width = binary.shape
        vertical_strip = binary[:, width//2-10:width//2+10]
        
        color_sections = np.array_split(vertical_strip, 3)
        active_sections = sum(np.sum(section == 255) > section.size * 0.3 for section in color_sections)
        
        return "Traffic Light" if active_sections >= 2 else None
    
    # Stop sign detection
    def detect_stop_sign():
        height, width = binary.shape
        
        # Create circular mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (width//2, height//2), min(height, width)//2, 255, -1)
        
        # Apply mask
        masked = cv2.bitwise_and(binary, mask)
        
        # Check for circular shape with high white percentage
        circle_white_percent = np.sum(masked == 255) / np.sum(mask == 255) * 100
        
        return "Stop Sign" if circle_white_percent > 50 else None
    
    # Hand stop detection
    def detect_hand_stop():
        height, width = binary.shape
        palm_region = binary[height//3:2*height//3, width//3:2*width//3]
        
        # Check for palm-like region characteristics
        white_percent = np.sum(palm_region == 255) / palm_region.size * 100
        return "Hand Stop" if white_percent > 40 else None
    
    # Face recognition detection
    def detect_face_recognition():
        height, width = binary.shape
        face_region = binary[height//4:3*height//4, width//4:3*width//4]
        
        # Check for face-like region characteristics
        white_percent = np.sum(face_region == 255) / face_region.size * 100
        return "Face Recognition" if white_percent > 30 else None
    
    # Run detections
    detections = [
        detect_directional_arrow(),
        detect_traffic_light(),
        detect_stop_sign(),
        detect_hand_stop(),
        detect_face_recognition()
    ]
    
    # Return first non-None detection
    return next((det for det in detections if det is not None), None)

def main():
    # Load images from the folder
    symbol_folder = "Symbol-images"
    
    # Ensure folder exists
    if not os.path.exists(symbol_folder):
        print(f"Folder {symbol_folder} not found!")
        return
    
    # Process test images
    test_images = [f for f in os.listdir(symbol_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not test_images:
        print(f"No images found in {symbol_folder}")
        return
    
    for image_name in test_images:
        # Read the image
        image_path = os.path.join(symbol_folder, image_name)
        frame = cv2.imread(image_path)
        
        if frame is not None:
            # Process the image
            output = detect_color_contours(frame)
            
            # Show the processed image
            cv2.imshow(f"Processed: {image_name}", output)
    
    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()