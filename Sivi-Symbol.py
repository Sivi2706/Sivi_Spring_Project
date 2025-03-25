import numpy as np
import cv2
import os

def detect_color_shapes(frame):
    """
    Detect shapes based on color boundaries and contours
    """
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # List to store unique colors and their masks
    color_ranges = [
        # White
        ((0, 0, 200), (180, 30, 255), (255, 255, 255), "White"),
        # Blue
        ((90, 50, 50), (130, 255, 255), (255, 0, 0), "Blue"),
        # Red (split into two ranges due to hue wrap-around)
        ((0, 100, 100), (10, 255, 255), (0, 0, 255), "Red"),
        ((160, 100, 100), (180, 255, 255), (0, 0, 255), "Red"),
        # Green
        ((40, 50, 50), (80, 255, 255), (0, 255, 0), "Green"),
    ]
    
    # Process each color range
    for lower, upper, draw_color, color_name in color_ranges:
        # Create mask for specific color range
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for c in contours:
            # Filter out small contours
            if cv2.contourArea(c) > 500:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(c)
                
                # Determine shape approximation
                epsilon = 0.04 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                
                # Determine shape name
                shape = "Unknown"
                if len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    shape = "Quadrilateral"
                elif len(approx) == 5:
                    shape = "Pentagon"
                elif len(approx) >= 6:
                    shape = "Circle/Irregular"
                
                # Draw contour
                cv2.drawContours(frame, [c], -1, draw_color, 2)
                
                # Label shape and color
                label = f"{color_name} {shape}"
                cv2.putText(frame, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def analyze_shape_interior(roi):
    """
    Analyze the interior of a shape for symbols
    """
    # Convert to grayscale and resize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(resized, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect symbols (reuse previous symbol detection logic)
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
    
    # Additional symbol detection methods (traffic light, stop sign, etc.)
    def detect_traffic_light():
        height, width = binary.shape
        vertical_strip = binary[:, width//2-10:width//2+10]
        
        color_sections = np.array_split(vertical_strip, 3)
        active_sections = sum(np.sum(section == 255) > section.size * 0.3 for section in color_sections)
        
        return "Traffic Light" if active_sections >= 2 else None
    
    # Run detections
    detections = [
        detect_directional_arrow(),
        detect_traffic_light()
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
            output = detect_color_shapes(frame)
            
            # Show the processed image
            cv2.imshow(f"Processed: {image_name}", output)
    
    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()