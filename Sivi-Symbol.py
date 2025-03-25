import numpy as np
import cv2
from picamera2 import Picamera2

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

def detect_color_edges(frame):
    """
    Highlight edges of color contrasts in the image with bright outlines
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Color ranges to detect (HSV ranges) with brighter BGR colors for outlines
    color_ranges = [
        ((0, 0, 200), (180, 30, 255), "White", (255, 255, 255)),   # White
        ((90, 50, 50), (130, 255, 255), "Blue", (255, 0, 0)),      # Blue
        ((0, 100, 100), (10, 255, 255), "Red", (0, 0, 255)),       # Red (lower range)
        ((160, 100, 100), (180, 255, 255), "Red", (0, 0, 255)),    # Red (upper range)
        ((40, 50, 50), (80, 255, 255), "Green", (0, 255, 0))       # Green
    ]
    
    # Create a black canvas for edges
    edge_display = np.zeros_like(frame)
    
    # Process each color range
    for lower, upper, color_name, color_bgr in color_ranges:
        # Create color mask
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Clean up the mask with morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find edges using Canny (with lower thresholds for more edges)
        edges = cv2.Canny(mask, 30, 100)
        
        # Dilate edges to make them thicker and more visible
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert edges to color and add to display
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        colored_edges[np.where((colored_edges == [255, 255, 255]).all(axis=2))] = color_bgr
        edge_display = cv2.add(edge_display, colored_edges)
    
    # Combine original with edges (more emphasis on edges)
    output = cv2.addWeighted(frame, 0.5, edge_display, 0.8, 0)
    
    return output

# Main program
picam2 = initialize_camera()
if picam2 is None:
    print("Exiting program. Camera could not be initialized.")
    exit()

try:
    while True:
        # Capture frame from the camera
        frame = picam2.capture_array()
        
        # Convert from RGB to BGR (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Flip the frame vertically if needed (depends on your camera orientation)
        # frame = cv2.flip(frame, -1)
        
        # Process the frame with our color edge detection
        output_frame = detect_color_edges(frame)
        
        # Display the frame
        cv2.imshow("Color Edge Detection", output_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()