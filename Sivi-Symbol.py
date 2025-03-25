import numpy as np
import cv2
import os

def detect_color_edges(frame):
    """
    Highlight edges of color contrasts in the image
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Color ranges to detect (HSV ranges)
    color_ranges = [
        ((0, 0, 200), (180, 30, 255), "White", (255, 255, 255)),   # White
        ((90, 50, 50), (130, 255, 255), "Blue", (255, 0, 0)),      # Blue
        ((0, 100, 100), (10, 255, 255), "Red", (0, 0, 255)),       # Red (lower range)
        ((160, 100, 100), (180, 255, 255), "Red", (0, 0, 255)),    # Red (upper range)
        ((40, 50, 50), (80, 255, 255), "Green", (0, 255, 0))       # Green
    ]
    
    # Create a blank canvas for edges
    edge_display = np.zeros_like(frame)
    
    # Process each color range
    for lower, upper, color_name, color_bgr in color_ranges:
        # Create color mask
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find edges using Canny on the mask
        edges = cv2.Canny(mask, 50, 150)
        
        # Convert edges to color and add to display
        colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        colored_edges[np.where((colored_edges == [255, 255, 255]).all(axis=2))] = color_bgr
        edge_display = cv2.add(edge_display, colored_edges)
    
    # Combine original with edges
    output = cv2.addWeighted(frame, 0.7, edge_display, 0.3, 0)
    
    return output

def main():
    # Load images from the folder
    image_folder = "Symbol-images"
    
    # Ensure folder exists
    if not os.path.exists(image_folder):
        print(f"Folder {image_folder} not found!")
        return
    
    # Process test images
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    for image_name in image_files:
        # Read the image
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        
        if frame is not None:
            # Process the image to detect color edges
            output = detect_color_edges(frame)
            
            # Show the processed image
            cv2.imshow(f"Color Edges: {image_name}", output)
    
    # Wait for a key press and then close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()