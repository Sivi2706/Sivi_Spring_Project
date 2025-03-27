import numpy as np
import cv2
import os
from picamera2 import Picamera2

class SymbolRecognizer:
    def __init__(self, symbol_dir):
        self.symbol_dir = symbol_dir
        self.symbol_templates = {}
        self.calibrate()

    def calibrate(self):
        print("Starting Calibration Stage...")
        
        # Iterate through subfolders
        for subfolder in os.listdir(self.symbol_dir):
            subfolder_path = os.path.join(self.symbol_dir, subfolder)
            
            # Ensure it's a directory
            if os.path.isdir(subfolder_path):
                # Iterate through PNG files in subfolder
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith('.png'):
                        full_path = os.path.join(subfolder_path, filename)
                        
                        # Load template
                        template = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                        
                        if template is not None:
                            # Create symbol name from subfolder and filename
                            symbol_name = f"{subfolder}_{os.path.splitext(filename)[0]}"
                            self.symbol_templates[symbol_name] = template
                            print(f"Loaded template: {symbol_name}")
        
        if not self.symbol_templates:
            print("No templates found. Exiting.")
            exit()
        print(f"Loaded {len(self.symbol_templates)} templates.")
        input("Press Enter to continue...")

    def match_symbol(self, roi):
        best_match = None
        best_score = float('inf')
        
        for name, template in self.symbol_templates.items():
            try:
                # Resize template to match ROI
                resized_template = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                
                # Compute template matching
                result = cv2.matchTemplate(roi, resized_template, cv2.TM_SQDIFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                
                if score < best_score:
                    best_score = score
                    best_match = name
            except Exception as e:
                print(f"Error matching template {name}: {e}")
        
        return best_match if best_score < 0.2 else None

def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

def process_edge_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    # Lower threshold is 50, upper threshold is 150
    edges = cv2.Canny(blurred, 50, 150)
    
    # Create a color version of edges for better visualization
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_colored

def main():
    # Initialize symbol recognizer
    symbol_dir = '/home/raspberry/Documents/S1V1/Sivi_Spring_Project/Symbol-images'
    symbol_recognizer = SymbolRecognizer(symbol_dir)
    
    # Initialize camera
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return

    try:
        # Main processing loop
        while True:
            # Capture frame from the camera
            frame = picam2.capture_array()

            # Flip the frame vertically (optional, depending on camera orientation)
            frame = cv2.flip(frame, -1)

            # Perform edge detection
            edge_frame = process_edge_detection(frame)

            # Display the edge detection frame
            cv2.imshow("Edge Detection", edge_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        picam2.stop()

# Run the main function
if __name__ == "__main__":
    main()