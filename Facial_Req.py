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
        # Debugging: print ROI shape and type
        print(f"ROI Shape: {roi.shape}, Type: {roi.dtype}")
        
        best_match = None
        best_score = float('inf')
        match_scores = {}
        
        for name, template in self.symbol_templates.items():
            try:
                # Resize template to match ROI
                try:
                    resized_template = cv2.resize(template, (roi.shape[1], roi.shape[0]))
                except Exception as resize_err:
                    print(f"Resize error for {name}: {resize_err}")
                    continue
                
                # Compute template matching with multiple methods
                methods = [
                    (cv2.TM_SQDIFF_NORMED, "SQDIFF"),
                    (cv2.TM_CCORR_NORMED, "CCORR"),
                    (cv2.TM_CCOEFF_NORMED, "CCOEFF")
                ]
                
                for method, method_name in methods:
                    result = cv2.matchTemplate(roi, resized_template, method)
                    _, score, _, _ = cv2.minMaxLoc(result)
                    
                    # Normalize score for consistent comparison
                    if method in [cv2.TM_SQDIFF_NORMED]:
                        score = 1 - score  # Invert SQDIFF score
                    
                    match_scores[f"{name}_{method_name}"] = score
                    
                    # Update best match
                    if score < best_score:
                        best_score = score
                        best_match = name
            
            except Exception as e:
                print(f"Error matching template {name}: {e}")
        
        # Detailed logging of match scores
        print("Match Scores:")
        for match, score in sorted(match_scores.items(), key=lambda x: x[1]):
            print(f"{match}: {score}")
        
        # More flexible matching threshold
        # Increased threshold and added print for debugging
        if best_score < 0.5:  # Adjusted from 0.2 to 0.5
            print(f"Best Match: {best_match} with score {best_score}")
            return best_match
        else:
            print(f"No good match found. Best score: {best_score}")
            return None

def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

def detect_shapes_and_symbols(frame, symbol_recognizer):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Convert edges to color for visualization
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Threshold and find contours
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        # Filter contours by area
        if cv2.contourArea(c) > 500:
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(c)
            
            # Extract ROI
            roi = gray[y:y+h, x:x+w]
            
            # Symbol recognition
            symbol_name = symbol_recognizer.match_symbol(roi)
            
            # Draw contours on edge image
            cv2.drawContours(edges_colored, [c], -1, (0, 255, 0), 2)
            
            # Draw rectangle
            cv2.rectangle(edges_colored, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display recognized symbol
            label = symbol_name if symbol_name else "Unknown Symbol"
            cv2.putText(edges_colored, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
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

            # Detect shapes and symbols
            output_frame = detect_shapes_and_symbols(frame, symbol_recognizer)

            # Display the frame
            cv2.imshow("Edge Detection", output_frame)

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