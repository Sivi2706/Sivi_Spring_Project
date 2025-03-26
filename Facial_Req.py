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
        
        for subfolder in os.listdir(self.symbol_dir):
            subfolder_path = os.path.join(self.symbol_dir, subfolder)
            
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(subfolder_path, filename)
                        
                        template = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                        
                        if template is not None:
                            symbol_name = f"{subfolder}_{os.path.splitext(filename)[0]}"
                            self.symbol_templates[symbol_name] = template
                            print(f"Loaded template: {symbol_name}")
        
        if not self.symbol_templates:
            print("No templates found. Exiting.")
            exit()
        print(f"Loaded {len(self.symbol_templates)} templates.")

    def match_symbol(self, roi):
        best_match = None
        best_score = float('inf')
        
        # Preprocess ROI
        roi_preprocessed = cv2.resize(roi, (100, 100))
        roi_preprocessed = cv2.GaussianBlur(roi_preprocessed, (5, 5), 0)
        
        for name, template in self.symbol_templates.items():
            try:
                # Resize and preprocess template
                template_preprocessed = cv2.resize(template, (100, 100))
                template_preprocessed = cv2.GaussianBlur(template_preprocessed, (5, 5), 0)
                
                # Multiple matching methods for robustness
                methods = [
                    cv2.TM_SQDIFF_NORMED, 
                    cv2.TM_CCORR_NORMED, 
                    cv2.TM_CCOEFF_NORMED
                ]
                
                scores = []
                for method in methods:
                    result = cv2.matchTemplate(roi_preprocessed, template_preprocessed, method)
                    _, score, _, _ = cv2.minMaxLoc(result)
                    scores.append(score)
                
                # Average score across methods
                avg_score = np.mean(scores)
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_match = name
            
            except Exception as e:
                print(f"Error matching template {name}: {e}")
        
        # Lowered threshold for more flexible matching
        return best_match if best_score < 0.3 else None

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
            
            # Shape and symbol recognition
            symbol_name = symbol_recognizer.match_symbol(roi)
            
            # Visualize results
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display recognized symbol
            label = symbol_name if symbol_name else "Unknown Symbol"
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

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
            cv2.imshow("Camera Feed", output_frame)

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