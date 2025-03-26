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
            
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith('.png'):
                        full_path = os.path.join(subfolder_path, filename)
                        
                        # Load template in color and convert to HSV
                        template = cv2.imread(full_path, cv2.IMREAD_COLOR)
                        if template is not None:
                            template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
                            symbol_name = f"{subfolder}_{os.path.splitext(filename)[0]}"
                            self.symbol_templates[symbol_name] = template_hsv
                            print(f"Loaded HSV template: {symbol_name}")
        
        if not self.symbol_templates:
            print("No templates found. Exiting.")
            exit()
        print(f"Loaded {len(self.symbol_templates)} HSV templates.")
        input("Press Enter to continue...")

    def match_symbol(self, roi_hsv):
        best_match = None
        best_score = float('inf')
        
        for name, template_hsv in self.symbol_templates.items():
            try:
                # Resize template to match ROI dimensions
                resized_template = cv2.resize(template_hsv, (roi_hsv.shape[1], roi_hsv.shape[0]))
                
                # Perform template matching in HSV space
                result = cv2.matchTemplate(roi_hsv, resized_template, cv2.TM_SQDIFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                
                if score < best_score:
                    best_score = score
                    best_match = name
            except Exception as e:
                print(f"Error matching template {name}: {e}")
        
        return best_match if best_score < 0.25 else None  # Adjusted threshold for HSV

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
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Use Value channel for thresholding
    value_channel = hsv[:, :, 2]
    
    # Adaptive thresholding for better light compensation
    thresh = cv2.adaptiveThreshold(value_channel, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            
            # Extract ROI from HSV image
            roi_hsv = hsv[y:y+h, x:x+w]
            
            # Symbol recognition in HSV space
            symbol_name = symbol_recognizer.match_symbol(roi_hsv)
            
            # Visualization
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            label = symbol_name if symbol_name else "Unknown Symbol"
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def main():
    symbol_dir = '/home/raspberry/Documents/S1V1/Sivi_Spring_Project/Symbol-images'
    symbol_recognizer = SymbolRecognizer(symbol_dir)
    
    picam2 = initialize_camera()
    if picam2 is None:
        return

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.flip(frame, -1)  # Adjust based on camera orientation
            
            # Process frame with HSV-based detection
            output_frame = detect_shapes_and_symbols(frame, symbol_recognizer)
            
            cv2.imshow("HSV Symbol Detection", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()