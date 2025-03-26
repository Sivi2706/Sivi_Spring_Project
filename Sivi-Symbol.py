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
        for filename in os.listdir(self.symbol_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(self.symbol_dir, filename)
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    symbol_name = os.path.splitext(filename)[0]
                    print(f"Calibrating: {symbol_name}")
                    
                    # Display each template
                    cv2.imshow(symbol_name, template)
                    cv2.waitKey(500)  # Brief display
                    
                    # Store template
                    self.symbol_templates[symbol_name] = template
        
        cv2.destroyAllWindows()
        print("Calibration complete. Press 'OK' to continue.")
        user_input = input("Enter 'OK' to start live feed: ")
        if user_input.upper() != 'OK':
            print("Calibration aborted.")
            exit()

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
    symbol_dir = 'Symbol-images'  # Update with actual path
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