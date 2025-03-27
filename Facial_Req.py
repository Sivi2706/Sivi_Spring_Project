import numpy as np
import cv2
import os
from picamera2 import Picamera2

class SymbolRecognizer:
    def __init__(self, symbol_dir):
        self.symbol_dir = symbol_dir
        # Dictionary: symbol_name -> (color_template (RGB), gray_template, contour)
        self.symbol_templates = {}
        self.calibrate()

    def calibrate(self):
        print("Starting Calibration Stage...")

        for subfolder in os.listdir(self.symbol_dir):
            subfolder_path = os.path.join(self.symbol_dir, subfolder)

            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith('.png'):
                        full_path = os.path.join(subfolder_path, filename)
                        
                        # Load template in BGR (OpenCV default)
                        template_bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
                        
                        if template_bgr is not None:
                            # Convert to RGB for internal storage
                            template_color = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
                            
                            # Process for grayscale and contour
                            template_gray = cv2.cvtColor(template_color, cv2.COLOR_RGB2GRAY)
                            _, thresh = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            contour = max(contours, key=cv2.contourArea) if contours else None

                            symbol_name = f"{subfolder}_{os.path.splitext(filename)[0]}"
                            self.symbol_templates[symbol_name] = (template_color, template_gray, contour)
                            print(f"Loaded template: {symbol_name}")

        if not self.symbol_templates:
            print("No templates found. Exiting.")
            exit()
        print(f"Loaded {len(self.symbol_templates)} templates.")
        input("Press Enter to continue...")

    def match_symbol(self, roi_gray):
        best_match = None
        best_score = float('inf')

        for name, (_, template_gray, _) in self.symbol_templates.items():
            try:
                resized_template_gray = cv2.resize(template_gray, 
                                                   (roi_gray.shape[1], roi_gray.shape[0]))
                result = cv2.matchTemplate(roi_gray, resized_template_gray, 
                                          cv2.TM_SQDIFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)

                if score < best_score:
                    best_score = score
                    best_match = name
            except Exception as e:
                print(f"Error matching template {name}: {e}")

        return (best_match, best_score) if best_score < 0.2 else (None, float('inf'))

def process_reference_image(template_color_rgb):
    """Process reference image in RGB and return BGR for display"""
    # Convert RGB to grayscale
    gray_template = cv2.cvtColor(template_color_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw on RGB image
    ref_img_with_contours = template_color_rgb.copy()
    cv2.drawContours(ref_img_with_contours, contours, -1, (0, 255, 0), 2)  # Green in RGB
    
    # Convert to BGR for OpenCV display
    return cv2.cvtColor(ref_img_with_contours, cv2.COLOR_RGB2BGR)

def initialize_camera():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

def detect_shapes_and_symbols(frame, symbol_recognizer):
    # Camera feed is in BGR format
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_frame_symbol = None
    best_frame_score = float('inf')

    for c in cnts:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)  # Green in BGR
            roi_gray = gray[y:y+h, x:x+w]

            symbol_name, score = symbol_recognizer.match_symbol(roi_gray)
            valid_shape = False

            if symbol_name:
                template_contour = symbol_recognizer.symbol_templates[symbol_name][2]
                if template_contour is not None:
                    shape_score = cv2.matchShapes(c, template_contour, cv2.CONTOURS_MATCH_I1, 0)
                    valid_shape = shape_score < 0.2

            if symbol_name and valid_shape:
                cv2.putText(frame, symbol_name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if score < best_frame_score:
                    best_frame_score = score
                    best_frame_symbol = symbol_name
            else:
                cv2.putText(frame, "Unknown Symbol", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, best_frame_symbol

def main():
    symbol_dir = '/home/raspberry/Documents/S1V1/Sivi_Spring_Project/Symbol-images'
    symbol_recognizer = SymbolRecognizer(symbol_dir)
    picam2 = initialize_camera()

    if picam2 is None:
        return

    try:
        while True:
            frame = picam2.capture_array()  # BGR format from camera
            frame = cv2.flip(frame, -1)
            output_frame, best_symbol = detect_shapes_and_symbols(frame, symbol_recognizer)

            # Show live feed (already in BGR)
            cv2.imshow("Camera Feed", output_frame)

            # Show reference image if match found
            if best_symbol:
                template_color_rgb = symbol_recognizer.symbol_templates[best_symbol][0]
                reference_display = process_reference_image(template_color_rgb)
                cv2.imshow("Reference Feed", reference_display)
            else:
                blank_ref = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(blank_ref, "No match", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Reference Feed", blank_ref)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()