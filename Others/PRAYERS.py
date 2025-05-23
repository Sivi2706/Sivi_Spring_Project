import numpy as np
import cv2
import os
from picamera2 import Picamera2
import time

class SymbolRecognizer:
    def __init__(self, symbol_dir):
        self.symbol_dir = symbol_dir
        self.raspi_img_dir = os.path.join(os.path.dirname(symbol_dir), "raspi-img")
        self.symbol_templates = {}
        self.calibrate()

    def calibrate(self):
        print("Starting Calibration Stage...")
        print("\nInitializing camera for calibration...")
        
        # Initialize camera for calibration
        calib_cam = Picamera2()
        config = calib_cam.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        calib_cam.configure(config)
        calib_cam.start()

        # Create symbol directories if they don't exist
        os.makedirs(self.symbol_dir, exist_ok=True)
        os.makedirs(self.raspi_img_dir, exist_ok=True)

        # Get list of existing subfolders or create default ones
        subfolders = [d for d in os.listdir(self.symbol_dir) 
                     if os.path.isdir(os.path.join(self.symbol_dir, d))]
        
        if not subfolders:
            print("No subfolders found in symbol directory. Creating default ones...")
            default_symbols = ["Arrow_up", "Arrow_down", "Circle", "Square", "Triangle"]
            for symbol in default_symbols:
                os.makedirs(os.path.join(self.symbol_dir, symbol), exist_ok=True)
            subfolders = default_symbols

        print("\nAvailable symbols for calibration:")
        for i, subfolder in enumerate(subfolders):
            print(f"{i+1}. {subfolder}")

        # Capture samples for each subfolder
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.symbol_dir, subfolder)
            raspi_subfolder_path = os.path.join(self.raspi_img_dir, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            os.makedirs(raspi_subfolder_path, exist_ok=True)
            
            print(f"\n=== Calibrating {subfolder} ===")
            print("Position the symbol in view. Press 's' to capture the image.")
            print("Press 'q' to skip to next symbol or 'x' to exit calibration.")
            
            cv2.namedWindow("Calibration Feed", cv2.WINDOW_NORMAL)
            capturing = False
            skip_symbol = False

            while True:
                frame = calib_cam.capture_array()
                frame = cv2.flip(frame, -1)  # Adjust rotation if needed
                
                display_frame = frame.copy()
                
                if capturing:
                    # Save captured frame to both directories
                    timestamp = int(time.time())
                    filename = f"{subfolder}_{timestamp}.png"
                    
                    # Save to original symbol directory
                    cv2.imwrite(os.path.join(subfolder_path, filename), frame)
                    
                    # Save to raspi-img directory
                    cv2.imwrite(os.path.join(raspi_subfolder_path, filename), frame)
                    
                    print(f"Captured 1 image for {subfolder}")
                    capturing = False
                    break
                else:
                    cv2.putText(display_frame, f"Calibrating: {subfolder}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, "Press 's' to capture image", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, "Press 'q' to skip", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Calibration Feed", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') and not capturing:
                    capturing = True
                    print(f"Capturing image for {subfolder}...")
                elif key == ord('q'):
                    print(f"Skipping {subfolder}")
                    skip_symbol = True
                    break
                elif key == ord('x'):
                    print("Exiting calibration early...")
                    calib_cam.stop()
                    cv2.destroyAllWindows()
                    return

            cv2.destroyWindow("Calibration Feed")
            if skip_symbol:
                continue

        # Stop calibration camera
        calib_cam.stop()
        cv2.destroyAllWindows()

        # Load templates from captured images
        print("\nLoading templates from captured images...")
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.symbol_dir, subfolder)
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
        print(f"\nLoaded {len(self.symbol_templates)} templates.")
        input("Calibration complete. Press Enter to start detection...")

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
    gray_template = cv2.cvtColor(template_color_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_img_with_contours = template_color_rgb.copy()
    cv2.drawContours(ref_img_with_contours, contours, -1, (0, 255, 0), 2)
    
    return cv2.cvtColor(ref_img_with_contours, cv2.COLOR_RGB2BGR)

def initialize_camera():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        # Add small delay to allow camera to initialize
        time.sleep(2)
        return picam2
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        return None

def detect_shapes_and_symbols(frame, symbol_recognizer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_frame_symbol = None
    best_frame_score = float('inf')

    for c in cnts:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
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
    
    # Initialize detection camera
    picam2 = initialize_camera()
    if picam2 is None:
        return

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.flip(frame, -1)
            output_frame, best_symbol = detect_shapes_and_symbols(frame, symbol_recognizer)

            cv2.imshow("Camera Feed", output_frame)

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