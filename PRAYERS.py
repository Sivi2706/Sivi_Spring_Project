import numpy as np
import cv2
import os
from picamera2 import Picamera2

class SymbolRecognizer:
    def __init__(self, symbol_dir):
        self.symbol_dir = symbol_dir
        # Dictionary: symbol_name -> (color_template (RGB), gray_template, contour)
        self.symbol_templates = {}
        self.subfolders = []
        self.load_subfolders()

    def load_subfolders(self):
        """Load list of subfolders for calibration"""
        for subfolder in os.listdir(self.symbol_dir):
            subfolder_path = os.path.join(self.symbol_dir, subfolder)
            if os.path.isdir(subfolder_path):
                self.subfolders.append(subfolder)

    def interactive_calibration(self):
        """Interactive calibration to capture sample images"""
        picam2 = initialize_camera()
        if picam2 is None:
            print("Camera initialization failed.")
            return False

        try:
            for subfolder in self.subfolders:
                # Create output directory for current subfolder
                output_dir = os.path.join(self.symbol_dir, subfolder, 'samples')
                os.makedirs(output_dir, exist_ok=True)

                print(f"\nPreparing to capture images for: {subfolder}")
                print("Position the symbol in the camera view.")
                input("Press Enter when ready to start capturing...")

                sample_count = 0
                while sample_count < 25:
                    # Capture frame
                    frame = picam2.capture_array()
                    frame = cv2.flip(frame, -1)  # Flip if needed

                    # Display frame with sample count
                    display_frame = frame.copy()
                    cv2.putText(display_frame, 
                                f"Samples: {sample_count}/25", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
                    cv2.imshow("Calibration", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        # Save image
                        filename = os.path.join(output_dir, f'sample_{sample_count:02d}.png')
                        cv2.imwrite(filename, frame)
                        sample_count += 1
                        print(f"Saved sample {sample_count}")

                    elif key == ord('q'):
                        # Option to quit current subfolder
                        break

                print(f"Completed capturing {sample_count} samples for {subfolder}")

            return True
        finally:
            cv2.destroyAllWindows()
            picam2.stop()

    def calibrate(self):
        """Standard calibration using existing images"""
        print("Starting Calibration Stage...")

        for subfolder in os.listdir(self.symbol_dir):
            subfolder_path = os.path.join(self.symbol_dir, subfolder)

            if os.path.isdir(subfolder_path):
                # Check both original images and sample images
                image_paths = []
                
                # Original template images
                image_paths.extend([
                    os.path.join(subfolder_path, f) 
                    for f in os.listdir(subfolder_path) 
                    if f.lower().endswith('.png') and 'samples' not in f
                ])
                
                # Sample images
                samples_path = os.path.join(subfolder_path, 'samples')
                if os.path.exists(samples_path):
                    image_paths.extend([
                        os.path.join(samples_path, f) 
                        for f in os.listdir(samples_path) 
                        if f.lower().endswith('.png')
                    ])

                for full_path in image_paths:
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

                        symbol_name = f"{subfolder}_{os.path.splitext(os.path.basename(full_path))[0]}"
                        self.symbol_templates[symbol_name] = (template_color, template_gray, contour)
                        print(f"Loaded template: {symbol_name}")

        if not self.symbol_templates:
            print("No templates found. Exiting.")
            exit()
        print(f"Loaded {len(self.symbol_templates)} templates.")

    # ... [rest of the SymbolRecognizer class remains the same as in the original code]

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

def main():
    symbol_dir = '/home/raspberry/Documents/S1V1/Sivi_Spring_Project/Symbol-images'
    symbol_recognizer = SymbolRecognizer(symbol_dir)

    # Interactive calibration
    print("Symbol Recognizer Calibration")
    print("-----------------------------")
    print("Instructions:")
    print("1. Position the symbol in the camera view")
    print("2. Press 's' to save a sample image")
    print("3. Capture 25 samples for each symbol")
    print("4. Press 'q' to skip to the next symbol or exit")
    input("Press Enter to start calibration...")

    # Run interactive calibration
    if symbol_recognizer.interactive_calibration():
        # Proceed to main symbol detection
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

# Include the existing detect_shapes_and_symbols and process_reference_image functions from the original code
# [These functions would be copied from the original script]

if __name__ == "__main__":
    main()