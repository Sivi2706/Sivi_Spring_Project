import numpy as np
import cv2
import os
from picamera2 import Picamera2

class SymbolRecognizer:
    def __init__(self, symbol_dir):
        self.symbol_dir = symbol_dir
        # Dictionary: symbol_name -> (color_template (BGR), gray_template)
        self.symbol_templates = {}
        self.calibrate()

    def calibrate(self):
        print("Starting Calibration Stage...")

        # Iterate through subfolders in symbol_dir
        for subfolder in os.listdir(self.symbol_dir):
            subfolder_path = os.path.join(self.symbol_dir, subfolder)

            if os.path.isdir(subfolder_path):
                # Iterate through PNG files in subfolder
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith('.png'):
                        full_path = os.path.join(subfolder_path, filename)

                        # Load template in color (BGR)
                        template_color = cv2.imread(full_path, cv2.IMREAD_COLOR)

                        if template_color is not None:
                            # Convert to grayscale for matching
                            template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

                            # Create symbol name from subfolder and filename
                            symbol_name = f"{subfolder}_{os.path.splitext(filename)[0]}"

                            # Store both BGR color and grayscale versions
                            self.symbol_templates[symbol_name] = (template_color, template_gray)
                            print(f"Loaded template: {symbol_name}")

        if not self.symbol_templates:
            print("No templates found. Exiting.")
            exit()
        print(f"Loaded {len(self.symbol_templates)} templates.")
        input("Press Enter to continue...")

    def match_symbol(self, roi_gray):
        """
        Attempt to match the given grayscale ROI with one of the stored templates.
        Returns (best_match, best_score). If no match is good enough, returns (None, inf).
        """
        best_match = None
        best_score = float('inf')

        for name, (template_color, template_gray) in self.symbol_templates.items():
            try:
                # Resize the grayscale template to match the ROI size
                resized_template_gray = cv2.resize(template_gray, (roi_gray.shape[1], roi_gray.shape[0]))

                # Compute template matching (SQDIFF_NORMED => lower is better)
                result = cv2.matchTemplate(roi_gray, resized_template_gray, cv2.TM_SQDIFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)

                if score < best_score:
                    best_score = score
                    best_match = name
            except Exception as e:
                print(f"Error matching template {name}: {e}")

        if best_score < 0.2:
            return best_match, best_score
        else:
            return None, float('inf')

def process_reference_image(template_color):
    """
    Process the reference (template) image (in BGR) by converting it to grayscale,
    thresholding it, finding contours, and drawing the contours in green on the color image.
    """
    gray_template = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
    _, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_img_with_contours = template_color.copy()
    cv2.drawContours(ref_img_with_contours, contours, -1, (0, 255, 0), 2)
    return ref_img_with_contours

def initialize_camera():
    """
    Initializes the PiCamera2 to output 3-channel BGR frames at 640x480.
    """
    try:
        picam2 = Picamera2()
        # Force 640x480 BGR888 output to get a 3-channel image in BGR
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
    """
    1. Finds contours in the live frame (in BGR).
    2. For each contour, tries to match a symbol using the grayscale ROI.
    3. Labels the contour in the live feed.
    4. Returns the frame (with drawn contours and labels) and the best-matching symbol name.
    """
    # Convert frame to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold and find contours in the grayscale image
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

            if symbol_name:
                cv2.putText(frame, symbol_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if score < best_frame_score:
                    best_frame_score = score
                    best_frame_symbol = symbol_name
            else:
                cv2.putText(frame, "Unknown Symbol", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, best_frame_symbol

def main():
    symbol_dir = '/home/raspberry/Documents/S1V1/Sivi_Spring_Project/Symbol-images'
    symbol_recognizer = SymbolRecognizer(symbol_dir)

    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return

    try:
        while True:
            # Capture frame from the camera (in BGR)
            frame = picam2.capture_array()

            # Optionally flip the frame if needed (vertical flip example)
            frame = cv2.flip(frame, -1)

            # Detect shapes and symbols in the frame
            output_frame, best_symbol_name = detect_shapes_and_symbols(frame, symbol_recognizer)

            cv2.imshow("Camera Feed", output_frame)

            # Show the reference image with outlines for the best match
            if best_symbol_name:
                template_color, _ = symbol_recognizer.symbol_templates[best_symbol_name]
                reference_display = process_reference_image(template_color)
                cv2.imshow("Reference Feed", reference_display)
            else:
                blank_ref = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(blank_ref, "No match", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Reference Feed", blank_ref)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()
