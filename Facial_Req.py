import numpy as np
import cv2
import os
from picamera2 import Picamera2

class SymbolRecognizer:
    def __init__(self, symbol_dir):
        self.symbol_dir = symbol_dir
        # Dictionary: symbol_name -> (color_template, gray_template)
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

                        # Load template in color
                        template_color = cv2.imread(full_path, cv2.IMREAD_COLOR)

                        if template_color is not None:
                            # Convert to grayscale for matching
                            template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

                            # Create symbol name from subfolder and filename
                            symbol_name = f"{subfolder}_{os.path.splitext(filename)[0]}"

                            # Store both color and gray versions
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
        Returns the best matching symbol name if below a similarity threshold,
        otherwise None.
        """
        best_match = None
        best_score = float('inf')

        for name, (template_color, template_gray) in self.symbol_templates.items():
            try:
                # Resize the grayscale template to match the ROI size
                resized_template_gray = cv2.resize(template_gray, (roi_gray.shape[1], roi_gray.shape[0]))

                # Compute template matching (SQDIFF is minimized for better matches)
                result = cv2.matchTemplate(roi_gray, resized_template_gray, cv2.TM_SQDIFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)

                if score < best_score:
                    best_score = score
                    best_match = name
            except Exception as e:
                print(f"Error matching template {name}: {e}")

        # Return the best match if it's reasonably close
        return best_match if best_score < 0.2 else None

def initialize_camera():
    """
    Initializes the PiCamera2 in BGR888 mode to ensure a 3-channel image
    and avoid broadcasting errors with 4-channel frames.
    """
    try:
        picam2 = Picamera2()
        # Force 640x480 BGR888 output
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
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold and find contours
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # Filter contours by area
        if cv2.contourArea(c) > 500:
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(c)

            # Extract grayscale ROI for matching
            roi_gray = gray[y:y+h, x:x+w]

            # Attempt to match symbol
            symbol_name = symbol_recognizer.match_symbol(roi_gray)

            if symbol_name:
                # Retrieve the color template
                template_color, template_gray = symbol_recognizer.symbol_templates[symbol_name]
                # Resize to match the bounding box
                resized_template_color = cv2.resize(template_color, (w, h))

                # Overlay the color template onto the original frame
                frame[y:y+h, x:x+w] = resized_template_color

                # Optionally, draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, symbol_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Unknown symbol
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Unknown Symbol", (x, y - 10),
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

            # Flip frame if needed (vertical flip in this example)
            frame = cv2.flip(frame, -1)

            # Detect shapes and symbols, overlay original color if matched
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
