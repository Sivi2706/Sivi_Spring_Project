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

def process_reference_image(template_color):
    """
    Process the full reference (template) image to draw its outlines.
    The image is converted to grayscale, thresholded, and its contours are drawn.
    """
    gray_template = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
    _, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ref_img_with_contours = template_color.copy()
    cv2.drawContours(ref_img_with_contours, contours, -1, (0, 255, 0), 2)
    return ref_img_with_contours

def initialize_camera():
    """
    Initializes the PiCamera2 in BGR888 mode to ensure a 3-channel image.
    """
    try:
        picam2 = Picamera2()
        # Force 640x480 BGR888 output to get a 3-channel image
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
    Processes the live frame to detect shapes, match symbols, overlay the reference
    template on the live feed, and also return a processed reference image with outlines.
    """
    reference_display = None
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold and find contours in the live frame
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # Filter out small contours
        if cv2.contourArea(c) > 500:
            # Get bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(c)

            # Extract grayscale ROI for matching
            roi_gray = gray[y:y+h, x:x+w]

            # Attempt to match symbol using the grayscale ROI
            symbol_name = symbol_recognizer.match_symbol(roi_gray)

            if symbol_name:
                # Retrieve the color template for this symbol
                template_color, template_gray = symbol_recognizer.symbol_templates[symbol_name]
                # Resize the template to match the detected bounding box size
                resized_template_color = cv2.resize(template_color, (w, h))

                # Overlay the resized color template on the live feed
                frame[y:y+h, x:x+w] = resized_template_color

                # Draw bounding box and label on the live frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, symbol_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Process the full reference image to show its outlines
                reference_display = process_reference_image(template_color)
            else:
                # For unknown symbols, draw a blue bounding box with label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Unknown Symbol", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, reference_display

def main():
    # Initialize symbol recognizer with the directory containing template images
    symbol_dir = '/home/raspberry/Documents/S1V1/Sivi_Spring_Project/Symbol-images'
    symbol_recognizer = SymbolRecognizer(symbol_dir)

    # Initialize the camera
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return

    try:
        while True:
            # Capture frame from the camera
            frame = picam2.capture_array()

            # Flip frame if needed (vertical/horizontal flip as required)
            frame = cv2.flip(frame, -1)

            # Process the frame to detect shapes, symbols, and get the reference display
            output_frame, reference_display = detect_shapes_and_symbols(frame, symbol_recognizer)

            # Show the live feed with outlines and identifiers
            cv2.imshow("Camera Feed", output_frame)

            # Show the processed reference image (if a symbol was matched)
            if reference_display is not None:
                cv2.imshow("Reference Feed", reference_display)
            else:
                # If no match, show a black image or a message
                blank_ref = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(blank_ref, "No match", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Reference Feed", blank_ref)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup resources
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()
