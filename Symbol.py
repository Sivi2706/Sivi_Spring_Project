import numpy as np
import cv2
from picamera2 import Picamera2

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

picam2 = initialize_camera()
if picam2 is None:
    print("Exiting program. Camera could not be initialized.")
    exit()

def detect_shapes_and_arrows(frame):
    # Convert to grayscale and apply edge detection on full frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours for overall shapes
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > 1000:  # Ignore small contours
            # Approximate the contour for shape detection
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)

            # Draw the contour and bounding box
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Determine the basic shape type based on vertices count
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 7:
                shape = "Arrow"
            else:
                shape = "Circle"

            # Process the ROI to detect an arrow inside the shape
            roi = gray[y:y+h, x:x+w]
            roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
            roi_edges = cv2.Canny(roi_blur, 50, 150)
            # Apply a morphological closing to join broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            roi_closed = cv2.morphologyEx(roi_edges, cv2.MORPH_CLOSE, kernel)
            roi_cnts, _ = cv2.findContours(roi_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            arrow_found = False
            for ac in roi_cnts:
                if cv2.contourArea(ac) > 300:  # Adjust threshold for arrow details
                    # Adjust the epsilon factor for finer approximation
                    epsilon2 = 0.02 * cv2.arcLength(ac, True)
                    approx2 = cv2.approxPolyDP(ac, epsilon2, True)
                    # Check if the approximated contour has around 7 vertices
                    if len(approx2) == 7:
                        arrow_found = True
                        # Optionally, draw the arrow contour in a different color
                        pts = approx2 + [x, y]  # Adjust coordinates relative to full image
                        cv2.drawContours(frame, [pts], -1, (0, 255, 255), 2)
                        break

            # Update label based on arrow detection
            if arrow_found and shape != "Arrow":
                label = f"{shape} with Arrow"
            else:
                label = shape

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# Main loop
while True:
    # Capture frame from the camera
    frame = picam2.capture_array()

    # Flip the frame vertically (optional, depending on your camera orientation)
    frame = cv2.flip(frame, -1)

    # Detect shapes and search for an arrow inside each shape
    output_frame = detect_shapes_and_arrows(frame)

    # Display the frame
    cv2.imshow("Camera Feed", output_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()