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

# Function to detect shapes and arrows
def detect_shapes_and_arrows(frame):
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > 1000:  # Ignore small contours
            # Approximate the contour
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)

            # Draw the contour and bounding box
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Detect shapes based on the number of vertices
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

            # Label the shape
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Convert the grayscale edges back to BGR (RGB) for display
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Optionally, you can overlay the edges on the original frame
    output_frame = cv2.addWeighted(frame, 0.8, edges_bgr, 0.2, 0)

    return output_frame
# Main loop
while True:
    # Capture frame from the camera
    frame = picam2.capture_array()

    # Flip the frame vertically (optional, depending on your camera orientation)
    frame = cv2.flip(frame, -1)

    # Detect shapes and arrows
    output_frame = detect_shapes_and_arrows(frame)

    # Display the frame
    cv2.imshow("Camera Feed", output_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()