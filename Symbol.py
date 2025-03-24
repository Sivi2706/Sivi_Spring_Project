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

# Function to detect shapes and highlight solid colors within them
def detect_shapes_and_highlight_colors(frame):
    # Convert to grayscale for edge detection
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

            # Create a mask for the detected shape
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [approx], -1, 255, -1)

            # Get the mean color within the shape
            mean_color = cv2.mean(frame, mask=mask)

            # Create a solid color image using the mean color
            solid_color = np.zeros_like(frame)  # Shape will be (480, 640, 4)
            solid_color[:, :] = mean_color[:3]  # Assign BGR values to all pixels

            # Blend the solid color with the original frame using the mask
            highlighted = cv2.bitwise_and(solid_color, solid_color, mask=mask)
            frame = cv2.addWeighted(frame, 1, highlighted, 0.5, 0)

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

    return frame

# Main loop
while True:
    # Capture frame from the camera
    frame = picam2.capture_array()

    # Flip the frame vertically (optional, depending on your camera orientation)
    frame = cv2.flip(frame, -1)

    # Detect shapes and highlight colors
    output_frame = detect_shapes_and_highlight_colors(frame)

    # Display the frame
    cv2.imshow("Camera Feed", output_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()