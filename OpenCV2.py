import numpy as np
import cv2
from picamera2 import Picamera2

# Initialize Raspberry Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

def nothing(x):
    pass

# Create trackbars for dynamic HSV adjustment
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    # Capture frame from Raspberry Pi camera
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Define color range based on trackbars (for general tuning)
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Define specific range for black color detection
    lower_black = np.array([0, 0, 20])  # Ignore very faint shadows
    upper_black = np.array([180, 255, 40])  # Less sensitive to slightly brighter areas

    # Threshold the images
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours for black color detection
    cnts, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > 1000:  # Ignore small noise
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "BLACK DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show images
    cv2.imshow("Mask (Trackbar Range)", mask)
    cv2.imshow("Mask (Black Detection)", mask_black)
    cv2.imshow("Result", result)
    cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()