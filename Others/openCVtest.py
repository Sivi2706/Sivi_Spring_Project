import cv2

# Define GStreamer pipeline
pipeline = (
    "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! video/x-raw, format=GRAY8 ! appsink"
)

# Open the video capture
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Display the video feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Grayscale Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
