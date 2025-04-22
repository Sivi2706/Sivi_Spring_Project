import cv2

# Load the Haar Cascade classifier
cascade_path = 'stop_data.xml'
stop_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade classifier loaded successfully
if stop_cascade.empty():
    print("Error: Couldn't load Haar Cascade classifier.")
    exit()

# Initialize video capture from the default camera (usually webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Couldn't open the camera.")
    exit()

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects in the grayscale frame
    found = stop_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # Draw rectangles around detected objects
    for (x, y, w, h) in found:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # Display the frame with detected objects
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()