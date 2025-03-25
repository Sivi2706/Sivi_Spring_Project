import cv2
import numpy as np
import os
from picamera2 import Picamera2

def load_training_data(data_folder):
    images = []
    labels = []
    label_dict = {}
    current_label = 0
    
    # Loop through each subfolder in Symbol-images
    for person_name in os.listdir(data_folder):
        person_dir = os.path.join(data_folder, person_name)
        if os.path.isdir(person_dir):
            label_dict[current_label] = person_name
            # Load all PNG files in the subfolder
            for filename in os.listdir(person_dir):
                if filename.endswith(".png"):
                    img_path = os.path.join(person_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize to consistent size (adjust if needed)
                        img = cv2.resize(img, (100, 100))
                        images.append(img)
                        labels.append(current_label)
            current_label += 1
    
    return images, labels, label_dict

# Initialize face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load training data
train_folder = "Symbol-images"
images, labels, label_dict = load_training_data(train_folder)

if not images:
    print("Error: No training images found in", train_folder)
    exit()

# Train the recognizer
recognizer.train(images, np.array(labels))

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Starting face recognition. Press 'q' to quit.")

while True:
    # Capture frame from camera
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract and resize face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))  # Match training size
        
        # Recognize face
        label, confidence = recognizer.predict(face_roi)
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if confidence < 100:  # Adjust threshold as needed
            text = f"{label_dict.get(label, 'Unknown')} ({confidence:.1f})"
        else:
            text = "Unknown"
        
        cv2.putText(frame, text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Face Recognition', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()