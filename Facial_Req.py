import cv2
import numpy as np
import os
from picamera2 import Picamera2
from imutils import resize

def load_single_image_data(data_folder):
    images = []
    labels = []
    label_dict = {}
    
    # Get all subfolders (one per symbol)
    symbol_folders = [f for f in os.listdir(data_folder) 
                     if os.path.isdir(os.path.join(data_folder, f))]
    
    for label, symbol_folder in enumerate(symbol_folders):
        symbol_path = os.path.join(data_folder, symbol_folder)
        png_files = [f for f in os.listdir(symbol_path) 
                    if f.lower().endswith('.png')]
        
        if not png_files:
            continue
            
        # Load the first (and only) PNG file
        img_path = os.path.join(symbol_path, png_files[0])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Resize and add to training data
            img = resize(img, width=200)  # Match detection size
            images.append(img)
            labels.append(label)
            label_dict[label] = symbol_folder  # Store label name
            
    return images, labels, label_dict

# Initialize LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16)

# Load training data
train_folder = "Symbol-images"
images, labels, label_dict = load_single_image_data(train_folder)

if not images:
    print(f"No valid training images found in {train_folder}")
    exit()

# Train the model (even with single images)
recognizer.train(images, np.array(labels))

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Initialize face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

print("Starting recognition. Press 'q' to quit.")

while True:
    # Capture frame
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect symbols/faces
    detections = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in detections:
        # Process region of interest
        roi = gray[y:y+h, x:x+w]
        roi = resize(roi, width=200)  # Match training size
        
        # Recognize symbol
        label, confidence = recognizer.predict(roi)
        
        # Draw results
        color = (0, 255, 0) if confidence < 80 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        text = f"{label_dict.get(label, 'Unknown')} ({confidence:.1f})"
        cv2.putText(frame, text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display output
    cv2.imshow('Symbol Recognition', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()