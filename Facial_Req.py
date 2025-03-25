import cv2
import numpy as np
import os
from picamera2 import Picamera2

def load_single_image_data(data_folder):
    images = []
    labels = []
    label_dict = {}
    
    symbol_folders = [f for f in os.listdir(data_folder) 
                     if os.path.isdir(os.path.join(data_folder, f))]
    
    for label, symbol_folder in enumerate(symbol_folders):
        symbol_path = os.path.join(data_folder, symbol_folder)
        png_files = [f for f in os.listdir(symbol_path) 
                    if f.lower().endswith('.png')]
        
        if png_files:
            img_path = os.path.join(symbol_path, png_files[0])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                images.append(img)
                labels.append(label)
                label_dict[label] = symbol_folder
            
    return images, labels, label_dict

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load training data
train_folder = "Symbol-images"
images, labels, label_dict = load_single_image_data(train_folder)

if not images:
    print(f"No training images found in {train_folder}")
    exit()

recognizer.train(images, np.array(labels))

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# FIXED: Use absolute path to Haar cascade
cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    cascade_path = cv2.__file__.replace("__init__.py", "data/haarcascade_frontalface_default.xml")

detector = cv2.CascadeClassifier(cascade_path)

print("Starting recognition. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detections = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    for (x, y, w, h) in detections:
        roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(roi)
        
        color = (0, 255, 0) if confidence < 80 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{label_dict.get(label, 'Unknown')} ({confidence:.1f})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Symbol Recognition', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()