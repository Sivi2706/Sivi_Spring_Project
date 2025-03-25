import cv2
import numpy as np
import os
from picamera2 import Picamera2

def load_training_data(data_folder):
    images = []
    labels = []
    label_dict = {}
    folder_names = []
    
    symbol_folders = sorted([f for f in os.listdir(data_folder) 
                          if os.path.isdir(os.path.join(data_folder, f))])
    
    for label, folder in enumerate(symbol_folders):
        folder_path = os.path.join(data_folder, folder)
        png_files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith('.png')]
        
        if png_files:
            img_path = os.path.join(folder_path, png_files[0])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                images.append(img)
                labels.append(label)
                label_dict[label] = folder
                folder_names.append(folder)
    
    return images, labels, label_dict, folder_names

def validate_training(recognizer, images, labels, label_dict, folder_names):
    # Create validation window
    cv2.namedWindow('Validation Results', cv2.WINDOW_NORMAL)
    
    for idx, (img, true_label) in enumerate(zip(images, labels)):
        # Predict using trained model
        predicted_label, confidence = recognizer.predict(img)
        
        # Get actual and predicted names
        actual_name = folder_names[true_label]
        predicted_name = label_dict.get(predicted_label, "Unknown")
        
        # Create display image with outline
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Choose outline color (green=correct, red=incorrect)
        color = (0, 255, 0) if predicted_name == actual_name else (0, 0, 255)
        thickness = 5
        
        # Add border
        display_img = cv2.copyMakeBorder(
            display_img,
            thickness, thickness, thickness, thickness,
            cv2.BORDER_CONSTANT,
            value=color
        )
        
        # Add text
        text = f"Actual: {actual_name} | Predicted: {predicted_name}"
        cv2.putText(display_img, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show results
        cv2.imshow('Validation Results', display_img)
        cv2.waitKey(2000)  # Show each result for 2 seconds
    
    cv2.destroyWindow('Validation Results')

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load training data
train_folder = "Symbol-images"
images, labels, label_dict, folder_names = load_training_data(train_folder)

if not images:
    print(f"No training images found in {train_folder}")
    exit()

# Train and validate
recognizer.train(images, np.array(labels))
validate_training(recognizer, images, labels, label_dict, folder_names)

# Real-time recognition setup
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load detector
cascade_path = cv2.__file__.replace("__init__.py", "data/haarcascade_frontalface_default.xml")
detector = cv2.CascadeClassifier(cascade_path)

print("Starting real-time recognition. Press 'q' to quit.")

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
        predicted_name = label_dict.get(label, "Unknown")
        
        # Draw outline and text
        color = (0, 255, 0) if confidence < 80 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, f"{predicted_name} ({confidence:.1f})",
                    (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Real-time Recognition', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()