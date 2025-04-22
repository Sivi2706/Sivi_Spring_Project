#!/usr/bin/env python

"""
LiveShapeRecognizer.py: Real-time shape recognition with OpenCV, displaying outlines and results.
"""

import cv2
import glob
import os
import numpy as np

# Configuration
MATCH_THRESHOLD = 0.75  # Minimum similarity score to consider a match
MIN_CONTOUR_AREA = 500  # Minimum area to consider for shape detection
DISPLAY_SCALE = 0.7     # Scale factor for display window

# Get all folders (shape categories)
folders = [
    "3 quaters of a circle",
    "Arrow down",
    "Arrow up",
    "Blue rectangle",
    "Facial recompulsion",
    "Hexagon",
    "Left arrow",
    "Measure Distance",
    "Pentagon",
    "Red circle",
    "Right arrow",
    "Stop sign",
    "Traffic light stop",
    "Triangle"
]

# Build template database
templates = {}
print("Loading template database...")
for folder in folders:
    pattern = os.path.join(folder, "*.png")
    image_files = glob.glob(pattern)
    
    for file in image_files:
        template = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        
        if folder not in templates:
            templates[folder] = []
        templates[folder].append(template)
        
print(f"Loaded {sum(len(v) for v in templates.values())} templates across {len(templates)} categories")

def recognize_shape(test_img, template_db):
    best_match = "unknown"
    best_score = 0
    
    for shape_name, shape_templates in template_db.items():
        for template in shape_templates:
            if test_img.shape != template.shape:
                resized_template = cv2.resize(template, (test_img.shape[1], test_img.shape[0]))
            else:
                resized_template = template
            
            result = cv2.matchTemplate(test_img, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = shape_name
    
    return best_match if best_score > MATCH_THRESHOLD else "unknown", best_score

def process_frame(frame):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    processed_frame = frame.copy()
    
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        
        # Get bounding box and ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        
        # Recognize shape
        shape, confidence = recognize_shape(roi, templates)
        
        # Draw contour and label
        cv2.drawContours(processed_frame, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        label = f"{shape} ({confidence:.2f})"
        cv2.putText(processed_frame, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return processed_frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting live recognition. Press 'q' to quit...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result_frame = process_frame(frame)
        
        # Resize for display
        h, w = result_frame.shape[:2]
        display_frame = cv2.resize(result_frame, (int(w*DISPLAY_SCALE), int(h*DISPLAY_SCALE)))
        
        # Show result
        cv2.imshow('Shape Recognition', display_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()