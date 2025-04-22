#!/usr/bin/env python

"""
RPi_ShapeRecognizer.py: Optimized for Raspberry Pi with better error handling
"""

import cv2
import glob
import os
import numpy as np

# Configuration - Adjust these for your Raspberry Pi
MATCH_THRESHOLD = 0.75
MIN_CONTOUR_AREA = 500
DISPLAY_SCALE = 0.7
USE_PICAMERA = True  # Set to False for USB camera
RESOLUTION = (640, 480)  # Lower resolution for better performance

# Shape categories (ensure these folders exist)
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

def load_templates():
    """Load template images with error handling"""
    templates = {}
    template_count = 0
    
    print("Loading template database...")
    for folder in folders:
        try:
            pattern = os.path.join(folder, "*.png")
            image_files = glob.glob(pattern)
            
            if not image_files:
                print(f"Warning: No PNG files found in {folder}")
                continue
                
            templates[folder] = []
            
            for file in image_files:
                template = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[folder].append(template)
                    template_count += 1
                else:
                    print(f"Warning: Could not load {file}")
                    
        except Exception as e:
            print(f"Error loading templates from {folder}: {str(e)}")
    
    print(f"Loaded {template_count} templates across {len(templates)} categories")
    return templates

def initialize_camera(use_picamera):
    """Initialize camera with appropriate backend"""
    if use_picamera:
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": RESOLUTION})
            picam2.configure(config)
            picam2.start()
            return picam2
        except ImportError:
            print("picamera2 not found, falling back to USB camera")
        except Exception as e:
            print(f"PiCamera error: {str(e)}, falling back to USB camera")
    
    # Fall back to OpenCV's USB camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try different backends
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                break
                
    if not cap.isOpened():
        raise RuntimeError("Could not initialize any camera")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    return cap

def recognize_shape(test_img, template_db):
    best_match = "unknown"
    best_score = 0
    
    for shape_name, shape_templates in template_db.items():
        for template in shape_templates:
            try:
                if test_img.shape != template.shape:
                    resized_template = cv2.resize(template, (test_img.shape[1], test_img.shape[0]))
                else:
                    resized_template = template
                
                result = cv2.matchTemplate(test_img, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = shape_name
            except Exception as e:
                print(f"Error during template matching: {str(e)}")
                continue
    
    return best_match if best_score > MATCH_THRESHOLD else "unknown", best_score

def main():
    # Load templates
    templates = load_templates()
    if not templates:
        print("Error: No templates loaded. Check your folders and images.")
        return
    
    # Initialize camera
    try:
        camera = initialize_camera(USE_PICAMERA)
    except Exception as e:
        print(f"Camera initialization failed: {str(e)}")
        return
    
    print("Starting live recognition. Press 'q' to quit...")
    
    try:
        while True:
            # Capture frame
            if USE_PICAMERA and hasattr(camera, 'capture_array'):
                frame = camera.capture_array()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break
            
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                roi = gray[y:y+h, x:x+w]
                
                shape, confidence = recognize_shape(roi, templates)
                
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{shape} ({confidence:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 1)
            
            # Display
            h, w = frame.shape[:2]
            display_frame = cv2.resize(frame, (int(w*DISPLAY_SCALE), int(h*DISPLAY_SCALE)))
            cv2.imshow('Shape Recognition', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if USE_PICAMERA and hasattr(camera, 'stop'):
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()