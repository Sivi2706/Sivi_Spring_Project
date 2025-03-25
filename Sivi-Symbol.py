import numpy as np
import cv2
import os
from picamera2 import Picamera2
import pickle

class SymbolDetector:
    def __init__(self):
        self.reference_symbols = []
        self.min_contour_area = 500
        self.match_threshold = 0.15  # Lower is better match
        self.initialize_camera()

    def initialize_camera(self):
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (640, 480)}))
            self.picam2.start()
        except RuntimeError as e:
            print(f"Camera initialization failed: {e}")
            exit()

    def process_reference_images(self, folder_path="Symbol-images"):
        if not os.path.exists(folder_path):
            print(f"Reference folder {folder_path} not found!")
            return False

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                frame = cv2.imread(image_path)
                
                if frame is not None:
                    # Get symbol name without extension
                    symbol_name = os.path.splitext(filename)[0]
                    
                    # Process the reference image
                    edges, contours = self.detect_edges(frame)
                    
                    if contours:
                        # Store the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        self.reference_symbols.append({
                            'name': symbol_name,
                            'contour': largest_contour,
                            'shape': self.determine_shape(largest_contour)
                        })
                        print(f"Loaded reference: {symbol_name}")
        
        if not self.reference_symbols:
            print("No valid reference images found!")
            return False
        
        # Save references for future use
        self.save_references()
        return True

    def detect_edges(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Combined color ranges for better edge detection
        lower = np.array([0, 50, 50])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Edge detection
        edges = cv2.Canny(mask, 30, 100)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return edges, [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

    def determine_shape(self, contour):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 3:
            return "Triangle"
        elif len(approx) == 4:
            return "Rectangle"
        elif len(approx) == 5:
            return "Pentagon"
        elif len(approx) >= 6:
            return "Circle"
        return "Unknown"

    def match_contour(self, contour):
        for ref in self.reference_symbols:
            # Compare shapes using Hu moments
            match_value = cv2.matchShapes(ref['contour'], contour, cv2.CONTOURS_MATCH_I2, 0)
            if match_value < self.match_threshold:
                return ref['name']
        return None

    def save_references(self, filename="symbol_references.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.reference_symbols, f)

    def load_references(self, filename="symbol_references.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.reference_symbols = pickle.load(f)
            return True
        return False

    def process_frame(self, frame):
        # Convert from RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detect edges and contours
        edges, contours = self.detect_edges(frame)
        
        # Draw edges (semi-transparent)
        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        output = cv2.addWeighted(frame, 0.7, edge_display, 0.3, 0)
        
        # Check each contour against references
        for contour in contours:
            matched_name = self.match_contour(contour)
            if matched_name:
                # Draw bounding box and label
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output, matched_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output

    def run(self):
        # Try to load saved references first
        if not self.load_references():
            # If no saved references, process the images
            if not self.process_reference_images():
                print("No reference data available. Exiting.")
                return

        print("Starting live detection. Press 'q' to quit.")
        try:
            while True:
                frame = self.picam2.capture_array()
                output = self.process_frame(frame)
                cv2.imshow('Symbol Detection', output)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            self.picam2.stop()

if __name__ == "__main__":
    detector = SymbolDetector()
    detector.run()