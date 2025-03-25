import numpy as np
import cv2
import os
from picamera2 import Picamera2
import pickle
import math

class AdvancedSymbolDetector:
    def __init__(self):
        self.reference_symbols = []
        self.min_contour_area = 500
        self.shape_match_threshold = 0.2  # Lower is better match
        self.symbol_match_threshold = 0.3  # Higher threshold for symbol matching
        self.initialize_camera()
        self.setup_detection_parameters()

    def setup_detection_parameters(self):
        # Parameters for feature detection
        self.orb = cv2.ORB_create()
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
            dict(checks=50)
        )

    def initialize_camera(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(main={"size": (1280, 720)})
            self.picam2.configure(config)
            self.picam2.start()
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            exit()

    def process_reference_images(self, folder_path="Symbol-images"):
        if not os.path.exists(folder_path):
            print(f"Reference folder {folder_path} not found!")
            return False

        # Clear existing references before processing
        self.reference_symbols = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                frame = cv2.imread(image_path)
                
                if frame is not None:
                    symbol_name = os.path.splitext(filename)[0]
                    processed = self.process_symbol(frame, symbol_name)
                    if processed:
                        print(f"Loaded reference: {symbol_name}")
        
        if not self.reference_symbols:
            print("No valid reference images found!")
            return False
        
        self.save_references()
        return True

    def process_symbol(self, frame, symbol_name):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return False

        # Get shape features
        shape_type = self.determine_shape(largest_contour)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None:
            return False

        # Convert KeyPoints to picklable format
        picklable_kp = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp]

        self.reference_symbols.append({
            'name': symbol_name,
            'contour': largest_contour,
            'shape': shape_type,
            'keypoints': picklable_kp,
            'descriptors': des
        })
        return True

    def determine_shape(self, contour):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices == 3:
            return "Triangle"
        elif num_vertices == 4:
            # Check if square or rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif num_vertices == 5:
            return "Pentagon"
        elif num_vertices == 6:
            return "Hexagon"
        elif num_vertices >= 8:
            # Check circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            return "Circle" if circularity > 0.8 else "Irregular"
        return "Unknown"

    def match_symbol(self, frame, contour):
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect features in ROI
        kp, des = self.orb.detectAndCompute(gray_roi, None)
        if des is None:
            return None

        best_match = None
        best_score = 0
        
        for ref in self.reference_symbols:
            if ref['descriptors'] is None:
                continue
                
            matches = self.flann.knnMatch(ref['descriptors'], des, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) > 10:  # Minimum matches threshold
                match_score = len(good_matches) / len(ref['keypoints'])
                if match_score > best_score and match_score > 0.2:  # Minimum score threshold
                    best_score = match_score
                    best_match = ref['name']
        
        return best_match

    def save_references(self, filename="symbol_references_advanced.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.reference_symbols, f)

    def load_references(self, filename="symbol_references_advanced.pkl"):
        try:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                with open(filename, 'rb') as f:
                    self.reference_symbols = pickle.load(f)
                return True
            else:
                print(f"Reference file {filename} not found or is empty.")
                return False
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading references: {e}")
            return False

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output = frame.copy()
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
                
            shape_type = self.determine_shape(contour)
            matched_name = self.match_symbol(frame, contour)
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw contour and bounding box
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Prepare label
            label = shape_type
            if matched_name:
                label = f"{matched_name} ({shape_type})"
            
            cv2.putText(output, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return output

    def run(self):
        if not self.load_references():
            if not self.process_reference_images():
                print("No reference data available. Exiting.")
                return

        print("Starting live detection. Press 'q' to quit.")
        try:
            while True:
                frame = self.picam2.capture_array()
                output = self.process_frame(frame)
                cv2.imshow('Advanced Symbol Detection', output)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            self.picam2.stop()

if __name__ == "__main__":
    detector = AdvancedSymbolDetector()
    detector.run()