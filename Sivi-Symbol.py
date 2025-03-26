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
        self.color_threshold = 50  # Allowable color difference
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

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):  # Ensure it's a folder
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith('.png'):
                        image_path = os.path.join(subfolder_path, filename)
                        frame = cv2.imread(image_path)
                        
                        if frame is not None:
                            symbol_name = subfolder  # Use folder name as symbol name
                            edges, contours = self.detect_edges(frame)
                            dominant_color = self.get_dominant_color(frame)
                            
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                self.reference_symbols.append({
                                    'name': symbol_name,
                                    'contour': largest_contour,
                                    'shape': self.determine_shape(largest_contour),
                                    'color': dominant_color
                                })
                                print(f"Loaded reference: {symbol_name}")
        
        if not self.reference_symbols:
            print("No valid reference images found!")
            return False
        
        self.save_references()
        return True

    def detect_edges(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 50, 50])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.Canny(mask, 30, 100)
        edges = cv2.dilate(edges, kernel, iterations=1)
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

    def get_dominant_color(self, frame):
        pixels = frame.reshape(-1, 3)
        avg_color = np.mean(pixels, axis=0)
        return avg_color

    def match_contour(self, contour, frame):
        for ref in self.reference_symbols:
            match_value = cv2.matchShapes(ref['contour'], contour, cv2.CONTOURS_MATCH_I2, 0)
            if match_value < self.match_threshold:
                # Extract color from detected contour
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                detected_color = self.get_dominant_color(cv2.bitwise_and(frame, frame, mask=mask))
                
                # Compare color similarity
                if np.linalg.norm(detected_color - ref['color']) < self.color_threshold:
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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        edges, contours = self.detect_edges(frame)
        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        output = cv2.addWeighted(frame, 0.7, edge_display, 0.3, 0)
        for contour in contours:
            matched_name = self.match_contour(contour, frame)
            if matched_name:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output, matched_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
                cv2.imshow('Symbol Detection', output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            self.picam2.stop()

if __name__ == "__main__":
    detector = SymbolDetector()
    detector.run()