import numpy as np
import cv2
import os
import pickle
from picamera import PiCamera
from time import sleep

class SymbolDetector:
    def __init__(self):
        self.reference_symbols = []
        self.min_contour_area = 500
        self.match_threshold = 0.2
        self.color_threshold = 75
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)

    def capture_image(self):
        self.camera.start_preview()
        sleep(2)  # Allow camera to adjust exposure
        self.camera.capture('/tmp/captured.jpg')
        self.camera.stop_preview()
        return cv2.imread('/tmp/captured.jpg')

    def process_reference_images(self, folder_path="Symbol-images"):
        if not os.path.exists(folder_path):
            print(f"Reference folder {folder_path} not found!")
            return False

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith('.png'):
                        image_path = os.path.join(subfolder_path, filename)
                        frame = cv2.imread(image_path)
                        
                        if frame is not None:
                            symbol_name = subfolder
                            segmented_image = self.fuzzy_color_segmentation(frame)
                            edges, contours = self.detect_edges(segmented_image)
                            feature_vector = self.encode_features(frame)
                            
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                self.reference_symbols.append({
                                    'name': symbol_name,
                                    'contour': largest_contour,
                                    'features': feature_vector
                                })
                                print(f"Loaded reference: {symbol_name}")
        
        if not self.reference_symbols:
            print("No valid reference images found!")
            return False
        
        self.save_references()
        return True

    def fuzzy_color_segmentation(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'blue': [(100, 50, 50), (140, 255, 255)],
            'yellow': [(20, 100, 100), (30, 255, 255)]
        }
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (lower, upper) in color_ranges.values():
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def detect_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return edges, [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

    def encode_features(self, frame):
        resized = cv2.resize(frame, (32, 32))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return np.concatenate((h.flatten(), s.flatten(), v.flatten()))

    def match_contour(self, contour, frame):
        detected_features = self.encode_features(frame)
        for ref in self.reference_symbols:
            if 'features' not in ref:
                continue
            shape_match = cv2.matchShapes(ref['contour'], contour, cv2.CONTOURS_MATCH_I2, 0)
            feature_distance = np.linalg.norm(detected_features - ref['features'])
            if shape_match < self.match_threshold and feature_distance < self.color_threshold:
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
        segmented_frame = self.fuzzy_color_segmentation(frame)
        edges, contours = self.detect_edges(segmented_frame)
        edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        output = cv2.addWeighted(frame, 0.7, edge_display, 0.3, 0)
        for contour in contours:
            matched_name = self.match_contour(contour, frame)
            if matched_name:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y, x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, matched_name, (x, y - 10), 
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
                frame = self.capture_image()
                output = self.process_frame(frame)
                cv2.imshow('Symbol Detection', output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            self.camera.close()

if __name__ == "__main__":
    detector = SymbolDetector()
    detector.run()
