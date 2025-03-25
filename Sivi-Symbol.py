import numpy as np
import cv2
import os
import pickle

class ColorEdgeDetector:
    def __init__(self):
        self.reference_data = []
        self.min_contour_area = 500
        self.match_threshold = 0.85  # Shape matching threshold

    def detect_color_edges(self, frame):
        """Highlight edges of color contrasts in the image with bright outlines"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = [
            ((0, 0, 200), (180, 30, 255), "White", (255, 255, 255)),
            ((90, 50, 50), (130, 255, 255), "Blue", (255, 0, 0)),
            ((0, 100, 100), (10, 255, 255), "Red", (0, 0, 255)),
            ((160, 100, 100), (180, 255, 255), "Red", (0, 0, 255)),
            ((40, 50, 50), (80, 255, 255), "Green", (0, 255, 0))
        ]
        
        edge_display = np.zeros_like(frame)
        detected_shapes = []
        
        for lower, upper, color_name, color_bgr in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            edges = cv2.Canny(mask, 30, 100)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find and store contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_contour_area:
                    # Store contour information
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    detected_shapes.append((color_name, approx, color_bgr))
            
            # Draw edges
            colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            colored_edges[np.where((colored_edges == [255, 255, 255]).all(axis=2))] = color_bgr
            edge_display = cv2.add(edge_display, colored_edges)
        
        output = cv2.addWeighted(frame, 0.5, edge_display, 0.8, 0)
        return output, detected_shapes

    def save_reference(self, image_path):
        """Process reference image and store its parameters"""
        frame = cv2.imread(image_path)
        if frame is not None:
            _, shapes = self.detect_color_edges(frame)
            if shapes:
                # Get the image name without extension
                name = os.path.splitext(os.path.basename(image_path))[0]
                # Store the first significant shape found
                color_name, approx, color_bgr = shapes[0]
                self.reference_data.append({
                    'name': name,
                    'color': color_name,
                    'contour': approx,
                    'color_bgr': color_bgr
                })
                print(f"Saved reference for {name}")
                return True
        return False

    def match_shapes(self, contour):
        """Match detected contour against reference shapes"""
        for ref in self.reference_data:
            match = cv2.matchShapes(ref['contour'], contour, cv2.CONTOURS_MATCH_I2, 0)
            if match < self.match_threshold:
                return ref['name']
        return None

    def process_live_feed(self):
        """Process live video feed and detect known patterns"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output, detected_shapes = self.detect_color_edges(frame)
            
            # Check for matches with reference shapes
            for color_name, approx, color_bgr in detected_shapes:
                matched_name = self.match_shapes(approx)
                if matched_name:
                    # Draw bounding box and label
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(output, (x, y), (x+w, y+h), color_bgr, 2)
                    cv2.putText(output, matched_name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
            
            cv2.imshow('Live Detection', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def save_references_to_file(self, filename='reference_data.pkl'):
        """Save reference data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.reference_data, f)
        print(f"Saved references to {filename}")

    def load_references_from_file(self, filename='reference_data.pkl'):
        """Load reference data from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.reference_data = pickle.load(f)
            print(f"Loaded {len(self.reference_data)} references from {filename}")
            return True
        return False

def main():
    detector = ColorEdgeDetector()
    
    # First mode: Process reference images
    image_folder = "Symbol-images"
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            print("Processing reference images...")
            for image_name in image_files:
                image_path = os.path.join(image_folder, image_name)
                detector.save_reference(image_path)
            
            # Save references to file
            detector.save_references_to_file()
        else:
            print(f"No images found in {image_folder}")
    else:
        print(f"Folder {image_folder} not found!")
    
    # Try to load references if we didn't just save them
    if not detector.reference_data:
        detector.load_references_from_file()
    
    # Second mode: Live detection
    if detector.reference_data:
        print("Starting live detection... Press 'q' to quit")
        detector.process_live_feed()
    else:
        print("No reference data available for detection")

if __name__ == "__main__":
    main()