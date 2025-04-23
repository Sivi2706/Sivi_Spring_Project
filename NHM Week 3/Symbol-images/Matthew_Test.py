import cv2
import numpy as np
import os
from picamera2 import Picamera2
from collections import deque

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

# Load and preprocess reference images using ORB
def load_reference_images():
    reference_images = {}
    orb = cv2.ORB_create()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(script_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(script_dir, filename)
            img = cv2.imread(img_path, 0)
            if img is None:
                print(f"Failed to load reference image: {filename}")
                continue
            # Resize reference image to match live feed resolution for better matching
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                reference_images[filename] = (keypoints, descriptors, img)
            else:
                print(f"Warning: No features detected in {filename}")
    return reference_images, orb

# ORB Feature Matching with Orientation Detection
def match_image(frame, orb, reference_images):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply slight blur to reduce noise in live feed
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        print("No descriptors detected in frame.")
        return None, None, gray
    matches_dict = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_matches = None
    best_ref_keypoints = None
    # Lower the threshold for live feed to increase sensitivity
    match_threshold = 15  # Reduced from 30
    for name, (ref_keypoints, ref_descriptors, ref_img) in reference_images.items():
        matches = bf.match(descriptors, ref_descriptors)
        matches_dict[name] = len(matches)
        if len(matches) > match_threshold:
            matches = sorted(matches, key=lambda x: x.distance)
            if not best_match or len(matches) > matches_dict[best_match]:
                best_match = name
                best_matches = matches[:10]
                best_ref_keypoints = ref_keypoints
    print("\nMatch Results:")
    for name, count in sorted(matches_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {count} matches")
    if best_match:
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([best_ref_keypoints[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is not None:
            h, w = reference_images[best_match][2].shape
            ref_center = (w // 2, h // 2)
            ref_tip = (w // 2, 0)
            center = cv2.perspectiveTransform(np.array([[ref_center]], dtype=np.float32), M)[0][0]
            tip = cv2.perspectiveTransform(np.array([[ref_tip]], dtype=np.float32), M)[0][0]
            dx = tip[0] - center[0]
            dy = tip[1] - center[1]
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            return best_match, (center, tip, angle), gray
    return None, None, gray

# Shape Detection Function
def detect_shapes(frame, gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 30, 200)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_detected = None
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        defect_count = 0
        max_defect = 0
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is not None and len(hull) > 3 and len(contour) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    defect_count = len(defects)
                    max_defect = np.max(defects[:, 0, 3]) if defect_count > 0 else 0
        except cv2.error as e:
            print(f"Convexity defect calculation skipped: {e}")
        print(f"Debug - Sides: {len(approx)}, Circularity: {circularity:.3f}, Defects: {defect_count}, Max Defect: {max_defect}")
        sides = len(approx)
        if sides == 3:
            shape_detected = "Triangle"
        elif sides == 4:
            aspect_ratio = float(w) / h
            shape_detected = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif sides == 5:
            shape_detected = "Pentagon"
        elif sides == 6:
            shape_detected = "Hexagon"
        else:
            if circularity > 0.8:
                shape_detected = "Circle"
            elif 0.5 <= circularity <= 0.8 and defect_count > 0 and max_defect > 300:
                shape_detected = "Pac-Man"
            else:
                shape_detected = "Unknown"
        cv2.putText(frame, shape_detected, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 2)
        break
    return shape_detected, blurred, thresh, edges

# Process Frame (for both live and sample images)
def process_frame(frame, prev_detections, orb, reference_images, max_len=5):
    # Ensure frame is in BGR format
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    # Resize frame to match reference image resolution
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    match_name, orientation, gray = match_image(frame, orb, reference_images)
    shape_detected, blurred, thresh, edges = (None, gray, None, None)
    if not match_name:
        shape_detected, blurred, thresh, edges = detect_shapes(frame.copy(), gray)
    current_detection = match_name if match_name else shape_detected
    prev_detections.append(current_detection)
    if len(prev_detections) > max_len:
        prev_detections.popleft()
    valid_detections = [d for d in prev_detections if d is not None]
    if valid_detections:
        detected_name = max(set(valid_detections), key=valid_detections.count)
        label = f"Detected: {detected_name}"
        if match_name and orientation:
            center, tip, angle = orientation
            cv2.arrowedLine(frame, (int(center[0]), int(center[1])), 
                           (int(tip[0]), int(tip[1])), (0, 255, 0), 2, tipLength=0.3)
            label += f" | Angle: {angle:.1f}°"
            print(f"Orientation Detected: {angle:.1f} degrees")
    else:
        detected_name = None
        label = "Detected: None"
    cv2.rectangle(frame, (5, 5), (400, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"Stabilized Detection Result: {detected_name if detected_name else 'None'}")
    return frame, detected_name, gray, blurred, thresh, edges

# Process Sample Images
def process_sample_images(reference_images, orb):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prev_detections = deque()
    print("\nAvailable sample images:")
    image_files = [f for f in os.listdir(script_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for i, filename in enumerate(image_files, 1):
        print(f"{i}. {filename}")
    while True:
        choice = input("\nEnter image number to process (or 'q' to return to main menu): ")
        if choice.lower() == 'q':
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                img_path = os.path.join(script_dir, image_files[idx])
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Failed to load image: {image_files[idx]}")
                    continue
                output_frame, detected_name, gray, blurred, thresh, edges = process_frame(frame, prev_detections, orb, reference_images)
                cv2.imshow("Processed Image", output_frame)
                if gray is not None:
                    cv2.imshow("Grayscale", gray)
                if blurred is not None:
                    cv2.imshow("Blurred", blurred)
                if thresh is not None:
                    cv2.imshow("Threshold", thresh)
                if edges is not None:
                    cv2.imshow("Edges", edges)
                print(f"Processing {image_files[idx]}: {detected_name if detected_name else 'None'}")
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
            else:
                print("Invalid image number.")
        except ValueError:
            print("Please enter a valid number or 'q'.")

# Live Feed Processing
def process_live_feed(picam2, reference_images, orb, live_feed_enabled):
    prev_detections = deque()
    while live_feed_enabled[0]:
        frame = picam2.capture_array()
        output_frame, detected_name, gray, blurred, thresh, edges = process_frame(frame, prev_detections, orb, reference_images)
        cv2.imshow("Live Feed", output_frame)
        if gray is not None:
            cv2.imshow("Grayscale", gray)
        if blurred is not None:
            cv2.imshow("Blurred", blurred)
        if thresh is not None:
            cv2.imshow("Threshold", thresh)
        if edges is not None:
            cv2.imshow("Edges", edges)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            live_feed_enabled[0] = False
            cv2.destroyAllWindows()
            break

# Main Menu
def main_menu():
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return
    reference_images, orb = load_reference_images()
    live_feed_enabled = [False]
    try:
        while True:
            print("\nMain Menu:")
            print("1. Process sample images")
            print("2. Toggle live feed")
            print("3. Exit")
            choice = input("Enter your choice (1-3): ")
            if choice == '1':
                process_sample_images(reference_images, orb)
            elif choice == '2':
                live_feed_enabled[0] = not live_feed_enabled[0]
                if live_feed_enabled[0]:
                    print("Starting live feed. Press 'q' to stop.")
                    process_live_feed(picam2, reference_images, orb, live_feed_enabled)
                else:
                    print("Live feed stopped.")
            elif choice == '3':
                break
            else:
                print("Invalid choice. Please try again.")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("Resources released")

if __name__ == "__main__":
    main_menu()