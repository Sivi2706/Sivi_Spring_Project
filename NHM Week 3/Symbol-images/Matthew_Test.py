import cv2
import numpy as np
import os
from picamera2 import Picamera2
from collections import deque
from sklearn import svm

# Initialize Raspberry Pi Camera with enhanced error handling
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        print("Ensure the camera is connected properly. Retrying with fallback...")
        return None

# Load and preprocess reference images with enhanced ORB features
def load_reference_images():
    reference_images = {}
    orb = cv2.ORB_create(nfeatures=1500)  # Increased for better feature capture
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(script_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(script_dir, filename)
            img = cv2.imread(img_path, 0)
            if img is None:
                print(f"Failed to load reference image: {filename}")
                continue
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                reference_images[filename] = (keypoints, descriptors, img)
            else:
                print(f"Warning: No features detected in {filename}")
    print("Recommendation: Use high-quality images with varied lighting and angles.")
    print("Ensure reference images include clear directional markers.")
    return reference_images, orb

# Train SVM classifier with directional features
def train_svm_classifier():
    # Training data with red ratio, verticality, horizontality
    features = np.array([
        [0.1, 0.9, 0.0],  # Upward arrow: low red, high verticality
        [0.1, 0.9, 0.0],  # Upward arrow
        [0.1, 0.2, 1.0],  # Leftward arrow: low red, high horizontality
        [0.1, 0.2, 1.0],  # Leftward arrow
        [0.8, 0.3, 0.2],  # Stop sign: high red, moderate features
        [0.7, 0.2, 0.3]   # Stop sign
    ])
    labels = np.array([0, 0, 1, 1, 2, 2])  # 0: up arrow, 1: left arrow, 2: stop sign
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(features, labels)
    return clf

# Validate stop sign characteristics
def validate_stop_sign(image, keypoints):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    red_ratio = np.sum(mask) / (mask.size * 255)
    confidence = 0.5 if red_ratio > 0.2 else 0.3
    return confidence

# Validate arrow orientation using PCA
def validate_orientation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            _, (eigen_vecs, _) = cv2.PCACompute2(cnt.reshape(-1, 2), np.array([]))
            angle = np.arctan2(eigen_vecs[0, 1], eigen_vecs[0, 0]) * 180 / np.pi
            if abs(angle) < 45 or abs(angle - 180) < 45:
                return "up", angle
            elif abs(angle - 90) < 45 or abs(angle - 270) < 45:
                return "left" if angle > 0 else "right", angle
    return None, 0.0

# Enhanced ORB feature matching with orientation validation
def match_image(frame, orb, reference_images, svm_clf):
    frame_rgb = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    frame_keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        print("No descriptors detected in frame.")
        return None, None, gray, None, 0, None, None, frame_keypoints, 0.0
    matches_dict = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_matches = None
    best_ref_keypoints = None
    supposed_match = None
    supposed_match_count = 0
    supposed_matches_list = []
    supposed_keypoints = None
    match_threshold = 20  # Increased for stricter matching
    for name, (ref_keypoints, ref_descriptors, ref_img) in reference_images.items():
        matches = bf.match(descriptors, ref_descriptors)
        matches_dict[name] = len(matches)
        if len(matches) > supposed_match_count:
            supposed_match = name
            supposed_match_count = len(matches)
            supposed_matches_list = matches
            supposed_keypoints = ref_keypoints
        if len(matches) > match_threshold:
            matches = sorted(matches, key=lambda x: x.distance)
            if not best_match or len(matches) > matches_dict[best_match]:
                best_match = name
                best_matches = matches[:10]
                best_ref_keypoints = ref_keypoints
    print("\nMatch Results:")
    for name, count in sorted(matches_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {count} matches")
    confidence = 0.0
    if supposed_match:
        # Directional validation for arrows
        if "arrow" in supposed_match.lower():
            direction, detected_angle = validate_orientation(frame_rgb)
            ref_direction = "left" if "left" in supposed_match.lower() else "up" if "up" in supposed_match.lower() else None
            if direction and ref_direction and direction != ref_direction:
                confidence = 0.0
                print(f"Direction mismatch: Detected {direction}, Expected {ref_direction}")
            else:
                confidence = 0.5
            print(f"Detected Orientation: {direction or 'Unknown'} at {detected_angle:.1f}°")
        elif "stop" in supposed_match.lower():
            confidence = validate_stop_sign(frame_rgb, frame_keypoints)
            print(f"Stop sign validation confidence: {confidence:.2f}")
        # SVM classification with directional features
        red_ratio = np.sum(cv2.inRange(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV), 
                                       np.array([0, 120, 70]), np.array([10, 255, 255]))) / (640 * 480 * 255)
        contours, _ = cv2.findContours(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        verticality = 1.0 if contours and cv2.boundingRect(contours[0])[3] > cv2.boundingRect(contours[0])[2] else 0.2
        horizontality = 1.0 if contours and cv2.boundingRect(contours[0])[2] > cv2.boundingRect(contours[0])[3] else 0.2
        features = np.array([[red_ratio, verticality, horizontality]])
        svm_pred = svm_clf.predict_proba(features)[0]
        if "arrow" in supposed_match.lower():
            confidence = svm_pred[1 if "left" in supposed_match.lower() else 0]
            print(f"SVM arrow probability: {confidence:.2f}")
        elif "stop" in supposed_match.lower():
            confidence = max(confidence, svm_pred[2])
            print(f"SVM stop sign probability: {svm_pred[2]:.2f}")
    if best_match and best_matches:
        src_pts = np.float32([frame_keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
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
            if "stop" in best_match.lower():
                rotation_angle = -angle
                rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_angle, 1.0)
                frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_keypoints, descriptors = orb.detectAndCompute(gray, None)
                print(f"Normalized angle to 0° from {angle:.1f}°")
            return best_match, (center, tip, angle), gray, supposed_match, supposed_match_count, supposed_matches_list, supposed_keypoints, frame_keypoints, confidence
    return None, None, gray, supposed_match, supposed_match_count, supposed_matches_list, supposed_keypoints, frame_keypoints, confidence

# Shape detection with enhanced preprocessing
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
        sides = len(approx)
        if sides == 3:
            shape_detected = "Triangle"
        elif sides == 4:
            aspect_ratio = float(w) / h
            shape_detected = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif sides == 5:
            shape_detected = "Pentagon"
        elif sides >= 7:
            shape_detected = "Circle" if cv2.contourArea(contour) / (np.pi * (w/2)**2) > 0.8 else "Unknown"
        cv2.putText(frame, shape_detected or "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        break
    return shape_detected, blurred, thresh, edges

# Process frame with multi-frame consistency
def process_frame(frame, prev_detections, orb, reference_images, svm_clf, max_len=5, is_live_feed=False):
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    match_name, orientation, gray, supposed_match, supposed_match_count, supposed_matches_list, supposed_keypoints, frame_keypoints, confidence = match_image(frame, orb, reference_images, svm_clf)
    shape_detected, blurred, thresh, edges = (None, gray, None, None)
    if not match_name:
        shape_detected, blurred, thresh, edges = detect_shapes(frame.copy(), gray)
    current_detection = (match_name or shape_detected, confidence)
    prev_detections.append(current_detection)
    if len(prev_detections) > max_len:
        prev_detections.popleft()
    valid_detections = [(d, c) for d, c in prev_detections if d is not None]
    if valid_detections:
        detected_name = max(set([d for d, _ in valid_detections]), key=[d for d, _ in valid_detections].count)
        avg_confidence = sum(c for _, c in valid_detections) / len(valid_detections)
        if avg_confidence < 0.6 and len(set([d for d, _ in valid_detections])) > 1:
            detected_name = None
        label = f"Detected: {detected_name or 'None'} (Confidence: {avg_confidence:.2f})"
    else:
        detected_name = None
        label = "Detected: None"
    if match_name and orientation:
        center, tip, angle = orientation
        cv2.arrowedLine(frame, (int(center[0]), int(center[1])), 
                        (int(tip[0]), int(tip[1])), (0, 255, 0), 2, tipLength=0.3)
        print(f"Orientation Detected: {angle:.1f} degrees")
    if is_live_feed and supposed_match and supposed_match_count > 0:
        ref_img = reference_images[supposed_match][2]
        match_img = cv2.drawMatches(gray, frame_keypoints, ref_img, supposed_keypoints, supposed_matches_list[:30], None, flags=2)
        cv2.putText(match_img, f"Supposed Match: {supposed_match}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(match_img, f"Matches: {supposed_match_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(match_img, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if "arrow" in supposed_match.lower():
            direction, _ = validate_orientation(frame)
            cv2.putText(match_img, f"Direction: {direction or 'Unknown'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Matching Debug", match_img)
    cv2.rectangle(frame, (5, 5), (400, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"Stabilized Detection Result: {detected_name or 'None'}")
    return frame, detected_name, gray, blurred, thresh, edges

# Process sample images
def process_sample_images(reference_images, orb, svm_clf):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prev_detections = deque()
    image_files = [f for f in os.listdir(script_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print("\nAvailable sample images:")
    for i, filename in enumerate(image_files, 1):
        print(f"{i}. {filename}")
    while True:
        choice = input("\nEnter image number to process (or 'q' to return): ")
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
                output_frame, detected_name, gray, blurred, thresh, edges = process_frame(frame, prev_detections, orb, reference_images, svm_clf)
                cv2.imshow("Processed Image", output_frame)
                cv2.imshow("Grayscale", gray)
                cv2.imshow("Blurred", blurred)
                cv2.imshow("Threshold", thresh)
                cv2.imshow("Edges", edges)
                print(f"Processing {image_files[idx]}: {detected_name or 'None'}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Invalid image number.")
        except ValueError:
            print("Please enter a valid number or 'q'.")

# Live feed processing
def process_live_feed(picam2, reference_images, orb, svm_clf, live_feed_enabled):
    prev_detections = deque()
    while live_feed_enabled[0]:
        frame = picam2.capture_array()
        output_frame, detected_name, gray, blurred, thresh, edges = process_frame(frame, prev_detections, orb, reference_images, svm_clf, is_live_feed=True)
        cv2.imshow("Live Feed", output_frame)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Edges", edges)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            live_feed_enabled[0] = False
            cv2.destroyAllWindows()
            break

# Main menu
def main_menu():
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return
    reference_images, orb = load_reference_images()
    svm_clf = train_svm_classifier()
    live_feed_enabled = [False]
    try:
        while True:
            print("\nMain Menu:")
            print("1. Process sample images")
            print("2. Toggle live feed")
            print("3. Exit")
            choice = input("Enter your choice (1-3): ")
            if choice == '1':
                process_sample_images(reference_images, orb, svm_clf)
            elif choice == '2':
                live_feed_enabled[0] = not live_feed_enabled[0]
                if live_feed_enabled[0]:
                    print("Starting live feed. Press 'q' to stop.")
                    process_live_feed(picam2, reference_images, orb, svm_clf, live_feed_enabled)
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