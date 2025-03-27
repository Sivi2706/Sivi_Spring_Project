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

picam2 = initialize_camera()
if picam2 is None:
    print("Exiting program. Camera could not be initialized.")
    exit()

# Load and preprocess reference images using ORB
reference_images = {}
orb = cv2.ORB_create()

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
for filename in os.listdir(script_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(script_dir, filename)
        img = cv2.imread(img_path, 0)  # Load in grayscale
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            reference_images[filename] = (keypoints, descriptors, img)
        else:
            print(f"Warning: No features detected in {filename}")

# ORB Feature Matching with Orientation Detection
def match_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        return None, None  # No features detected

    matches_dict = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_matches = None
    best_ref_keypoints = None

    for name, (ref_keypoints, ref_descriptors, ref_img) in reference_images.items():
        matches = bf.match(descriptors, ref_descriptors)
        matches_dict[name] = len(matches)
        if len(matches) > 30:  # Threshold for a valid match
            matches = sorted(matches, key=lambda x: x.distance)
            if not best_match or len(matches) > matches_dict[best_match]:
                best_match = name
                best_matches = matches[:10]  # Use top 10 matches for orientation
                best_ref_keypoints = ref_keypoints

    # Print all match counts for debugging
    print("\nMatch Results:")
    for name, count in sorted(matches_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {count} matches")

    if best_match:
        # Calculate orientation using matched keypoints
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([best_ref_keypoints[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        
        # Find homography to align the images
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is not None:
            # Determine arrow orientation (assuming arrow points from center to tip)
            h, w = reference_images[best_match][2].shape
            ref_center = (w // 2, h // 2)
            ref_tip = (w // 2, 0)  # Assuming arrow points up in reference image
            
            # Transform reference points to frame coordinates
            center = cv2.perspectiveTransform(np.array([[ref_center]], dtype=np.float32), M)[0][0]
            tip = cv2.perspectiveTransform(np.array([[ref_tip]], dtype=np.float32), M)[0][0]
            
            # Calculate angle (in degrees) from center to tip
            dx = tip[0] - center[0]
            dy = tip[1] - center[1]
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            
            return best_match, (center, tip, angle)
    return None, None

# Improved Shape Detection Function
def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 30, 200)
    
    # Apply morphological operations to clean up edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_detected = None
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Skip small contours
            continue
            
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        
        # Draw contour and bounding box
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate convexity defects safely
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
            defects = None

        # Debug output
        print(f"Debug - Sides: {len(approx)}, Circularity: {circularity:.3f}, Defects: {defect_count}, Max Defect: {max_defect}")

        # Determine shape based on properties
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
            # Differentiate between Circle and Pac-Man
            if circularity > 0.8:
                shape_detected = "Circle"
            elif 0.5 <= circularity <= 0.8 and defect_count > 0 and max_defect > 300:
                shape_detected = "Pac-Man"
            else:
                shape_detected = "Unknown"

        # Label the shape on the frame
        cv2.putText(frame, shape_detected, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 2)
        break  # Only process the largest contour
    
    return shape_detected

# Image Detection with Temporal Smoothing and Orientation Display
def detect_images(frame, prev_detections, max_len=5):
    # First, try to match the frame against reference images
    match_name, orientation = match_image(frame)

    # If no image match is found, detect shapes as a fallback
    shape_detected = None
    if not match_name:
        shape_detected = detect_shapes(frame)

    # Add the current detection to the deque (priority to image match)
    current_detection = match_name if match_name else shape_detected
    prev_detections.append(current_detection)
    if len(prev_detections) > max_len:
        prev_detections.popleft()

    # Determine the most common detection in recent frames
    valid_detections = [d for d in prev_detections if d is not None]
    if valid_detections:
        detected_name = max(set(valid_detections), key=valid_detections.count)
        label = f"Detected: {detected_name}"
        if match_name and orientation:  # Only show orientation for image matches
            center, tip, angle = orientation
            # Draw arrow orientation
            cv2.arrowedLine(frame, (int(center[0]), int(center[1])), 
                           (int(tip[0]), int(tip[1])), (0, 255, 0), 2, tipLength=0.3)
            label += f" | Angle: {angle:.1f}Â°"
            print(f"Orientation Detected: {angle:.1f} degrees")
    else:
        detected_name = None
        label = "Detected: None"

    # Display the label on the frame
    cv2.rectangle(frame, (5, 5), (400, 40), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"Stabilized Detection Result: {detected_name if detected_name else 'None'}")

    return frame, detected_name

# Main loop
prev_detections = deque()  # Store recent detections for smoothing
while True:
    frame = picam2.capture_array()

    # Flip frame vertically (optional)
    # frame = cv2.flip(frame, -1)

    # Convert frame to BGR format if needed
    if len(frame.shape) == 2:  # If grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # If RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Detect images and shapes
    output_frame, current_detection = detect_images(frame, prev_detections)

    # Show processed frames
    cv2.imshow("Image and Shape Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()