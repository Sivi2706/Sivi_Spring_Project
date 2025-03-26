import numpy as np
import cv2
from picamera2 import Picamera2

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

def detect_shapes_and_arrows(frame):
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        # Filter contours by area to reduce noise
        if cv2.contourArea(c) > 500:  # Increased threshold
            # Approximate the contour
            epsilon = 0.04 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            
            # Determine shape based on vertex count with more precision
            x, y, w, h = cv2.boundingRect(approx)
            shape = None
            
            # Enhanced shape recognition
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                # Check aspect ratio for rectangles/squares
                aspect_ratio = w / float(h)
                shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            elif 7 <= len(approx) <= 10:
                shape = "Circle"
            
            # Arrow detection
            if shape:
                # ROI for detailed arrow analysis
                roi = gray[y:y+h, x:x+w]
                roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
                roi_edges = cv2.Canny(roi_blur, 50, 150)
                
                # Morphological operations to improve edge detection
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                roi_closed = cv2.morphologyEx(roi_edges, cv2.MORPH_CLOSE, kernel)
                
                # Find contours in ROI
                roi_cnts, _ = cv2.findContours(roi_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                arrow_type = None
                for ac in roi_cnts:
                    if cv2.contourArea(ac) > 300:
                        # Arrow detection logic
                        epsilon2 = 0.02 * cv2.arcLength(ac, True)
                        approx2 = cv2.approxPolyDP(ac, epsilon2, True)
                        
                        # Determine arrow direction
                        if len(approx2) >= 5:
                            # Compute the moments to find centroid
                            M = cv2.moments(ac)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # Directional analysis
                                if cx < w/3:
                                    arrow_type = "Left Arrow"
                                elif cx > 2*w/3:
                                    arrow_type = "Right Arrow"
                                elif cy < h/3:
                                    arrow_type = "Up Arrow"
                                elif cy > 2*h/3:
                                    arrow_type = "Down Arrow"
                
                # Visualization
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Final label
                label = f"{shape} {arrow_type}" if arrow_type else shape
                cv2.putText(frame, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# Main execution
def main():
    # Initialize camera
    picam2 = initialize_camera()
    if picam2 is None:
        print("Exiting program. Camera could not be initialized.")
        return

    try:
        # Main processing loop
        while True:
            # Capture frame from the camera
            frame = picam2.capture_array()

            # Flip the frame vertically (optional, depending on camera orientation)
            frame = cv2.flip(frame, -1)

            # Detect shapes and arrows
            output_frame = detect_shapes_and_arrows(frame)

            # Display the frame
            cv2.imshow("Camera Feed", output_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        picam2.stop()

# Run the main function
if __name__ == "__main__":
    main()