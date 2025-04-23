import cv2
import numpy as np

# Define color ranges in HSV for red, green, and blue
COLOR_RANGES = {
    'red':    ([0, 100, 100], [10, 255, 255]),
    'green':  ([45, 50, 50], [85, 255, 255]),
    'blue':   ([100, 150, 0], [140, 255, 255]),
}

def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        return 'triangle'
    elif len(approx) == 4:
        return 'square'
    elif len(approx) > 5:
        return 'circle'
    return 'unidentified'

def main():
    cap = cv2.VideoCapture(0)  # Adjust the index depending on your camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        for color_name, (lower, upper) in COLOR_RANGES.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # Filter out small noise
                    shape = detect_shape(cnt)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.drawContours(frame, [cnt], -1, (255, 255, 255), 2)
                        cv2.putText(frame, f"{color_name} {shape}", (cx - 50, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Shape and Color Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
