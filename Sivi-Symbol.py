import numpy as np
import cv2
import os

def load_symbols_from_folder(folder_path):
    """
    Load all images from the specified folder
    """
    symbols = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Use filename (without extension) as the key
                symbol_name = os.path.splitext(filename)[0]
                symbols[symbol_name] = img
    return symbols

def preprocess_symbol(image):
    """
    Preprocess symbol image for better comparison
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to a standard size
    resized = cv2.resize(gray, (100, 100))
    
    # Apply thresholding
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def match_symbol(test_image, symbol_library):
    """
    Match the input image against the symbol library
    """
    # Preprocess test image
    processed_test = preprocess_symbol(test_image)
    
    best_match = None
    best_score = float('inf')
    
    # Compare with each symbol in the library
    for symbol_name, symbol_img in symbol_library.items():
        processed_symbol = preprocess_symbol(symbol_img)
        
        # Calculate difference
        diff = cv2.absdiff(processed_test, processed_symbol)
        score = np.sum(diff)
        
        # Find the symbol with the lowest difference score
        if score < best_score:
            best_score = score
            best_match = symbol_name
    
    return best_match, best_score

def detect_shapes_and_symbols(frame, symbol_library):
    """
    Detect shapes and match symbols
    """
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        if cv2.contourArea(c) > 1000:  # Ignore small contours
            # Approximate the contour
            epsilon = 0.04 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            
            x, y, w, h = cv2.boundingRect(approx)
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Determine basic shape
            shape = "Unknown"
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Quadrilateral"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 6:
                shape = "Hexagon"
            else:
                shape = "Circle/Irregular"
            
            # Try to match symbol
            symbol_match, match_score = match_symbol(roi, symbol_library)
            
            # Draw contour and label
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Combine shape and symbol information
            label = f"{shape}: {symbol_match}" if symbol_match else shape
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame

def main():
    # Load symbols from the folder
    symbol_folder = "Symbol-images"
    symbol_library = load_symbols_from_folder(symbol_folder)
    
    # Process each symbol individually
    print("Loaded Symbols:")
    for name, img in symbol_library.items():
        print(f"- {name}")
        cv2.imshow(name, img)
    
    # Process test images or start camera feed
    test_images = [f for f in os.listdir(symbol_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for image_name in test_images:
        # Read the image
        image_path = os.path.join(symbol_folder, image_name)
        frame = cv2.imread(image_path)
        
        if frame is not None:
            # Process the image
            output = detect_shapes_and_symbols(frame, symbol_library)
            
            # Show the processed image
            cv2.imshow(f"Processed: {image_name}", output)
    
    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()