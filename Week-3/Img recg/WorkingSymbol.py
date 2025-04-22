#!/usr/bin/env python

"""
WorkingSymbol.py: A Python script to recognize shapes in .png images using OpenCV.
This script assumes it is located in the 'Img rcg' folder with the .png images.
It identifies basic shapes (triangle, rectangle, pentagon, hexagon, circle) using contour detection.
"""

import cv2
import glob
import os

# Get list of all .png files in the current directory
image_files = glob.glob('*.png')

for file in image_files:
    # Load the image in grayscale
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not load image: {file}")
        continue

    # Apply binary thresholding
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        num_sides = len(approx)

        if num_sides == 3:
            shape = "triangle"
        elif num_sides == 4:
            shape = "rectangle"
        elif num_sides == 5:
            shape = "pentagon"
        elif num_sides == 6:
            shape = "hexagon"
        else:
            # Check for circle using circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * 3.14159 * area / (perimeter * perimeter)
                if circularity > 0.8:
                    shape = "circle"
                else:
                    shape = "other"
            else:
                shape = "other"
    else:
        shape = "no shape found"

    # Print the result
    print(f"{os.path.basename(file)}: {shape}")