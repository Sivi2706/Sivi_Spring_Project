#!/usr/bin/env python

"""
RPi_ShapeRecognizer_Fixed.py: Fixed path handling for template loading
"""

import cv2
import glob
import os
import numpy as np

# Configuration
MATCH_THRESHOLD = 0.75
MIN_CONTOUR_AREA = 500
DISPLAY_SCALE = 0.7
USE_PICAMERA = True
RESOLUTION = (640, 480)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shape categories (subdirectories relative to script location)
folders = [
    "3 quaters of a circle",
    "Arrow down",
    "Arrow up",
    "Blue rectangle",
    "Facial recompulsion",
    "Hexagon",
    "Left arrow",
    "Measure Distance",
    "Pentagon",
    "Red circle",
    "Right arrow",
    "Stop sign",
    "Traffic light stop",
    "Triangle"
]

def load_templates():
    """Load template images with proper path handling"""
    templates = {}
    template_count = 0
    
    print("Loading template database from:", SCRIPT_DIR)
    
    for folder in folders:
        # Create full path to the folder
        folder_path = os.path.join(SCRIPT_DIR, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder does not exist - {folder_path}")
            continue
            
        # Search for PNG files (case insensitive)
        image_files = glob.glob(os.path.join(folder_path, "*.png")) + \
                     glob.glob(os.path.join(folder_path, "*.PNG"))
        
        if not image_files:
            print(f"Warning: No PNG files found in {folder_path}")
            continue
            
        templates[folder] = []
        
        for file in image_files:
            try:
                print(f"Loading template: {file}")
                template = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[folder].append(template)
                    template_count += 1
                else:
                    print(f"Warning: Could not load image (may be corrupt) - {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    print(f"\nSuccessfully loaded {template_count} templates across {len(templates)} categories")
    
    # Print loaded categories
    if templates:
        print("Loaded categories:")
        for category in templates.keys():
            print(f"- {category} ({len(templates[category])} templates)")
    else:
        print("Warning: No templates were loaded!")
    
    return templates

# ... [rest of the code remains the same as previous version] ...

if __name__ == "__main__":
    templates = load_templates()
    
    if not templates:
        print("\nERROR: No templates loaded. Please check:")
        print("1. The folder names in the script match your actual folders")
        print("2. The folders are in the same directory as the script")
        print("3. The folders contain PNG images")
        print(f"\nCurrent script directory: {SCRIPT_DIR}")
        print("Contents of script directory:")
        print(os.listdir(SCRIPT_DIR))
    else:
        main()