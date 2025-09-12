#!/usr/bin/env python3
"""
Test script to validate OCR improvements locally
"""

import requests
from PIL import Image
import numpy as np
import cv2
from predict import Predictor
import tempfile
import os
from pathlib import Path

def download_test_image():
    """Download the test image"""
    url = "https://replicate.delivery/pbxt/NhBf4jJCYYl3TZYzGoWMweuFdz5rfsFug7a5m99CCpsNlkqU/rab.webp"
    
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webp') as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

def main():
    """Test the improved OCR"""
    print("Downloading test image...")
    image_path = download_test_image()
    
    try:
        print("Initializing predictor...")
        predictor = Predictor()
        predictor.setup()
        
        print("Running OCR with improved settings...")
        result = predictor.predict(
            image=Path(image_path),
            languages="en", 
            text_only=False,
            preprocessing=True,
            include_bboxes=True,
            min_confidence=0.25,
            include_polygons=False
        )
        
        print("\n" + "="*80)
        print("IMPROVED OCR RESULT:")
        print("="*80)
        print(result)
        print("="*80)
        
        # Also test with original settings for comparison
        print("\nRunning OCR with original settings for comparison...")
        result_original = predictor.predict(
            image=Path(image_path),
            languages="auto",  # Multiple languages like original
            text_only=False,
            preprocessing=True,
            include_bboxes=True,
            min_confidence=0.25,
            include_polygons=False
        )
        
        print("\n" + "="*80)
        print("ORIGINAL SETTINGS RESULT:")
        print("="*80)
        print(result_original)
        print("="*80)
        
    finally:
        # Clean up
        os.unlink(image_path)

if __name__ == "__main__":
    main()
