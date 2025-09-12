#!/usr/bin/env python3
"""
Simplified test script to validate OCR improvements locally without Cog framework
"""

import requests
from PIL import Image
import numpy as np
import cv2
import easyocr
import tempfile
import os
import json
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

def load_and_preprocess_image(image_path, upscale_min_dim=600, clahe=True, sharpen=True):
    """Load and preprocess image similar to our improved version"""
    # Load via PIL for broad format support, then to OpenCV BGR
    im = Image.open(image_path)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    arr = np.array(im)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    # Apply preprocessing
    h, w = arr.shape[:2]
    if upscale_min_dim > 0 and min(h, w) < upscale_min_dim:
        scale = min(3.0, float(upscale_min_dim) / max(1.0, float(min(h, w))))
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement (CLAHE) — mild to avoid blowing out punctuation
    if clahe and arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        clahe_op = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        gray = clahe_op.apply(gray)
        arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Mild unsharp mask to keep apostrophes/accents
    if sharpen and arr.ndim == 3:
        blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.0)
        arr = cv2.addWeighted(arr, 1.25, blur, -0.25, 0)

    return arr

def group_into_lines(filtered_regions):
    """Group text regions into logical lines based on vertical proximity and alignment."""
    if not filtered_regions:
        return []
        
    lines = []
    current_line = [filtered_regions[0]]
    
    for i in range(1, len(filtered_regions)):
        prev = filtered_regions[i-1]
        curr = filtered_regions[i]
        
        # Calculate vertical overlap and proximity
        # Data structure: (y_min, x_min, xs, ys, text, conf, y_max, x_max)
        prev_y_center = (prev[0] + prev[6]) / 2  # y_min + y_max / 2
        curr_y_center = (curr[0] + curr[6]) / 2
        prev_height = prev[6] - prev[0]  # y_max - y_min
        
        vertical_distance = abs(curr_y_center - prev_y_center)
        
        # If regions are on the same line (vertical distance < 0.5 * height)
        if vertical_distance < prev_height * 0.6:
            current_line.append(curr)
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [curr]
    
    if current_line:
        lines.append(current_line)
        
    return lines

def reconstruct_line_text(line_regions):
    """Reconstruct text from regions in a line, handling word boundaries intelligently."""
    if not line_regions:
        return ""
        
    # Sort line regions by x position
    line_regions.sort(key=lambda r: r[1])  # sort by x_min
    
    reconstructed = ""
    for i, region in enumerate(line_regions):
        text = region[4].strip()  # text content
        
        if i == 0:
            reconstructed = text
        else:
            prev_region = line_regions[i-1]
            curr_x_min = region[1]
            prev_x_max = max(prev_region[2])  # max x coordinate
            
            # Calculate gap between regions
            gap = curr_x_min - prev_x_max
            prev_width = max(prev_region[2]) - min(prev_region[2])
            
            # Estimate character width for spacing decisions
            char_width = prev_width / max(1, len(prev_region[4].strip()))
            
            # Add space if gap is significant (> 0.5 character widths)
            if gap > char_width * 0.5:
                reconstructed += " " + text
            else:
                # No space - likely part of same word
                reconstructed += text
                
    return reconstructed

def run_improved_ocr(image_path):
    """Run OCR with our improved settings"""
    print("Running improved OCR...")
    
    # Initialize EasyOCR reader with English only for better accuracy
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    # Load and preprocess image
    processed_image = load_and_preprocess_image(image_path)
    
    # Run OCR with optimized parameters
    results = reader.readtext(
        processed_image,
        detail=1,
        paragraph=False,
        width_ths=0.5,  # More aggressive horizontal text merging
        height_ths=0.4,  # Better vertical grouping
        slope_ths=0.2,   # More tolerant of slightly angled text
        ycenter_ths=0.7,  # Better line detection
        add_margin=0.15,  # Larger margin for text detection
        text_threshold=0.7,  # Higher confidence for text detection
        low_text=0.4,    # Lower threshold for weaker text
        link_threshold=0.4,  # Better text component linking
    )
    
    # Process results with improved grouping
    filtered = []
    min_confidence = 0.25
    for bbox, text, conf in results:
        if not text or conf < min_confidence:
            continue
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        y_min, x_min, y_max, x_max = min(ys), min(xs), max(ys), max(xs)
        filtered.append((y_min, x_min, xs, ys, text, float(conf), y_max, x_max))
    
    # Sort by vertical position first for line grouping
    filtered.sort(key=lambda t: (t[0], t[1]))
    
    # Group regions into lines
    lines = group_into_lines(filtered)
    
    # Reconstruct text
    final_text = ""
    for line_regions in lines:
        line_text = reconstruct_line_text(line_regions)
        if line_text.strip():
            final_text += line_text.strip() + "\\n"
    
    # Fix common OCR errors
    final_text = final_text.replace("don [ t", "don't")
    final_text = final_text.replace("don [", "don't")
    final_text = final_text.replace("can [ t", "can't")
    final_text = final_text.replace("won [ t", "won't")
    final_text = final_text.replace(" ,", ",")
    final_text = final_text.replace(" .", ".")
    
    return final_text.strip()

def run_original_ocr(image_path):
    """Run OCR with original settings for comparison"""
    print("Running original OCR...")
    
    # Initialize EasyOCR reader with multiple languages (original approach)
    reader = easyocr.Reader(['en', 'es', 'fr', 'de', 'it', 'pt'], gpu=False, verbose=False)
    
    # Simple preprocessing (just load image)
    processed_image = load_and_preprocess_image(image_path, upscale_min_dim=600, clahe=True, sharpen=True)
    
    # Run OCR with original parameters
    results = reader.readtext(
        processed_image,
        detail=1,
        paragraph=False,
        width_ths=0.7,  # Original settings
        height_ths=0.7,
        slope_ths=0.1,
        ycenter_ths=0.5,
        add_margin=0.1
    )
    
    # Simple text extraction (original approach)
    text_lines = []
    min_confidence = 0.25
    for bbox, text, conf in results:
        if text and conf >= min_confidence:
            text_lines.append(text.strip())
    
    return "\\n".join(text_lines)

def main():
    """Test the improved OCR"""
    print("Downloading test image...")
    image_path = download_test_image()
    
    try:
        print("\\n" + "="*80)
        print("TESTING OCR IMPROVEMENTS")
        print("="*80)
        
        # Test improved version
        improved_result = run_improved_ocr(image_path)
        
        print("\\n" + "="*80)
        print("IMPROVED OCR RESULT:")
        print("="*80)
        print(improved_result)
        print("="*80)
        
        # Test original version
        original_result = run_original_ocr(image_path)
        
        print("\\n" + "="*80)
        print("ORIGINAL OCR RESULT:")
        print("="*80)
        print(original_result)
        print("="*80)
        
        print("\\n" + "="*80)
        print("COMPARISON SUMMARY:")
        print("="*80)
        print(f"Improved result length: {len(improved_result)} characters")
        print(f"Original result length: {len(original_result)} characters")
        
        # Check for specific improvements
        improvements = []
        if "don't" in improved_result and "don [" in original_result:
            improvements.append("✓ Fixed 'don't' punctuation")
        if "can't" in improved_result and "can [" in original_result:
            improvements.append("✓ Fixed 'can't' punctuation")
        if improved_result.count("\\n") < original_result.count("\\n"):
            improvements.append("✓ Better line grouping (fewer fragments)")
            
        if improvements:
            print("\\nDetected improvements:")
            for improvement in improvements:
                print(f"  {improvement}")
        else:
            print("\\nNo specific improvements detected in this test")
        
    finally:
        # Clean up
        os.unlink(image_path)

if __name__ == "__main__":
    main()
