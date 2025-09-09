#!/usr/bin/env python3
"""
Render Text From Coordinates — LLM‑Ready Implementation

Given text regions with pixel bounding boxes, render each string inside its box 
as large as possible, wrapping if needed, and center it. Output a guided image 
(with light boxes) and a clean image (text only).
"""

import json
import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont

# Common font paths for different systems
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/Library/Fonts/Arial Unicode.ttf",  # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS fallback
    "C:/Windows/Fonts/arial.ttf",  # Windows
]

def load_font(size: int, font_path: str = None) -> ImageFont.FreeTypeFont:
    """Load a Unicode-capable font, falling back to system fonts if needed."""
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    
    for path in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    
    # Final fallback to default font
    return ImageFont.load_default()

def measure_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Measure text width and height."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def wrap_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wrap text to fit within max_width, using greedy word-wrapping."""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = (current_line + " " + word).strip()
        if current_line and measure_text(draw, test_line, font)[0] > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    return lines or [""]

def find_best_font_size(draw: ImageDraw.Draw, text: str, box_width: int, box_height: int, font_path: str = None) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Binary search for the largest font size that fits the text in the given box."""
    low, high = 4, max(6, int(box_height))
    best_result = None
    
    while low <= high:
        mid = (low + high) // 2
        font = load_font(mid, font_path)
        line_height = int(mid * 1.15)  # 1.15x line spacing
        
        # Try single line first
        width, height = measure_text(draw, text, font)
        if width <= box_width and height <= box_height:
            best_result = (font, [text], line_height)
            low = mid + 1
            continue
        
        # Try wrapped lines
        lines = wrap_text(draw, text, font, box_width)
        max_line_width = max((measure_text(draw, line, font)[0] for line in lines), default=0)
        total_height = line_height * len(lines)
        
        if max_line_width <= box_width and total_height <= box_height:
            best_result = (font, lines, line_height)
            low = mid + 1
        else:
            high = mid - 1
    
    # If no size fits, fall back to size 4 with wrapping
    if not best_result:
        font = load_font(4, font_path)
        line_height = int(4 * 1.15)
        lines = wrap_text(draw, text, font, box_width)
        return font, lines, line_height
    
    return best_result

def main():
    parser = argparse.ArgumentParser(description="Render text from OCR coordinates")
    parser.add_argument("json_path", help="Path to JSON file with text regions")
    parser.add_argument("--font", default=None, help="Path to font file")
    parser.add_argument("--padding", type=int, default=30, help="Canvas padding")
    args = parser.parse_args()
    
    # Load OCR data
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    regions = data.get("regions", [])
    if not regions:
        sys.exit("No regions found in input file")
    
    # Calculate canvas size
    padding = args.padding
    canvas_width = max(r["x2"] for r in regions) + padding
    canvas_height = max(r["y2"] for r in regions) + padding
    
    # Create images
    img_with_boxes = Image.new("RGB", (canvas_width, canvas_height), "white")
    img_clean = Image.new("RGB", (canvas_width, canvas_height), "white")
    
    draw_boxes = ImageDraw.Draw(img_with_boxes)
    draw_clean = ImageDraw.Draw(img_clean)
    
    print(f"📐 Canvas size: {canvas_width} x {canvas_height}")
    print(f"📝 Processing {len(regions)} text regions...")
    
    for i, region in enumerate(regions):
        x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
        text = region.get("text", "")
        
        if not text.strip():
            continue
            
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        
        # Draw light gray box for guided version
        draw_boxes.rectangle(
            [x1, y1, x2, y2], 
            fill=(245, 245, 245), 
            outline=(220, 220, 220), 
            width=1
        )
        
        # Find best font size and layout
        font, lines, line_height = find_best_font_size(draw_boxes, text, box_width, box_height, args.font)
        
        # Calculate text positioning (centered)
        total_text_height = len(lines) * line_height
        start_y = y1 + (box_height - total_text_height) // 2
        
        # Render text lines
        for line_idx, line in enumerate(lines):
            line_width, _ = measure_text(draw_boxes, line, font)
            line_x = x1 + (box_width - line_width) // 2
            line_y = start_y + line_idx * line_height
            
            # Draw on both images
            draw_boxes.text((line_x, line_y), line, font=font, fill=(0, 0, 0))
            draw_clean.text((line_x, line_y), line, font=font, fill=(0, 0, 0))
        
        print(f"  ✅ {i+1:2d}. '{text}' -> {len(lines)} line(s), font size ~{font.size}")
    
    # Save images
    img_with_boxes.save("rendered_with_boxes.png")
    img_clean.save("rendered_clean.png")
    
    print(f"\n🎉 Generated images:")
    print(f"   📦 rendered_with_boxes.png")
    print(f"   ✨ rendered_clean.png")

if __name__ == "__main__":
    main()
