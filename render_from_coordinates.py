#!/usr/bin/env python3
"""
Render Text From Coordinates — Background‑aware Boxes

Given text regions with pixel bounding boxes, render each string inside its box
as large as possible, wrapping if needed, and center it. Outputs a single image:
- rendered_with_boxes.png (background‑aware box colors + text)
"""

import json
import os
import sys
import argparse
from typing import List, Tuple
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
    return ImageFont.load_default()

def measure_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """Measure text width and height."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def wrap_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width, using greedy word-wrapping."""
    words = text.split()
    lines: List[str] = []
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

def find_best_font_size(draw: ImageDraw.Draw, text: str, box_width: int, box_height: int, font_path: str = None) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
    """Binary search for the largest font size that fits the text in the given box."""
    low, high = 4, max(6, int(box_height))
    best_result = None
    while low <= high:
        mid = (low + high) // 2
        font = load_font(mid, font_path)
        line_height = int(mid * 1.15)  # 1.15x line spacing
        width, height = measure_text(draw, text, font)
        if width <= box_width and height <= box_height:
            best_result = (font, [text], line_height)
            low = mid + 1
            continue
        lines = wrap_text(draw, text, font, box_width)
        max_line_width = max((measure_text(draw, line, font)[0] for line in lines), default=0)
        total_height = line_height * len(lines)
        if max_line_width <= box_width and total_height <= box_height:
            best_result = (font, lines, line_height)
            low = mid + 1
        else:
            high = mid - 1
    if not best_result:
        font = load_font(4, font_path)
        line_height = int(4 * 1.15)
        lines = wrap_text(draw, text, font, box_width)
        return font, lines, line_height
    return best_result

def average_bg_color(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int]:
    """Approximate the average color of the background region by downsampling to 1x1."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    crop = img.crop((x1, y1, x2, y2))
    # Use BOX filter for average-like sampling
    small = crop.resize((1, 1), resample=Image.BOX)
    r, g, b, *a = small.getpixel((0, 0))
    return int(r), int(g), int(b)

def adjust_color(c: Tuple[int, int, int], delta: int = 12) -> Tuple[int, int, int]:
    """Slightly lighten the sampled color to keep boxes visible while matching background."""
    r, g, b = c
    return min(255, r + delta), min(255, g + delta), min(255, b + delta)

def main():
    parser = argparse.ArgumentParser(description="Render text from OCR coordinates")
    parser.add_argument("json_path", help="Path to JSON file with text regions")
    parser.add_argument("--font", default=None, help="Path to font file")
    parser.add_argument("--padding", type=int, default=30, help="Canvas padding")
    parser.add_argument("--background", default=None, help="Optional background image to draw on")
    args = parser.parse_args()

    # Load OCR data
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support either top-level regions or metadata.regions
    regions = data.get("regions")
    if regions is None and isinstance(data.get("metadata"), dict):
        regions = data["metadata"].get("regions")
    if not regions:
        sys.exit("No regions found in input file")

    def extract_bbox(r):
        # Primary schema: x1,y1,x2,y2
        if all(k in r for k in ("x1", "y1", "x2", "y2")):
            return int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        # Fallback schema: bbox [x1,y1,x2,y2]
        if isinstance(r.get("bbox"), (list, tuple)) and len(r["bbox"]) == 4:
            x1, y1, x2, y2 = r["bbox"]
            return int(x1), int(y1), int(x2), int(y2)
        raise KeyError("Region missing bounding box fields (expected x1,y1,x2,y2 or bbox[4])")

    # Calculate canvas size
    padding = args.padding
    max_x2 = max(extract_bbox(r)[2] for r in regions)
    max_y2 = max(extract_bbox(r)[3] for r in regions)

    base_bg = None
    if args.background:
        try:
            base_bg = Image.open(args.background).convert("RGBA")
        except Exception as e:
            sys.exit(f"Failed to load background image: {e}")

    target_w = max(max_x2 + padding, base_bg.width if base_bg else 0)
    target_h = max(max_y2 + padding, base_bg.height if base_bg else 0)

    # Create base canvas (RGBA) and optional background
    base_canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
    if base_bg is not None:
        base_canvas.paste(base_bg, (0, 0))

    # Overlay for boxes (to keep them beneath the text)
    overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw_measure = ImageDraw.Draw(base_canvas)
    draw_overlay = ImageDraw.Draw(overlay)

    print(f"📐 Canvas size: {target_w} x {target_h}")
    print(f"📝 Processing {len(regions)} text regions...")

    # Prepare text layouts to draw after compositing boxes
    prepared = []  # list of (x1, y1, x2, y2, font, lines, line_height, text)

    for i, region in enumerate(regions):
        x1, y1, x2, y2 = extract_bbox(region)
        text = str(region.get("text", "")).strip()
        if not text:
            continue

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)

        # Sample background color and draw a semi-transparent matching box
        bg_color = average_bg_color(base_canvas, x1, y1, x2, y2)
        fill = (*adjust_color(bg_color, 12), 120)  # lightened + alpha
        outline = (*adjust_color(bg_color, -20), 180) if hasattr(tuple(), "__getitem__") else (*bg_color, 180)
        draw_overlay.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=1)

        # Find best font size and layout
        font, lines, line_height = find_best_font_size(draw_measure, text, box_width, box_height, args.font)

        prepared.append((x1, y1, x2, y2, font, lines, line_height, text))
        print(f"  ✅ {len(prepared):2d}. '{text}' -> {len(lines)} line(s), font size ~{font.size}")

    # Composite overlay under the text
    base_canvas = Image.alpha_composite(base_canvas, overlay)
    draw_text = ImageDraw.Draw(base_canvas)

    # Draw text after boxes
    for (x1, y1, x2, y2, font, lines, line_height, _text) in prepared:
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        total_text_height = len(lines) * line_height
        start_y = y1 + (box_height - total_text_height) // 2
        for line_idx, line in enumerate(lines):
            line_width, _ = measure_text(draw_text, line, font)
            line_x = x1 + (box_width - line_width) // 2
            line_y = start_y + line_idx * line_height
            draw_text.text((line_x, line_y), line, font=font, fill=(0, 0, 0, 255))

    # Save single output image
    out_path = "rendered_with_boxes.png"
    base_canvas.convert("RGB").save(out_path)
    print(f"\n🎉 Generated image: {out_path}")

if __name__ == "__main__":
    main()
