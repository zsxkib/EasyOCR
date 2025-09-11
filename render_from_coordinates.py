#!/usr/bin/env python3
"""
Render Text From Coordinates — Background‑aware Boxes

Given text regions with pixel bounding boxes, render each string inside its box
as large as possible, wrapping if needed, and center it.

Outputs a single image:
- rendered_with_boxes.png (background‑aware rounded boxes + adaptive text)

Features:
- Samples the local background color and chooses a matching, subtle fill
- Rounded corners and adjustable alpha for pleasant overlays
- Adaptive text color (black/white) with optional stroke for readability
- Greedy word wrap and binary‑searched font size to maximize legibility
- Supports either {"regions": [...]} or {"metadata": {"regions": [...]}}
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

def luminance(c: Tuple[int, int, int]) -> float:
    r, g, b = c
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def clamp(x: int) -> int:
    return 0 if x < 0 else 255 if x > 255 else x

def lighten(c: Tuple[int, int, int], delta: int) -> Tuple[int, int, int]:
    r, g, b = c
    return clamp(r + delta), clamp(g + delta), clamp(b + delta)

def darken(c: Tuple[int, int, int], delta: int) -> Tuple[int, int, int]:
    r, g, b = c
    return clamp(r - delta), clamp(g - delta), clamp(b - delta)

def auto_fill_and_outline(bg: Tuple[int, int, int]) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """Choose a subtle fill and outline based on background brightness."""
    L = luminance(bg)
    if L < 128:
        fill_rgb = lighten(bg, 16)
        outline_rgb = lighten(bg, 6)
    else:
        fill_rgb = darken(bg, 16)
        outline_rgb = darken(bg, 40)
    return (*fill_rgb, 120), (*outline_rgb, 180)

def auto_text_and_stroke(bg: Tuple[int, int, int]) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """Return (text_color, stroke_color) with contrast against background."""
    L = luminance(bg)
    if L < 140:
        # dark bg → light text
        return (255, 255, 255, 255), (0, 0, 0, 255)
    else:
        return (0, 0, 0, 255), (255, 255, 255, 255)

def estimate_text_is_dark(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Heuristic: downsample crop and check the 25th percentile luminance.
    If the lower quartile is dark, we assume original text was dark.
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    crop = img.crop((x1, y1, x2, y2)).convert("L")
    # small sample for speed and stability
    crop = crop.resize((32, 32), resample=Image.BOX)
    vals = sorted(crop.getdata())
    if not vals:
        return True  # default to dark text on empty input
    idx = max(0, min(len(vals) - 1, int(0.25 * len(vals))))
    p25 = vals[idx]
    return p25 < 110

def main():
    parser = argparse.ArgumentParser(description="Render text from OCR coordinates")
    parser.add_argument("json_path", help="Path to JSON file with text regions")
    parser.add_argument("--font", default=None, help="Path to font file")
    parser.add_argument("--padding", type=int, default=30, help="Canvas padding around the entire canvas")
    parser.add_argument("--background", default=None, help="Optional background image to draw on")
    parser.add_argument("--out", default="rendered_with_boxes.png", help="Output image path")
    parser.add_argument("--alpha", type=int, default=220, help="Box fill alpha (0-255)")
    parser.add_argument("--radius", type=int, default=6, help="Corner radius for boxes")
    parser.add_argument("--inset", type=int, default=2, help="Inner padding inside each box for text")
    parser.add_argument("--stroke", type=int, default=1, help="Text stroke width for readability")
    parser.add_argument("--text-color", choices=["auto","black","white"], default="auto", help="Text color mode")
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

        # Sample background color and draw a subtle, matching box (rounded corners)
        bg_color = average_bg_color(base_canvas, x1, y1, x2, y2)
        fill_rgba, outline_rgba = auto_fill_and_outline(bg_color)
        # Apply user alpha override
        fill_rgba = (fill_rgba[0], fill_rgba[1], fill_rgba[2], max(0, min(255, args.alpha)))
        rect = [x1, y1, x2, y2]
        try:
            draw_overlay.rounded_rectangle(rect, radius=max(0, args.radius), fill=fill_rgba, outline=outline_rgba, width=1)
        except Exception:
            draw_overlay.rectangle(rect, fill=fill_rgba, outline=outline_rgba, width=1)

        # Find best font size and layout within inset area
        inset = max(0, args.inset)
        tx1, ty1, tx2, ty2 = x1 + inset, y1 + inset, x2 - inset, y2 - inset
        tw, th = max(1, tx2 - tx1), max(1, ty2 - ty1)
        font, lines, line_height = find_best_font_size(draw_measure, text, tw, th, args.font)

        prepared.append((tx1, ty1, tx2, ty2, font, lines, line_height, text, bg_color))
        print(f"  ✅ {len(prepared):2d}. '{text}' -> {len(lines)} line(s), font size ~{font.size}")

    # Composite overlay under the text
    base_canvas = Image.alpha_composite(base_canvas, overlay)
    draw_text = ImageDraw.Draw(base_canvas)

    # Draw text after boxes
    for (x1, y1, x2, y2, font, lines, line_height, _text, bg_color) in prepared:
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        total_text_height = len(lines) * line_height
        start_y = y1 + (box_height - total_text_height) // 2
        # Choose text and stroke colors (respect likely original tone where possible)
        if args.text_color == "black":
            text_rgba, stroke_rgba = (0, 0, 0, 255), (255, 255, 255, 255)
        elif args.text_color == "white":
            text_rgba, stroke_rgba = (255, 255, 255, 255), (0, 0, 0, 255)
        else:
            # Prefer black on light backgrounds, white on dark backgrounds;
            # only use the text-tone heuristic for mid luminance cases.
            Lbg = luminance(bg_color)
            if Lbg >= 150:
                text_rgba, stroke_rgba = (0, 0, 0, 255), (255, 255, 255, 255)
            elif Lbg <= 90:
                text_rgba, stroke_rgba = (255, 255, 255, 255), (0, 0, 0, 255)
            else:
                dark = estimate_text_is_dark(base_canvas, x1, y1, x2, y2)
                if dark:
                    text_rgba, stroke_rgba = (0, 0, 0, 255), (255, 255, 255, 255)
                else:
                    text_rgba, stroke_rgba = (255, 255, 255, 255), (0, 0, 0, 255)
        for line_idx, line in enumerate(lines):
            line_width, _ = measure_text(draw_text, line, font)
            line_x = x1 + (box_width - line_width) // 2
            line_y = start_y + line_idx * line_height
            draw_text.text(
                (line_x, line_y), line, font=font, fill=text_rgba,
                stroke_width=max(0, args.stroke), stroke_fill=stroke_rgba
            )

    # Save single output image
    out_path = args.out
    base_canvas.convert("RGB").save(out_path)
    print(f"\n🎉 Generated image: {out_path}")

if __name__ == "__main__":
    main()
