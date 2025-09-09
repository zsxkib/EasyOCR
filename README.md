# Screenshot OCR with Coordinate Extraction

> **Production-ready OCR model for building camera translation apps**  
> Extract text with precise pixel coordinates • GPU-accelerated • Multi-language • Deploy to Replicate

<br>

![Replicate](https://replicate.com/zsxkib/easyocr/badge) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![CUDA](https://img.shields.io/badge/CUDA-12.1-green)

## Quick Start

### 🚀 Use the Deployed Model

```python
import replicate
import json

output = replicate.run(
    "zsxkib/easyocr",
    input={
        "image": "https://example.com/your-image.jpg",
        "include_bboxes": True,
        "preprocessing": True,
        "min_confidence": 0.25
    }
)

# Extract coordinates for overlay translation
metadata = json.loads(output["metadata"])
for region in metadata["regions"]:
    text = region["text"]
    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    # Now you can overlay translations at these exact coordinates
```

### 🔧 Deploy Your Own

```bash
git clone https://github.com/zsxkib/EasyOCR.git
cd EasyOCR
cog predict -i image=@your-image.jpg
cog push r8.im/your-username/easyocr
```

## Why This OCR Model?

**Perfect for camera translation apps** like Google Translate's camera feature:

1. **📍 Precise Coordinates**: Get exact pixel positions for text overlay
2. **🎯 Layout Preservation**: Maintains original text positioning and reading order  
3. **🌍 Multi-Language**: 80+ languages with Unicode support
4. **⚡ GPU Optimized**: Fast inference with automatic CPU fallback
5. **📱 Mobile-Ready**: Handles low-resolution screenshots and mobile captures

## Input & Output

### Input Parameters
```python
{
    "image": "Screenshot or image file",
    "languages": "Comma-separated codes (e.g., 'en,es,fr')",
    "min_confidence": 0.25,          # Filter low-confidence detections
    "preprocessing": True,           # Enhanced image processing
    "include_bboxes": True,         # Essential for coordinate overlay
    "include_polygons": False       # Optional detailed shapes
}
```

### Output Structure
```python
{
    "markdown": "path/to/extracted_text.md",
    "metadata": {
        "total_regions": 9,
        "avg_confidence": 0.944,
        "languages_used": ["en", "es"],
        "regions": [
            {
                "text": "Hello World",
                "confidence": 0.99,
                "x1": 100, "y1": 50,    # Top-left coordinates
                "x2": 200, "y2": 80     # Bottom-right coordinates
            }
            // ... more text regions
        ]
    }
}
```

## Camera Translation App Architecture

```
📱 Camera Input
    ↓
🔍 Screenshot OCR (this model)
    ↓
📍 Text + Coordinates Extracted
    ↓
🌐 Translation API (Google/Azure/etc.)
    ↓
🎨 Overlay Translated Text
    ↓
📱 Augmented Camera View
```

This model handles the **crucial first step**: extracting text with pixel-perfect coordinates so you can overlay translations in the exact same positions.

## Examples

### Text Detection Results
```python
# Input: Screenshot of a menu
# Output: Structured regions ready for translation overlay

regions = [
    {"text": "Pasta Carbonara", "x1": 120, "y1": 200, "x2": 280, "y2": 230},
    {"text": "€12.50", "x1": 350, "y1": 200, "x2": 400, "y2": 230},
    {"text": "Fresh ingredients", "x1": 120, "y1": 235, "x2": 260, "y2": 255}
]
```

### Integration Example
```python
def translate_camera_view(image_path):
    # Step 1: Extract text with coordinates
    ocr_result = replicate.run("zsxkib/easyocr", input={"image": image_path})
    regions = json.loads(ocr_result["metadata"])["regions"]
    
    # Step 2: Translate each text region
    for region in regions:
        original_text = region["text"]
        translated = translate_api(original_text, target_lang="es")
        
        # Step 3: Overlay translation at exact coordinates
        overlay_text(image, translated, region["x1"], region["y1"], 
                    region["x2"], region["y2"])
    
    return augmented_image
```

## Technical Details

- **Framework**: Cog + EasyOCR + PyTorch
- **GPU**: CUDA 12.1 with automatic CPU fallback  
- **Languages**: 80+ supported with Unicode text
- **Preprocessing**: DPI upscaling, CLAHE enhancement, denoising
- **Performance**: Sub-second inference on typical screenshots
- **Memory**: Efficient cleanup prevents GPU memory leaks

## Repository Structure

```
├── predict.py              # Core OCR predictor (205 lines)
├── cog.yaml               # Deployment configuration
├── requirements.txt       # Dependencies
├── test_replicate.py      # Usage examples
├── render_from_coordinates.py  # Text rendering utility
└── SUMMARY.md            # Technical documentation
```

## Contributing

This is designed to be the **minimal, focused OCR backbone** for camera translation apps. 

- Keep it simple and readable
- No unnecessary features or bloated code
- Focus on coordinate accuracy and performance
- World-class open source standards

## License

This repository contains both original code and upstream EasyOCR code.

- Original code in this fork (predict.py, cog.yaml, README.md, SUMMARY.md, render_from_coordinates.py, test_replicate.py, .dockerignore) is licensed under the MIT License. See LICENSE-MIT.
- The included upstream EasyOCR code remains licensed under the Apache License 2.0. See LICENSE.
- See NOTICE for details.

---

**Ready to build the next Google Translate Camera?** This OCR foundation has you covered. ✨
