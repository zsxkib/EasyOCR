# Screenshot OCR with EasyOCR

Clean, production-ready implementation for Cog deployment on Replicate.

## Features

- **World-class code**: Clean, readable Pydantic models with proper type safety
- **Smart preprocessing**: DPI upscaling, CLAHE contrast enhancement, denoising
- **Flexible output**: Text-only mode or structured markdown with coordinates
- **Multi-language**: 6 default languages (en, es, fr, de, it, pt) + custom support
- **GPU optimized**: Auto-detects and uses GPU when available
- **Easy coordinates**: Flat arrays in metadata for simple extraction

## Input Parameters

- `image`: Screenshot or image file (PNG, JPG, WebP, etc.)
- `languages`: Comma-separated language codes (empty = defaults)
- `min_confidence`: Confidence threshold 0.0-1.0 (default: 0.25)
- `preprocessing`: Apply image enhancement (default: true)
- `text_only`: Return simple text lines (default: false)
- `include_bboxes`: Include x1,y1,x2,y2 coordinates (default: true)
- `include_polygons`: Include 4-point polygon as flat list (default: false)

## Output Format

Returns `ModelOutput` with:
- `markdown`: Path to generated markdown file
- `metadata`: JSON string with regions, coordinates, and processing info

## Deployment

1. **Production**: `cog.yaml` has `gpu: true` for Replicate deployment
2. **Local testing**: Set `gpu: false` in cog.yaml for Mac development

## Code Quality

- ✅ No excessive try/catch blocks
- ✅ Minimal dependencies (easyocr, opencv, PIL, numpy, pydantic)
- ✅ Clean method structure with clear responsibilities  
- ✅ Proper type hints and Pydantic validation
- ✅ Smart reading order (top-to-bottom, left-to-right)
- ✅ Memory cleanup and GPU cache management

Perfect for production OCR at scale! 🚀
