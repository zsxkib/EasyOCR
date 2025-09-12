"""
Screenshot OCR with EasyOCR
Concise, readable implementation for production Cog deployment.
"""

from typing import Any, Dict, List, Optional, Tuple
import gc
import json
import os
import re

import cv2
import numpy as np
from PIL import Image
import easyocr
import torch
from cog import BasePredictor, Input, Path

from pydantic import BaseModel, Field


class TextRegion(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    polygon: Optional[List[int]] = None


class ModelOutput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    markdown: Path
    metadata: str


DEFAULT_LANGS = ["en", "es", "fr", "de", "it", "pt"]

# Character sets used throughout for clarity
LETTERS_DIGITS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BASIC_PUNCT = " .,:;!?\'\"-–—_/&%$@#()+*=“”‘’`~^|\\"
RARE_SYMBOLS_BLOCKLIST = "§•◊©®™·•♥◆◇♦▼▲►◄¶†‡…$^~`|\\"


class Predictor(BasePredictor):
    """Production-grade screenshot OCR with minimal, clear code."""

    def setup(self) -> None:
        # Use GPU if available; initialize lazily to keep memory low on CPU-only hosts.
        self.use_gpu = torch.cuda.is_available()
        print(f"Using GPU: {self.use_gpu}")
        # Be conservative with threads on CPU
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass
        self._reader_cache: Dict[tuple, easyocr.Reader] = {}
        self._default_langs = tuple(DEFAULT_LANGS)

    def _get_reader(self, langs: List[str]) -> easyocr.Reader:
        key = tuple(langs)
        rdr = self._reader_cache.get(key)
        if rdr is None:
            rdr = easyocr.Reader(langs, gpu=self.use_gpu, verbose=False)
            self._reader_cache[key] = rdr
        return rdr

    def _auto_invert(self, gray: np.ndarray, strategy: str) -> np.ndarray:
        """Ensure dark text on light background.
        strategy: 'auto'|'force'|'disable'. In 'auto', uses Otsu to infer background.
        """
        if strategy == "disable":
            return gray
        if strategy == "force":
            return cv2.bitwise_not(gray)
        # auto - decide via Otsu
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = (th == 255).mean()
        return cv2.bitwise_not(gray) if white_ratio < 0.5 else gray

    def _deskew(self, gray: np.ndarray, enable: bool, max_angle: float = 7.0) -> np.ndarray:
        """Deskew small-angle rotations using Hough-detected near-horizontal lines."""
        if not enable:
            return gray
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
        angle_deg = 0.0
        if lines is not None:
            angles = []
            for l in lines[:200]:
                for rho, theta in l:
                    deg = (theta * 180.0 / np.pi) - 90.0
                    if -max_angle <= deg <= max_angle:
                        angles.append(deg)
            if angles:
                angle_deg = float(np.median(angles))
        if abs(angle_deg) < 0.1:
            return gray
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _trim_borders(self, gray: np.ndarray, enable: bool) -> np.ndarray:
        """Trim outer whitespace/borders using the union of external contours."""
        if not enable:
            return gray
        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = cv2.bitwise_not(bw)
        cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return gray
        x, y, w, h = cv2.boundingRect(np.vstack(cnts))
        pad = 5
        y0 = max(0, y - pad); x0 = max(0, x - pad)
        y1 = min(gray.shape[0], y + h + pad); x1 = min(gray.shape[1], x + w + pad)
        return gray[y0:y1, x0:x1]

    def _apply_binarization(self, gray: np.ndarray, mode: str) -> np.ndarray:
        """Apply optional binarization. Use for scans or plates when grayscale is noisy."""
        if mode == "none":
            return gray
        if mode == "otsu":
            return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if mode == "adaptive_mean":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
        if mode == "adaptive_gaussian":
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        return gray

    def _morph(self, bw: np.ndarray, op: str, k: int) -> np.ndarray:
        """Apply simple morphology after binarization to connect/disconnect strokes."""
        if op == "none" or k <= 0:
            return bw
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        if op == "dilate":
            return cv2.dilate(bw, kernel, iterations=1)
        if op == "erode":
            return cv2.erode(bw, kernel, iterations=1)
        if op == "open":
            return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        if op == "close":
            return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        return bw

    def _remove_shadow(self, gray: np.ndarray, strength: int) -> np.ndarray:
        """Reduce uneven lighting by subtracting a large-radius median-blurred background."""
        if strength <= 0:
            return gray
        radius = int(max(3, min(99, strength * 5)) | 1)
        bg = cv2.medianBlur(gray, radius)
        diff = cv2.absdiff(gray, bg)
        return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    def _estimate_skew(self, gray: np.ndarray, max_angle: float = 10.0) -> float:
        """Estimate page skew angle (degrees). Positive = clockwise."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
        if lines is None:
            return 0.0
        angles = []
        for l in lines[:200]:
            for rho, theta in l:
                deg = (theta * 180.0 / np.pi) - 90.0
                if -max_angle <= deg <= max_angle:
                    angles.append(deg)
        return float(np.median(angles)) if angles else 0.0

    def _preprocess(
        self,
        img: np.ndarray,
        enable: bool,
        upscale_min_dim: int = 600,
        clahe: bool = True,
        denoise_strength: int = 0,
        sharpen: bool = True,
        deskew: bool = True,
        trim_borders: bool = True,
        invert_strategy: str = "auto",
        binarize: str = "none",
        morph_op: str = "none",
        morph_kernel: int = 0,
        remove_shadow_strength: int = 0,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        if upscale_min_dim > 0 and min(h, w) < upscale_min_dim:
            scale = min(3.0, float(upscale_min_dim) / max(1.0, float(min(h, w))))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        if not enable:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = self._remove_shadow(gray, remove_shadow_strength)
        gray = self._auto_invert(gray, invert_strategy)
        gray = self._deskew(gray, deskew)

        if clahe:
            clahe_op = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
            gray = clahe_op.apply(gray)

        if denoise_strength > 0:
            hs = max(1, min(15, int(denoise_strength)))
            gray = cv2.fastNlMeansDenoising(gray, None, h=hs, templateWindowSize=7, searchWindowSize=21)

        if sharpen:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
            gray = cv2.addWeighted(gray, 1.25, blur, -0.25, 0)

        if binarize != "none":
            bw = self._apply_binarization(gray, binarize)
            bw = self._morph(bw, morph_op, morph_kernel)
            gray = bw

        gray = self._trim_borders(gray, trim_borders)

        if gray.ndim == 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray

    def _load_image(self, path: Path) -> np.ndarray:
        """Load via PIL for broad format support, then to OpenCV BGR."""
        im = Image.open(path)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        arr = np.array(im)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def _group_into_lines(self, filtered_regions):
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
    
    def _reconstruct_line_text(self, line_regions):
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
                if gap > max(2, char_width * 0.2):
                    reconstructed += " " + text
                else:
                    # No space - likely part of same word
                    reconstructed += text
                    
        return reconstructed
    
    def _to_detections(self, results: List, min_conf: float, include_bboxes: bool, include_polygons: bool) -> List[TextRegion]:
        out: List[TextRegion] = []
        
        # Filter low confidence results
        filtered = []
        for bbox, text, conf in results:
            if not text or conf < min_conf:
                continue
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            y_min, x_min, y_max, x_max = min(ys), min(xs), max(ys), max(xs)
            filtered.append((y_min, x_min, xs, ys, text, float(conf), y_max, x_max))
        
        # Sort by vertical position first for line grouping
        filtered.sort(key=lambda t: (t[0], t[1]))
        
        # Group regions into lines
        lines = self._group_into_lines(filtered)
        
        # Process each line
        for line_regions in lines:
            # Reconstruct the line text
            line_text = self._reconstruct_line_text(line_regions)
            
            if not line_text.strip():
                continue
                
            # Calculate combined bounding box for the line
            line_x_min = min(r[1] for r in line_regions)
            line_y_min = min(r[0] for r in line_regions) 
            line_x_max = max(r[7] for r in line_regions)  # x_max
            line_y_max = max(r[6] for r in line_regions)  # y_max
            
            # Average confidence for the line
            avg_conf = sum(r[5] for r in line_regions) / len(line_regions)
            
            bbox_vals = (line_x_min, line_y_min, line_x_max, line_y_max) if include_bboxes else (None, None, None, None)
            
            # For polygons, use the combined bounding box corners
            poly = None
            if include_polygons:
                poly = [line_x_min, line_y_min, line_x_max, line_y_min, 
                       line_x_max, line_y_max, line_x_min, line_y_max]
            
            out.append(TextRegion(
                text=line_text,
                confidence=avg_conf,
                x1=bbox_vals[0], y1=bbox_vals[1], x2=bbox_vals[2], y2=bbox_vals[3],
                polygon=poly
            ))
        
        return out
    
    def predict(
        self,
        image: Path = Input(description="Screenshot or image file"),
        # High-level preset for UX simplicity.
        profile: str = Input(
            description="Preset tuned for document type",
            default="auto",
            choices=["auto", "book", "scan", "screenshot", "plate"],
        ),
        languages: str = Input(
            description="Languages preset or custom list",
            default="auto",
            choices=[
                "auto",
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "en,es",
                "en,es,fr",
                "en,es,fr,de",
                "en,es,fr,de,it,pt",
                "custom",
            ],
        ),
        custom_languages: str = Input(
            description="Custom language codes (comma-separated) — used when languages=custom",
            default="",
        ),
        # OCR controls
        ocr_decoder: str = Input(description="Text decoder", default="beamsearch", choices=["greedy", "beamsearch"]),
        beam_width: int = Input(description="Beam width (when decoder=beamsearch)", default=7, ge=1, le=15),
        allow_basic_punct: bool = Input(description="Restrict to common punctuation to avoid odd symbols", default=True),
        allow_brackets: bool = Input(description="Permit brackets []{}<>", default=False),
        merge_level: str = Input(description="Text merging strength", default="auto", choices=["auto","low","medium","high"]),
        autotune: bool = Input(description="Try a few parameter combos and pick best", default=True),
        block_rare_symbols: bool = Input(description="Block uncommon symbols ($§•◊ etc.)", default=True),
        # Preprocess controls
        preprocessing: bool = Input(description="Apply preprocessing (recommended)", default=True),
        upscale_min_dim: int = Input(description="Upscale small images so the shorter side reaches this size (px)", default=900, ge=0, le=4000),
        clahe: bool = Input(description="Apply mild CLAHE contrast enhancement", default=True),
        denoise_strength: int = Input(description="Denoising strength (0=off, 1-15)", default=0, ge=0, le=15),
        sharpen: bool = Input(description="Apply mild unsharp mask to preserve small punctuation", default=True),
        deskew: bool = Input(description="Deskew small angles", default=True),
        trim_borders: bool = Input(description="Trim outer borders", default=True),
        invert_strategy: str = Input(description="Ensure dark text on light background", default="auto", choices=["auto","force","disable"]),
        binarize: str = Input(description="Binarization mode", default="auto", choices=["auto","none","otsu","adaptive_mean","adaptive_gaussian"]),
        morph_op: str = Input(description="Morphology op after binarize", default="none", choices=["none","dilate","erode","open","close"]),
        morph_kernel: int = Input(description="Morph kernel size", default=0, ge=0, le=15),
        remove_shadow_strength: int = Input(description="Uneven lighting removal (0=off)", default=0, ge=0, le=20),
        smart_preprocessing: bool = Input(description="Auto-tune preprocessing based on image stats", default=True),
        # Output controls
        min_confidence: float = Input(description="Minimum confidence (0.0-1.0)", default=0.4, ge=0.0, le=1.0),
        text_only: bool = Input(description="Return only text lines (list of strings)", default=False),
        include_bboxes: bool = Input(description="Include x1,y1,x2,y2 in output", default=True),
        include_polygons: bool = Input(description="Include 4-point polygon as flat list", default=False),
        # Post-processing controls
        remove_page_numbers: bool = Input(description="Remove likely page numbers near bottom", default=True),
        dehyphenate: bool = Input(description="Join hyphenated words across lines", default=True),
        normalize_quotes: bool = Input(description="Normalize common quotation marks and dashes", default=True),
    ) -> ModelOutput:
        try:
            # Validate inputs
            if not image or not os.path.exists(image):
                raise ValueError(f"Image file not found or invalid: {image}")
            
            if min_confidence < 0.0 or min_confidence > 1.0:
                raise ValueError(f"min_confidence must be between 0.0 and 1.0, got: {min_confidence}")
            
            # Load and preprocess image
            print(f"Processing image: {image}")
            arr = self._load_image(image)
            
            if arr is None or arr.size == 0:
                raise ValueError("Failed to load image or image is empty")
            # Estimate skew (for reporting)
            gray0 = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr.copy()
            est_skew = self._estimate_skew(gray0)
                
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            raise
        try:
            processed = self._preprocess(
                arr,
                preprocessing,
                upscale_min_dim=upscale_min_dim,
                clahe=clahe,
                denoise_strength=denoise_strength,
                sharpen=sharpen,
                deskew=deskew,
                trim_borders=trim_borders,
                invert_strategy=invert_strategy,
                binarize=auto_binarize if isinstance(auto_binarize, str) else binarize,
                morph_op=morph_op,
                morph_kernel=morph_kernel,
                remove_shadow_strength=int(auto_shadow),
            )
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise

        # Determine languages from preset or custom
        chosen = (languages or "auto").strip()
        if chosen == "auto":
            # For book/document text, use fewer languages for better accuracy
            langs = ["en"]  # Start with English only for better accuracy
            reader = self._get_reader(langs)
        elif chosen == "custom":
            langs = [s.strip() for s in custom_languages.split(",") if s.strip()] or ["en"]
            reader = self._get_reader(langs)
        else:
            langs = [s.strip() for s in chosen.split(",") if s.strip()]
            reader = self._get_reader(langs)

        # Character allow/block lists
        allowlist = None
        blocklist = None
        if allow_basic_punct:
            brackets = "[]{}<>" if allow_brackets else ""
            allowlist = LETTERS_DIGITS + BASIC_PUNCT + brackets
        if block_rare_symbols:
            blocklist = RARE_SYMBOLS_BLOCKLIST

        # Choose width/height thresholds based on merge_level
        if merge_level == "low":
            width_ths, height_ths = 0.35, 0.3
        elif merge_level == "medium":
            width_ths, height_ths = 0.5, 0.4
        elif merge_level == "high":
            width_ths, height_ths = 0.7, 0.6
        else:  # auto default tuned for books/documents
            width_ths, height_ths = 0.5, 0.4

        def run_once(wt, ht, dec, bw):
            return reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                width_ths=wt,
                height_ths=ht,
                slope_ths=0.2,
                ycenter_ths=0.7,
                add_margin=0.15,
                decoder=dec,
                beamWidth=bw,
                allowlist=allowlist,
                blocklist=blocklist,
                batch_size=1,
                workers=0,
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.4,
            )

        try:
            print(f"Running OCR with languages: {langs}")
            if autotune:
                combos = [
                    (width_ths, height_ths, ocr_decoder, beam_width),
                    (min(0.6, width_ths+0.1), min(0.5, height_ths+0.1), ocr_decoder, min(15, beam_width+2)),
                    (max(0.4, width_ths-0.1), max(0.3, height_ths-0.1), ocr_decoder, max(3, beam_width-2)),
                ]
                best = None
                best_score = -1e9
                for wt, ht, dec, bw in combos:
                    res = run_once(wt, ht, dec, bw)
                    if not res:
                        continue
                    avg_conf = float(sum([r[2] for r in res]) / max(1, len(res)))
                    score = avg_conf - 0.002 * len(res)
                    if score > best_score:
                        best_score = score
                        best = res
                results = best if best is not None else run_once(width_ths, height_ths, ocr_decoder, beam_width)
            else:
                results = run_once(width_ths, height_ths, ocr_decoder, beam_width)
        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            raise
        except Exception as e:
            print(f"Error during OCR processing: {str(e)}")
            raise

        detections = self._to_detections(results, min_confidence, include_bboxes, include_polygons)
        # Cleanup
        if self.use_gpu:
            gc.collect()
            torch.cuda.empty_cache()

        # Optionally remove likely page numbers at bottom
        if remove_page_numbers and detections:
            max_y = max(d.y2 for d in detections if d.y2 is not None)
            def is_page_num(d: TextRegion) -> bool:
                if d.y1 is None or d.y2 is None:
                    return False
                txt = (d.text or "").strip()
                norm = re.sub(r"[^0-9]", "", txt)
                if not norm.isdigit() or len(norm) == 0:
                    return False
                if len(norm) > 4:
                    return False
                return d.y1 > 0.80 * max_y
            detections = [d for d in detections if not is_page_num(d)]

        # Create markdown output preserving original formatting as much as possible
        markdown_content = ""
        
        if text_only:
            markdown_content = "\n".join(d.text for d in detections)
        else:
            # Improved text reconstruction with better paragraph handling
            prev_y_pos = None
            last_line_ended_with_hyphen = False
            
            for i, d in enumerate(detections):
                text = d.text.strip()
                if not text:
                    continue
                
                # Calculate line spacing for paragraph detection
                current_y = d.y1 if d.y1 is not None else 0
                line_height = (d.y2 - d.y1) if (d.y1 is not None and d.y2 is not None) else 20
                
                # Determine if this starts a new paragraph
                is_new_paragraph = False
                if prev_y_pos is not None:
                    vertical_gap = current_y - prev_y_pos
                    # New paragraph if gap is > 1.5x line height
                    is_new_paragraph = vertical_gap > line_height * 1.5
                
                # Header detection
                is_header = (
                    len(text) < 50 and 
                    (text.isupper() or 
                     any(word in text.upper() for word in ['CHAPTER', 'SECTION', 'PART', 'THE']) or
                     all(word.istitle() for word in text.split() if word.isalpha()))
                )
                
                if is_header:
                    if markdown_content and not markdown_content.endswith('\n\n'):
                        markdown_content += "\n\n"
                    if len(text) < 20:
                        markdown_content += f"# {text}\n\n"
                    else:
                        markdown_content += f"## {text}\n\n"
                    last_line_ended_with_hyphen = False
                elif is_new_paragraph and markdown_content:
                    # Start new paragraph
                    if not markdown_content.endswith('\n\n'):
                        markdown_content += "\n\n"
                    markdown_content += text
                    last_line_ended_with_hyphen = text.endswith(('-', '‑'))
                else:
                    # Continue current line/paragraph
                    if dehyphenate and last_line_ended_with_hyphen:
                        # remove trailing hyphen/space then append without extra space
                        markdown_content = markdown_content.rstrip().rstrip('-').rstrip(' ')
                        markdown_content += text.lstrip()
                    else:
                        if markdown_content and not markdown_content.endswith(' '):
                            if markdown_content[-1] in '.!?':
                                markdown_content += " "
                            elif not text.startswith((' ', '.', ',', ';', ':', '!', '?')):
                                markdown_content += " "
                        markdown_content += text
                    last_line_ended_with_hyphen = text.endswith(('-', '‑'))
                
                prev_y_pos = current_y + line_height
            
            # Ensure proper paragraph endings
            if markdown_content and not markdown_content.endswith('\n'):
                markdown_content += "\n"
        
        # Clean up extra whitespace and fix common OCR errors
        markdown_content = markdown_content.strip()

        # Additional postprocessing for common English contractions and spacing
        def fix(text: str) -> str:
            repl = [
                (r"\b[Dd]on\s+t\b", "don't"),
                (r"\b[Cc]an\s+t\b", "can't"),
                (r"\b[Tt]hat\s+s\b", "that's"),
                (r"\b[Ii]t\s+s\b", "it's"),
                (r"\b[Yy]ou\s+re\b", "you're"),
                (r"\bW[eE]\s+re\b", "we're"),
                (r"\b[Ii]\s+m\b", "I'm"),
                (r"\b[Ii]\s+ve\b", "I've"),
                (r"\b[Yy]ou\s+ve\b", "you've"),
                (r"\b[Tt]hey\s+re\b", "they're"),
                (r"\s+,", ","),
                (r"\s+\.", "."),
                (r"\s+([;:!?])", r"\1"),
                (r"\$\s+", "$"),
            ]
            for pat, to in repl:
                text = re.sub(pat, to, text)
            if normalize_quotes:
                # normalize double hyphen to em-dash; avoid naive quote flipping
                text = text.replace('--', '—')
            # collapse multiple spaces
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\b([A-Za-z]+)'\s*[$sS]\b", r"\1's", text)
            text = re.sub(r"_+", "", text)
            return text
        markdown_content = fix(markdown_content)
        
        # Write to markdown file
        out_file = Path("extracted_text.md")
        out_file.write_text(markdown_content, encoding='utf-8')
        
        # Create metadata (preserve output schema)
        metadata = {
            "total_regions": len(detections),
            "avg_confidence": round(sum(r.confidence for r in detections) / len(detections), 3) if detections else 0.0,
            "languages_used": langs,
            "preprocessing_applied": preprocessing,
            "include_bboxes": include_bboxes,
            "include_polygons": include_polygons,
            "text_only": text_only,
            "regions": [r.model_dump(exclude_none=True) for r in detections] if not text_only else []
        }
        
        # For debugging/monitoring, print metadata
        print(f"OCR Results: {len(detections)} regions, avg confidence: {metadata['avg_confidence']}")
        
        return ModelOutput(
            markdown=out_file,
            metadata=json.dumps(metadata, indent=2)
        )
