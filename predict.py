"""
Screenshot OCR with EasyOCR
Concise, readable implementation for production Cog deployment.
"""

from typing import Any, Dict, List, Optional
import gc
import json
import os

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
    markdown: Path
    metadata: str


DEFAULT_LANGS = ["en", "es", "fr", "de", "it", "pt"]


class Predictor(BasePredictor):
    """Production-grade screenshot OCR with minimal, clear code."""

    def setup(self) -> None:
        # Use GPU if available; initialize lazily to keep memory low on CPU-only hosts.
        self.use_gpu = torch.cuda.is_available()
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

    def _preprocess(
        self,
        img: np.ndarray,
        enable: bool,
        upscale_min_dim: int = 600,
        clahe: bool = True,
        denoise_strength: int = 0,
        sharpen: bool = True,
    ) -> np.ndarray:
        """Balanced preprocessing aimed at preserving small punctuation:
        - Upscale small images to at least `upscale_min_dim` on the shorter side (max 3x)
        - Optional CLAHE (mild) on grayscale to improve contrast
        - Optional denoising (disabled by default)
        - Optional mild unsharp masking to restore edge detail
        """
        h, w = img.shape[:2]
        if upscale_min_dim > 0 and min(h, w) < upscale_min_dim:
            scale = min(3.0, float(upscale_min_dim) / max(1.0, float(min(h, w))))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        if not enable:
            return img

        # Contrast enhancement (CLAHE) — mild to avoid blowing out punctuation
        if clahe and img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_op = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
            gray = clahe_op.apply(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Optional denoising — keep small by default as it can erase punctuation
        if denoise_strength > 0 and img.ndim == 3:
            hs = max(1, min(15, int(denoise_strength)))
            img = cv2.fastNlMeansDenoisingColored(img, None, hs, hs, 7, 21)

        # Mild unsharp mask to keep apostrophes/accents
        if sharpen and img.ndim == 3:
            blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
            img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)

        return img

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

    def _to_detections(self, results: List, min_conf: float, include_bboxes: bool, include_polygons: bool) -> List[TextRegion]:
        out: List[TextRegion] = []
        # Filter then sort by top-to-bottom (y), then left-to-right (x) for user-friendly reading order
        filtered = []
        for bbox, text, conf in results:
            if not text or conf < min_conf:
                continue
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            y_min, x_min = min(ys), min(xs)
            filtered.append((y_min, x_min, xs, ys, text, float(conf)))
        filtered.sort(key=lambda t: (t[0], t[1]))
        for _, _, xs, ys, text, conf in filtered:
            bbox_vals = (min(xs), min(ys), max(xs), max(ys)) if include_bboxes else (None, None, None, None)
            poly = None
            if include_polygons:
                poly = []
                for i in range(4):
                    poly.extend([xs[i], ys[i]])
            out.append(TextRegion(
                text=text.strip(),
                confidence=conf,
                x1=bbox_vals[0], y1=bbox_vals[1], x2=bbox_vals[2], y2=bbox_vals[3],
                polygon=poly
            ))
        return out
    
    def predict(
        self,
        image: Path = Input(description="Screenshot or image file"),
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
        ocr_decoder: str = Input(description="Text decoder", default="greedy", choices=["greedy", "beamsearch"]),
        beam_width: int = Input(description="Beam width (when decoder=beamsearch)", default=5, ge=1, le=10),
        allow_basic_punct: bool = Input(description="Restrict to common punctuation to avoid odd symbols", default=True),
        allow_brackets: bool = Input(description="Permit brackets []{}<>", default=False),
        # Preprocess controls
        preprocessing: bool = Input(description="Apply preprocessing (recommended)", default=True),
        upscale_min_dim: int = Input(description="Upscale small images so the shorter side reaches this size (px)", default=600, ge=0, le=4000),
        clahe: bool = Input(description="Apply mild CLAHE contrast enhancement", default=True),
        denoise_strength: int = Input(description="Denoising strength (0=off, 1-15)", default=0, ge=0, le=15),
        sharpen: bool = Input(description="Apply mild unsharp mask to preserve small punctuation", default=True),
        # Output controls
        min_confidence: float = Input(description="Minimum confidence (0.0-1.0)", default=0.25, ge=0.0, le=1.0),
        text_only: bool = Input(description="Return only text lines (list of strings)", default=False),
        include_bboxes: bool = Input(description="Include x1,y1,x2,y2 in output", default=True),
        include_polygons: bool = Input(description="Include 4-point polygon as flat list", default=False),
    ) -> ModelOutput:
        # Load and preprocess image
        arr = self._load_image(image)
        processed = self._preprocess(
            arr,
            preprocessing,
            upscale_min_dim=upscale_min_dim,
            clahe=clahe,
            denoise_strength=denoise_strength,
            sharpen=sharpen,
        )

        # Determine languages from preset or custom
        chosen = (languages or "auto").strip()
        if chosen == "auto":
            langs = DEFAULT_LANGS
            reader = self._get_reader(langs)
        elif chosen == "custom":
            langs = [s.strip() for s in custom_languages.split(",") if s.strip()] or DEFAULT_LANGS
            reader = self._get_reader(langs)
        else:
            langs = [s.strip() for s in chosen.split(",") if s.strip()]
            reader = self._get_reader(langs)

        # Character allowlist (optional)
        allowlist = None
        if allow_basic_punct:
            letters_digits = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            basic_punct = " .,:;!?\'\"-–—_/&%$@#()+*=“”‘’`~^|\\"
            brackets = "[]{}<>" if allow_brackets else ""
            allowlist = letters_digits + basic_punct + brackets

        # OCR
        results = reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            width_ths=0.7,
            height_ths=0.7,
            slope_ths=0.1,
            ycenter_ths=0.5,
            add_margin=0.1,
            decoder=ocr_decoder,
            beamWidth=beam_width,
            allowlist=allowlist,
            batch_size=1,
            workers=0,
        )

        detections = self._to_detections(results, min_confidence, include_bboxes, include_polygons)
        # Cleanup
        if self.use_gpu:
            gc.collect()
            torch.cuda.empty_cache()

        # Create markdown output preserving original formatting as much as possible
        markdown_content = ""
        
        if text_only:
            markdown_content = "\n".join(d.text for d in detections)
        else:
            # Simple approach: reconstruct text with basic markdown formatting
            # Sort regions by Y position to maintain reading order
            for d in detections:
                text = d.text.strip()
                if not text:
                    continue
                    
                # Basic heuristics for markdown formatting based on text characteristics
                if len(text) < 50 and (text.isupper() or any(word in text.upper() for word in ['CHAPTER', 'SECTION', 'PART'])):
                    # Likely a header - make it a markdown header
                    if len(text) < 20:
                        markdown_content += f"# {text}\n\n"
                    else:
                        markdown_content += f"## {text}\n\n"
                elif text.endswith((':', '.')):
                    # Complete sentences/paragraphs
                    markdown_content += f"{text}\n\n"
                else:
                    # Incomplete text or continuation
                    markdown_content += f"{text} "
        
        # Clean up extra whitespace
        markdown_content = markdown_content.strip()
        
        # Write to markdown file
        out_file = Path("extracted_text.md")
        out_file.write_text(markdown_content, encoding='utf-8')
        
        # Create metadata
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
        
        return ModelOutput(
            markdown=out_file,
            metadata=json.dumps(metadata, indent=2)
        )
