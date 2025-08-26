"""
Screenshot OCR with EasyOCR
Concise, readable implementation for production Cog deployment.
"""

from typing import Any, Dict, List, Optional
import gc

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


DEFAULT_LANGS = ["en", "es", "fr", "de", "it", "pt"]




class Predictor(BasePredictor):
    """Production-grade screenshot OCR with minimal, clear code."""

    def setup(self) -> None:
        # Use GPU if available; this runs once per container.
        self.use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(DEFAULT_LANGS, gpu=self.use_gpu, verbose=False)
        # Warm up to avoid first-request latency.
        img = np.full((60, 160, 3), 255, np.uint8)
        cv2.putText(img, "TEST", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        _ = self.reader.readtext(img, detail=0)

    # ----- Preprocessing -----
    def _preprocess(self, img: np.ndarray, enable: bool) -> np.ndarray:
        """Simple, fast preprocessing that helps most screenshots.
        - Upscale small images (caps at 3x)
        - Optional CLAHE and light denoising
        """
        h, w = img.shape[:2]
        if min(h, w) < 600:  # DPI upsample for small screenshots
            scale = min(3.0, 600.0 / max(1.0, float(min(h, w))))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        if not enable:
            return img

        # Contrast enhancement (CLAHE) on grayscale then back to BGR
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Light denoising to clean compression artifacts
        if img.ndim == 3:
            img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
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

    def _to_detections(self, results: List, min_conf: float, include_bboxes: bool, include_polygons: bool) -> Any:
        out: List[TextRegion] = []
        for bbox, text, conf in results:
            if not text or conf < min_conf:
                continue
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            bbox_vals = (min(xs), min(ys), max(xs), max(ys)) if include_bboxes else (None, None, None, None)
            poly = None
            if include_polygons:
                poly = []
                for i in range(4):
                    poly.extend([xs[i], ys[i]])
            out.append(TextRegion(
                text=text.strip(),
                confidence=float(conf),
                x1=bbox_vals[0], y1=bbox_vals[1], x2=bbox_vals[2], y2=bbox_vals[3],
                polygon=poly
            ))
        return out

    # ----- Prediction -----
    def predict(
        self,
        image: Path = Input(description="Screenshot or image file"),
        languages: str = Input(description="Comma-separated language codes. Empty = defaults", default=""),
        min_confidence: float = Input(description="Minimum confidence (0.0-1.0)", default=0.25, ge=0.0, le=1.0),
        preprocessing: bool = Input(description="Apply preprocessing (recommended)", default=True),
        text_only: bool = Input(description="Return only text lines (list of strings)", default=False),
        include_bboxes: bool = Input(description="Include x1,y1,x2,y2 in output", default=True),
        include_polygons: bool = Input(description="Include 4-point polygon as flat list", default=True),
    ) -> Any:
        # Load and preprocess image
        arr = self._load_image(image)
        processed = self._preprocess(arr, preprocessing)

        # Use specified languages if provided
        if languages.strip():
            langs = [s.strip() for s in languages.split(",") if s.strip()]
            reader = easyocr.Reader(langs, gpu=self.use_gpu, verbose=False)
        else:
            langs = DEFAULT_LANGS
            reader = self.reader

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
        )

        detections = self._to_detections(results, min_confidence, include_bboxes, include_polygons)
        # Cleanup
        if self.use_gpu:
            gc.collect()
            torch.cuda.empty_cache()

        if text_only:
            return [r.text for r in detections]
        return [r.model_dump(exclude_none=True) for r in detections]
