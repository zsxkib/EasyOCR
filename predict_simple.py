"""
Simplified Screenshot OCR for Testing on Mac
Production features available in predict.py
"""

import os
import gc
import logging
import traceback
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image
import easyocr
from cog import BasePredictor, Input, Path
import torch


class Predictor(BasePredictor):
    """Simplified OCR Predictor for Mac testing"""
    
    def setup(self) -> None:
        """Initialize EasyOCR model"""
        try:
            print("Setting up EasyOCR...")
            
            # Check GPU availability
            self.gpu_available = torch.cuda.is_available()
            print(f"GPU available: {self.gpu_available}")
            
            # Initialize with English only for testing
            print("Loading EasyOCR with English...")
            self.reader = easyocr.Reader(
                ['en'],  # Just English for testing
                gpu=self.gpu_available,
                verbose=False
            )
            
            print("Setup complete!")
            
        except Exception as e:
            print(f"Setup failed: {e}")
            print(traceback.format_exc())
            raise e
    
    def predict(
        self,
        image: Path = Input(description="Image file to extract text from"),
        languages: str = Input(description="Language codes (e.g., 'en')", default="en"),
        min_confidence: float = Input(description="Minimum confidence (0.0-1.0)", default=0.25, ge=0.0, le=1.0),
        preprocessing: bool = Input(description="Apply basic preprocessing", default=True)
    ) -> Dict[str, Any]:
        """Extract text from images"""
        
        try:
            print(f"Processing image: {image}")
            
            # Load image
            pil_image = Image.open(image)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image_array = np.array(pil_image)
            
            # Basic preprocessing if enabled
            if preprocessing:
                # Simple DPI upscaling if image is small
                height, width = image_array.shape[:2]
                if min(height, width) < 600:
                    scale = 600 / min(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    print(f"Upscaled image: {width}x{height} -> {new_width}x{new_height}")
            
            # Run OCR
            print("Running OCR...")
            results = self.reader.readtext(image_array, detail=1)
            
            # Process results
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence >= min_confidence and text.strip():
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    extracted_text.append({
                        'text': text.strip(),
                        'confidence': float(confidence),
                        'bbox': {
                            'x1': int(min(x_coords)),
                            'y1': int(min(y_coords)),
                            'x2': int(max(x_coords)),
                            'y2': int(max(y_coords))
                        }
                    })
            
            all_text = ' '.join([item['text'] for item in extracted_text])
            avg_confidence = sum([item['confidence'] for item in extracted_text]) / len(extracted_text) if extracted_text else 0.0
            
            result = {
                'text': all_text,
                'detected_text': extracted_text,
                'total_detections': len(extracted_text),
                'average_confidence': round(avg_confidence, 3),
                'success': True
            }
            
            print(f"OCR completed: {len(extracted_text)} regions detected")
            return result
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return {
                'text': '',
                'detected_text': [],
                'total_detections': 0,
                'average_confidence': 0.0,
                'success': False,
                'error': str(e)
            }
        
        finally:
            gc.collect()
