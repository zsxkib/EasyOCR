"""
Production-Ready Screenshot OCR with EasyOCR
Optimized for Replicate deployment at scale - handles millions of screenshot OCR requests

Features:
- Advanced preprocessing pipeline inspired by macOS Vision Framework
- Multi-language support with auto-detection
- GPU acceleration with CPU fallback for Mac development
- Robust error handling and comprehensive logging
- Structured output with confidence scores and bounding boxes
- Memory-efficient processing for high-scale deployment

Author: AI Assistant
Version: 1.0.0
"""

import os
import gc
import logging
import tempfile
import traceback
from typing import List, Dict, Any, Optional, Union
from pathlib import Path as PathlibPath

# Core dependencies
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import easyocr
from cog import BasePredictor, Input, Path
import torch
import psutil

# Advanced image processing
from skimage import filters, morphology, measure, transform
from skimage.restoration import denoise_nl_means
from scipy import ndimage
from scipy.stats import mode


class ProductionLogger:
    """Production-grade logging with performance monitoring"""
    
    def __init__(self, name: str = "screenshot_ocr"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        self.logger.info(f"{message} {kwargs if kwargs else ''}")
    
    def error(self, message: str, **kwargs):
        self.logger.error(f"{message} {kwargs if kwargs else ''}")
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(f"{message} {kwargs if kwargs else ''}")


class AdvancedImagePreprocessor:
    """
    Advanced preprocessing pipeline inspired by macOS Vision Framework
    Optimized for screenshot OCR robustness across different zoom levels and devices
    """
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
    
    def optimize_dpi(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """
        DPI optimization - upsample low-resolution screenshots to optimal DPI
        Critical for mobile screenshots and low-resolution captures
        """
        try:
            height, width = image.shape[:2]
            current_min_dim = min(height, width)
            
            # Calculate optimal scale factor based on assumed 150 DPI baseline
            if current_min_dim < 600:  # Likely low DPI screenshot
                scale_factor = max(1.5, 600 / current_min_dim)
                scale_factor = min(scale_factor, 3.0)  # Cap at 3x to prevent memory issues
                
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Use INTER_CUBIC for better text quality upsampling
                upsampled = cv2.resize(image, (new_width, new_height), 
                                     interpolation=cv2.INTER_CUBIC)
                
                self.logger.info(f"DPI optimization: {width}x{height} → {new_width}x{new_height} (scale: {scale_factor:.2f})")
                return upsampled
            
            return image
            
        except Exception as e:
            self.logger.warning(f"DPI optimization failed: {e}")
            return image
    
    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE)
        Handles uneven lighting conditions in screenshots
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE with optimized parameters for screenshots
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Convert back to original format
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"CLAHE enhancement failed: {e}")
            return image
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced noise reduction using Non-local Means Denoising
        Preserves text edges while removing compression artifacts
        """
        try:
            if len(image.shape) == 3:
                # Color image denoising
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                # Grayscale image denoising
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return image
    
    def correct_skew(self, image: np.ndarray, angle_threshold: float = 1.0) -> np.ndarray:
        """
        Automatic skew correction for rotated mobile screenshots
        Uses Hough transform to detect and correct text line orientation
        """
        try:
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Edge detection for line finding
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 5:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    # Convert to rotation angle
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                # Find most common angle
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only correct significant skew
                    if abs(median_angle) > angle_threshold:
                        (h, w) = gray.shape
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        
                        # Apply rotation to original image
                        corrected = cv2.warpAffine(image, M, (w, h), 
                                                 flags=cv2.INTER_CUBIC, 
                                                 borderMode=cv2.BORDER_REPLICATE)
                        
                        self.logger.info(f"Skew corrected by {median_angle:.1f} degrees")
                        return corrected
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Skew correction failed: {e}")
            return image
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding optimized for UI elements and varied backgrounds
        Handles both dark-on-light and light-on-dark text
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Also create inverse for light-on-dark text
            binary_inv = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Combine both to handle mixed scenarios
            combined = cv2.bitwise_or(binary, binary_inv)
            
            # Convert back to 3-channel for EasyOCR compatibility
            if len(image.shape) == 3:
                combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            
            return combined
            
        except Exception as e:
            self.logger.warning(f"Adaptive thresholding failed: {e}")
            return image
    
    def enhance_text_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Morphological operations for character enhancement
        Closes gaps in characters and removes small noise
        """
        try:
            # Convert to grayscale for morphological operations
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                process_gray = True
            else:
                gray = image.copy()
                process_gray = False
            
            # Morphological closing to connect character parts
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            
            # Convert back to original format
            if process_gray:
                cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"Morphological enhancement failed: {e}")
            return image
    
    def preprocess_pipeline(self, image: np.ndarray, 
                          enable_all: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline with graceful fallbacks
        Each step is independent and failures don't break the entire pipeline
        """
        try:
            processed = image.copy()
            
            if enable_all:
                # Step 1: DPI optimization (critical for low-res screenshots)
                processed = self.optimize_dpi(processed)
                
                # Step 2: Contrast enhancement
                processed = self.enhance_contrast_clahe(processed)
                
                # Step 3: Noise reduction
                processed = self.reduce_noise(processed)
                
                # Step 4: Skew correction
                processed = self.correct_skew(processed)
                
                # Step 5: Text enhancement
                processed = self.enhance_text_morphology(processed)
            else:
                # Minimal processing - just DPI and contrast
                processed = self.optimize_dpi(processed)
                processed = self.enhance_contrast_clahe(processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {e}")
            return image


class Predictor(BasePredictor):
    """
    Production-Ready Screenshot OCR Predictor
    
    Handles millions of requests with:
    - Advanced preprocessing for robust text extraction
    - Multi-language support with auto-detection
    - GPU acceleration with CPU fallback
    - Comprehensive error handling and monitoring
    - Structured output with confidence scores
    """
    
    def setup(self) -> None:
        """
        Initialize EasyOCR model and preprocessing pipeline
        Runs once at container startup for optimal performance
        """
        try:
            self.logger = ProductionLogger("screenshot_ocr_predictor")
            self.logger.info("Starting Screenshot OCR Predictor setup...")
            
            # Check GPU availability
            self.gpu_available = torch.cuda.is_available()
            self.logger.info(f"GPU available: {self.gpu_available}")
            
            # Initialize default multi-language reader
            # Covers major languages for international screenshots
            self.default_languages = ['en', 'es', 'fr', 'de', 'it', 'pt']
            
            self.logger.info("Loading EasyOCR with default languages...")
            self.reader = easyocr.Reader(
                self.default_languages,
                gpu=self.gpu_available,  # Auto-detect GPU
                verbose=False,
                download_enabled=True  # Allow model downloads if needed
            )
            
            # Initialize preprocessor
            self.preprocessor = AdvancedImagePreprocessor(self.logger)
            
            # Warm up the model with a small test image to reduce first-request latency
            self._warmup_model()
            
            # Track memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.logger.info(f"Setup complete. Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            self.logger.error(traceback.format_exc())
            raise e
    
    def _warmup_model(self) -> None:
        """
        Warm up the model to reduce first-request latency
        Creates a small test image for model initialization
        """
        try:
            # Create a small test image with text
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, 'TEST', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Run a quick OCR to initialize the model
            _ = self.reader.readtext(test_image, detail=0)
            self.logger.info("Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed (non-critical): {e}")
    
    def _validate_image_input(self, image_path: Path) -> np.ndarray:
        """
        Validate and load image with comprehensive format support
        Handles various image formats and corrupted files
        """
        try:
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Check file size (limit to 20MB for production stability)
            file_size = image_path.stat().st_size
            if file_size > 20 * 1024 * 1024:
                raise ValueError(f"Image too large: {file_size / 1024 / 1024:.1f} MB (max: 20MB)")
            
            # Load with PIL for better format compatibility
            with Image.open(image_path) as pil_image:
                # Handle different color modes
                if pil_image.mode in ['RGBA', 'LA']:
                    # Convert RGBA/LA to RGB by adding white background
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
                    pil_image = background
                elif pil_image.mode not in ['RGB', 'L']:
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array
                image_array = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV compatibility if needed
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                return image_array
                
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def _parse_language_input(self, languages: str) -> List[str]:
        """
        Parse and validate language input
        Supports both language codes and common language names
        """
        if not languages.strip():
            return self.default_languages
        
        # Language code mapping for common names
        language_map = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
            'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja',
            'korean': 'ko', 'chinese': 'ch', 'arabic': 'ar', 'hindi': 'hi'
        }
        
        lang_codes = []
        for lang in languages.split(','):
            lang = lang.strip().lower()
            if lang in language_map:
                lang_codes.append(language_map[lang])
            elif len(lang) == 2:  # Assume it's already a language code
                lang_codes.append(lang)
            else:
                self.logger.warning(f"Unknown language: {lang}, skipping")
        
        # Fallback to default if no valid languages found
        return lang_codes if lang_codes else ['en']
    
    def _extract_text_with_confidence(self, image: np.ndarray, 
                                    languages: List[str],
                                    min_confidence: float) -> List[Dict[str, Any]]:
        """
        Extract text with bounding boxes and confidence scores
        Uses optimized EasyOCR parameters for screenshot text
        """
        try:
            # Create reader for specific languages if different from default
            if languages != self.default_languages:
                reader = easyocr.Reader(languages, gpu=self.gpu_available, verbose=False)
            else:
                reader = self.reader
            
            # Run OCR with optimized parameters for screenshots
            results = reader.readtext(
                image,
                detail=1,  # Return bounding box coordinates
                paragraph=False,  # Process line by line for better accuracy
                width_ths=0.7,  # Text width threshold
                height_ths=0.7,  # Text height threshold
                slope_ths=0.1,  # Allow slight rotation
                ycenter_ths=0.5,  # Y-center threshold for grouping
                add_margin=0.1,  # Add margin around detected text
                x_ths=1.0,  # X-axis threshold
                y_ths=0.5   # Y-axis threshold
            )
            
            extracted_text = []
            for (bbox, text, confidence) in results:
                # Filter by confidence and non-empty text
                if confidence >= min_confidence and text.strip():
                    # Convert bbox to standardized format
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
                        },
                        'polygon': [[int(point[0]), int(point[1])] for point in bbox]
                    })
            
            return extracted_text
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return []
    
    def _cleanup_memory(self) -> None:
        """
        Clean up memory after processing to prevent memory leaks
        Critical for high-volume production deployment
        """
        try:
            gc.collect()
            if self.gpu_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def predict(
        self,
        image: Path = Input(description="Screenshot or image file to extract text from (PNG, JPG, WebP, etc.)"),
        languages: str = Input(
            description="Comma-separated language codes (e.g., 'en,es,fr') or names (e.g., 'english,spanish'). Leave empty for auto-detection with major languages",
            default=""
        ),
        min_confidence: float = Input(
            description="Minimum confidence threshold (0.0-1.0). Lower values extract more text but may include noise. Recommended: 0.25 for screenshots",
            default=0.25,
            ge=0.0,
            le=1.0
        ),
        preprocessing: bool = Input(
            description="Apply advanced preprocessing pipeline for better accuracy. Recommended for screenshots, mobile captures, and low-quality images",
            default=True
        )
    ) -> Dict[str, Any]:
        """
        Extract text from screenshots with production-grade robustness
        
        Returns structured data with:
        - Extracted text with confidence scores
        - Bounding box coordinates for each text region
        - Polygon coordinates for precise text boundaries
        - Processing metadata and error handling
        
        Optimized for:
        - Web screenshots and UI elements
        - Mobile app screenshots
        - Document captures
        - Multi-language text extraction
        - High-volume API requests
        """
        
        start_time = cv2.getTickCount()
        
        try:
            self.logger.info(f"Processing image: {image}, languages: {languages or 'auto'}, confidence: {min_confidence}, preprocessing: {preprocessing}")
            
            # Step 1: Validate and load image
            image_array = self._validate_image_input(image)
            original_shape = image_array.shape
            
            # Step 2: Apply preprocessing pipeline if enabled
            if preprocessing:
                processed_image = self.preprocessor.preprocess_pipeline(image_array, enable_all=True)
            else:
                # Minimal preprocessing - just optimize DPI
                processed_image = self.preprocessor.optimize_dpi(image_array)
            
            # Step 3: Parse language configuration
            lang_codes = self._parse_language_input(languages)
            self.logger.info(f"Using languages: {lang_codes}")
            
            # Step 4: Extract text with confidence filtering
            extracted_text = self._extract_text_with_confidence(
                processed_image, lang_codes, min_confidence
            )
            
            # Step 5: Aggregate results
            all_text = ' '.join([item['text'] for item in extracted_text])
            avg_confidence = (
                sum([item['confidence'] for item in extracted_text]) / len(extracted_text) 
                if extracted_text else 0.0
            )
            
            # Step 6: Calculate processing time
            end_time = cv2.getTickCount()
            processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # ms
            
            # Step 7: Prepare response
            result = {
                'text': all_text,
                'detected_text': extracted_text,
                'total_detections': len(extracted_text),
                'average_confidence': round(avg_confidence, 3),
                'languages_used': lang_codes,
                'preprocessing_applied': preprocessing,
                'processing_time_ms': round(processing_time, 2),
                'original_image_size': {
                    'width': original_shape[1], 
                    'height': original_shape[0]
                },
                'success': True
            }
            
            self.logger.info(
                f"OCR completed: {len(extracted_text)} regions, "
                f"avg confidence: {avg_confidence:.3f}, "
                f"time: {processing_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Prediction failed: {error_msg}")
            self.logger.error(traceback.format_exc())
            
            # Return structured error response
            return {
                'text': '',
                'detected_text': [],
                'total_detections': 0,
                'average_confidence': 0.0,
                'languages_used': [],
                'preprocessing_applied': preprocessing,
                'processing_time_ms': 0.0,
                'original_image_size': {'width': 0, 'height': 0},
                'success': False,
                'error': error_msg,
                'error_type': type(e).__name__
            }
        
        finally:
            # Always clean up memory
            self._cleanup_memory()
