"""
OCR Module
Implements Step 3: Character extraction using Tesseract OCR
Extracts text from aligned license plate images
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Optional


class PlateOCR:
    """Extracts text from license plate images using Tesseract OCR"""
    
    def __init__(self):
        # Tesseract configuration for license plates
        # --psm 7: Treat image as single line
        # --oem 3: Default OCR Engine Mode
        # -c tessedit_char_whitelist: Restrict to alphanumeric characters
        self.config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
    def extract_text(self, aligned_plate: np.ndarray) -> Optional[str]:
        """
        Extract text from aligned plate image
        
        Args:
            aligned_plate: Preprocessed plate image
            
        Returns:
            Extracted text string or None if extraction fails
        """
        if aligned_plate is None or aligned_plate.size == 0:
            return None
        
        try:
            # Preprocess for better OCR
            processed = self._preprocess_for_ocr(aligned_plate)
            
            # Run Tesseract OCR
            text = pytesseract.image_to_string(processed, config=self.config)
            
            # Clean the extracted text
            cleaned_text = self._clean_text(text)
            
            return cleaned_text if cleaned_text else None
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return None
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image specifically for OCR
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply binary threshold
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to connect broken characters
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Resize for better OCR (Tesseract works better with larger text)
        scale_factor = 2
        height, width = morph.shape
        resized = cv2.resize(morph, (width * scale_factor, height * scale_factor),
                           interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def _clean_text(self, text: str) -> str:
        """
        Clean OCR output text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove whitespace and newlines
        cleaned = text.strip().replace(" ", "").replace("\n", "")
        
        # Remove special characters except alphanumeric
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned.upper())
        
        # Common OCR corrections for license plates
        # O (letter) vs 0 (zero), I (letter) vs 1 (one), etc.
        corrections = {
            # These corrections depend on position in plate
            # For Rwanda plates: typically 3 letters + 3 digits + 1 letter
        }
        
        # Apply corrections based on pattern
        # This is context-dependent and should be adjusted
        
        return cleaned
    
    def extract_with_confidence(self, aligned_plate: np.ndarray) -> tuple:
        """
        Extract text with confidence scores
        
        Args:
            aligned_plate: Preprocessed plate image
            
        Returns:
            Tuple of (text, confidence)
        """
        if aligned_plate is None or aligned_plate.size == 0:
            return None, 0.0
        
        try:
            processed = self._preprocess_for_ocr(aligned_plate)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(processed, config=self.config, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 0:  # Valid detection
                    text = data['text'][i]
                    if text.strip():
                        texts.append(text)
                        confidences.append(float(conf))
            
            if texts and confidences:
                full_text = ''.join(texts)
                cleaned = self._clean_text(full_text)
                avg_confidence = sum(confidences) / len(confidences)
                return cleaned, avg_confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"OCR with confidence error: {e}")
            return None, 0.0
    
    def visualize_ocr(self, image: np.ndarray, text: str) -> np.ndarray:
        """
        Create visualization of OCR result
        
        Args:
            image: Original image
            text: Extracted text
            
        Returns:
            Image with text overlay
        """
        if image is None or image.size == 0:
            return None
        
        result = image.copy()
        
        # Add text overlay
        if text:
            # Add background rectangle for text
            cv2.rectangle(result, (10, 10), (390, 50), (0, 0, 0), -1)
            cv2.rectangle(result, (10, 10), (390, 50), (0, 255, 0), 2)
            
            # Add text
            cv2.putText(result, text, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # No text detected
            cv2.putText(result, "NO TEXT", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result


def test_ocr():
    """Test the OCR module"""
    print("Testing Plate OCR...")
    
    # Check if Tesseract is installed
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
    except Exception as e:
        print(f"✗ Tesseract not found: {e}")
        print("  Please install Tesseract OCR")
        return False
    
    # Create test image with text
    test_img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "RAD123B", (50, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    ocr = PlateOCR()
    
    # Test text extraction
    text = ocr.extract_text(test_img)
    if text:
        print(f"✓ Text extracted: {text}")
    else:
        print("⚠ No text extracted (may be normal for test image)")
    
    # Test with confidence
    text, conf = ocr.extract_with_confidence(test_img)
    if text:
        print(f"✓ Text with confidence: {text} ({conf:.1f}%)")
    
    return True


if __name__ == "__main__":
    test_ocr()