"""
Plate Alignment Module
Implements Step 2: Perspective correction and alignment
Transforms skewed plate into rectangular form
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class PlateAligner:
    """Aligns license plates by correcting perspective distortion"""
    
    def __init__(self):
        # Target dimensions for aligned plate
        self.TARGET_WIDTH = 400
        self.TARGET_HEIGHT = 100
        
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in consistent manner: top-left, top-right, bottom-right, bottom-left
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        # Initialize ordered points
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left point has smallest sum
        # Bottom-right point has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference
        # Bottom-left point has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def calculate_dimensions(self, pts: np.ndarray) -> Tuple[int, int]:
        """
        Calculate appropriate dimensions for aligned plate
        
        Args:
            pts: Ordered corner points
            
        Returns:
            (width, height) tuple
        """
        (tl, tr, br, bl) = pts
        
        # Calculate width
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bottom))
        
        # Calculate height
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))
        
        return max_width, max_height
    
    def align(self, plate_roi: np.ndarray, corners: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Align the plate by correcting perspective
        
        Args:
            plate_roi: Region of interest containing the plate
            corners: Optional corner points for perspective transform
                    If None, basic alignment is performed
            
        Returns:
            Aligned plate image
        """
        if plate_roi is None or plate_roi.size == 0:
            return None
        
        # If corners provided, do full perspective correction
        if corners is not None and len(corners) == 4:
            return self._perspective_transform(plate_roi, corners)
        else:
            # Basic alignment without perspective correction
            return self._basic_align(plate_roi)
    
    def _perspective_transform(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation using corner points
        
        Args:
            frame: Input image
            corners: 4 corner points of the plate
            
        Returns:
            Transformed (aligned) image
        """
        # Order the corner points
        rect = self.order_points(corners)
        
        # Calculate appropriate dimensions
        width, height = self.calculate_dimensions(rect)
        
        # Ensure reasonable dimensions
        if width < 50 or height < 20 or width > 2000 or height > 500:
            # Fall back to fixed dimensions if calculated ones are unreasonable
            width = self.TARGET_WIDTH
            height = self.TARGET_HEIGHT
        
        # Define destination points (corners of output image)
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply the transformation
        aligned = cv2.warpPerspective(frame, matrix, (width, height))
        
        return aligned
    
    def _basic_align(self, plate_roi: np.ndarray) -> np.ndarray:
        """
        Basic alignment without corner points
        Simply resize and preprocess
        
        Args:
            plate_roi: Plate region of interest
            
        Returns:
            Processed plate
        """
        # Resize to standard dimensions
        aligned = cv2.resize(plate_roi, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
        
        return aligned
    
    def preprocess_for_ocr(self, aligned_plate: np.ndarray) -> np.ndarray:
        """
        Further preprocessing to improve OCR accuracy
        
        Args:
            aligned_plate: Aligned plate image
            
        Returns:
            Preprocessed image ready for OCR
        """
        if aligned_plate is None or aligned_plate.size == 0:
            return None
        
        # Convert to grayscale if not already
        if len(aligned_plate.shape) == 3:
            gray = cv2.cvtColor(aligned_plate, cv2.COLOR_BGR2GRAY)
        else:
            gray = aligned_plate
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        # This works better than global thresholding for varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned


def test_aligner():
    """Test the plate aligner"""
    print("Testing Plate Aligner...")
    
    # Create test image with perspective distortion
    test_img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (250, 150), (255, 255, 255), -1)
    cv2.putText(test_img, "RAD123B", (70, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Define skewed corner points
    corners = np.array([
        [60, 60],   # top-left
        [240, 50],  # top-right
        [250, 140], # bottom-right
        [50, 150]   # bottom-left
    ], dtype="float32")
    
    aligner = PlateAligner()
    
    # Test with corners
    aligned = aligner.align(test_img, corners)
    if aligned is not None:
        print("✓ Alignment with corners successful")
        print(f"  Aligned shape: {aligned.shape}")
    else:
        print("✗ Alignment failed")
    
    # Test without corners
    aligned_basic = aligner.align(test_img)
    if aligned_basic is not None:
        print("✓ Basic alignment successful")
        print(f"  Aligned shape: {aligned_basic.shape}")
    
    # Test preprocessing
    preprocessed = aligner.preprocess_for_ocr(aligned)
    if preprocessed is not None:
        print("✓ Preprocessing successful")
        print(f"  Preprocessed shape: {preprocessed.shape}")
    
    return True


if __name__ == "__main__":
    test_aligner()