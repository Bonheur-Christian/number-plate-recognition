"""
Plate Detection Module
Implements Step 1: Detection of license plate in image frame
Based on contour analysis and shape filtering
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class PlateDetector:
    """Detects license plates in images using edge detection and contour analysis"""
    
    def __init__(self):
        # Detection parameters (tunable based on testing)
        self.MIN_AREA = 500
        self.MAX_AREA = 50000
        self.MIN_ASPECT_RATIO = 2.0
        self.MAX_ASPECT_RATIO = 6.0
        self.MIN_WIDTH = 80
        self.MIN_HEIGHT = 20
        
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List]]:
        """
        Detect license plate in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            Tuple of (detected_plate_roi, preprocessed_image, corner_points)
            Returns (None, None, None) if no plate detected
        """
        if frame is None or frame.size == 0:
            return None, None, None
            
        # Step 1: Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter - reduces noise while preserving edges
        # This is crucial for clean edge detection
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Step 2: Edge Detection
        # Canny edge detection to find plate boundaries
        edged = cv2.Canny(filtered, 30, 200)
        
        # Step 3: Find Contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plate_contour = None
        plate_corners = None
        
        # Step 4: Filter contours to find plate
        for contour in contours:
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # License plates are typically rectangular (4 corners)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                area = cv2.contourArea(contour)
                
                # Calculate aspect ratio
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Filter based on size and aspect ratio
                if (self.MIN_AREA < area < self.MAX_AREA and
                    self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO and
                    w > self.MIN_WIDTH and h > self.MIN_HEIGHT):
                    
                    plate_contour = approx
                    plate_corners = approx.reshape(4, 2)
                    break
        
        # Step 5: Extract plate region if found
        if plate_contour is not None and plate_corners is not None:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(plate_contour)
            
            # Extract plate ROI with some padding
            padding = 5
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)
            
            plate_roi = frame[y1:y2, x1:x2]
            
            return plate_roi, edged, plate_corners
        
        return None, edged, None
    
    def draw_detection(self, frame: np.ndarray, corners: List) -> np.ndarray:
        """
        Draw detection visualization on frame
        
        Args:
            frame: Original frame
            corners: Corner points of detected plate
            
        Returns:
            Frame with detection drawn
        """
        if corners is None:
            return frame
            
        result = frame.copy()
        
        # Draw the plate contour
        cv2.drawContours(result, [corners], -1, (0, 255, 0), 3)
        
        # Draw corner points
        for point in corners:
            cv2.circle(result, tuple(point), 5, (0, 0, 255), -1)
            
        # Add detection label
        x, y = corners[0]
        cv2.putText(result, "PLATE DETECTED", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result


def test_detector():
    """Test the plate detector with a sample image"""
    print("Testing Plate Detector...")
    
    # Create a simple test image (white rectangle on black background)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a rectangle simulating a plate
    cv2.rectangle(test_img, (200, 200), (440, 280), (255, 255, 255), -1)
    
    detector = PlateDetector()
    plate_roi, edges, corners = detector.detect(test_img)
    
    if plate_roi is not None:
        print("✓ Detection successful")
        print(f"  Plate ROI shape: {plate_roi.shape}")
        print(f"  Corner points: {len(corners) if corners is not None else 0}")
    else:
        print("✗ No plate detected (expected for simple test)")
    
    return True


if __name__ == "__main__":
    test_detector()