"""
Main ANPR Pipeline
Integrates detection, alignment, OCR, and validation
Handles camera capture and real-time processing
"""

import cv2
import numpy as np
import csv
import os
import argparse
from datetime import datetime
from typing import Optional

from detect import PlateDetector
from align import PlateAligner
from ocr import PlateOCR
from validate import PlateValidator, PlateConfirmation


class ANPRSystem:
    """Complete Automatic Number Plate Recognition System"""
    
    def __init__(self, camera_index: int = 0, save_directory: str = "../data"):
        """
        Initialize ANPR system
        
        Args:
            camera_index: Camera device index (0 for default, 1 for external USB camera)
            save_directory: Directory to save results
        """
        self.camera_index = camera_index
        self.save_directory = save_directory
        
        # Initialize components
        self.detector = PlateDetector()
        self.aligner = PlateAligner()
        self.ocr = PlateOCR()
        self.validator = PlateValidator()
        self.confirmation = PlateConfirmation(required_confirmations=5)
        
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        # CSV file for saving plates
        self.csv_file = os.path.join(save_directory, "plates.csv")
        self._initialize_csv()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'plates_detected': 0,
            'plates_confirmed': 0,
            'ocr_attempts': 0,
            'ocr_successes': 0
        }
        
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Plate', 'Confidence', 'Valid'])
    
    def save_plate(self, plate_text: str, confidence: float, valid: bool):
        """
        Save confirmed plate to CSV
        
        Args:
            plate_text: The plate text
            confidence: OCR confidence score
            valid: Whether plate passed validation
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, plate_text, f"{confidence:.2f}", valid])
        
        print(f"✓ Plate saved: {plate_text}")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Dictionary with processing results
        """
        self.stats['frames_processed'] += 1
        
        result = {
            'frame': frame,
            'detection_viz': None,
            'aligned_plate': None,
            'ocr_result': None,
            'validation': None,
            'confirmation_status': None,
            'plate_detected': False,
            'plate_confirmed': False
        }
        
        # Step 1: Detection
        plate_roi, edges, corners = self.detector.detect(frame)
        
        if plate_roi is not None and corners is not None:
            result['plate_detected'] = True
            self.stats['plates_detected'] += 1
            
            # Visualize detection
            result['detection_viz'] = self.detector.draw_detection(frame.copy(), corners)
            
            # Step 2: Alignment
            # Get full frame corners for perspective transform
            # We need to find the corners in the original frame, not the ROI
            aligned_plate = self.aligner.align(frame, corners)
            
            if aligned_plate is not None:
                result['aligned_plate'] = aligned_plate
                
                # Preprocess for OCR
                preprocessed = self.aligner.preprocess_for_ocr(aligned_plate)
                
                # Step 3: OCR
                self.stats['ocr_attempts'] += 1
                plate_text, confidence = self.ocr.extract_with_confidence(preprocessed)
                
                if plate_text:
                    self.stats['ocr_successes'] += 1
                    
                    # Step 4: Validation
                    validation = self.validator.validate(plate_text)
                    result['validation'] = validation
                    
                    if validation['valid']:
                        # Apply corrections
                        corrected_plate = self.validator.format_plate(plate_text)
                        
                        # Step 5: Confirmation
                        confirmation_status = self.confirmation.add_detection(corrected_plate)
                        result['confirmation_status'] = confirmation_status
                        
                        # Check if confirmed
                        if confirmation_status['confirmed']:
                            result['plate_confirmed'] = True
                            confirmed_plate = self.confirmation.get_confirmed_plate()
                            
                            # Save to file (only once)
                            if confirmation_status['count'] == self.confirmation.required_confirmations:
                                self.save_plate(confirmed_plate, confidence, True)
                                self.stats['plates_confirmed'] += 1
                        
                        result['ocr_result'] = {
                            'text': corrected_plate,
                            'confidence': confidence
                        }
                    else:
                        # Invalid plate format
                        result['ocr_result'] = {
                            'text': plate_text,
                            'confidence': confidence,
                            'errors': validation['errors']
                        }
        
        return result
    
    def create_display(self, result: dict) -> np.ndarray:
        """
        Create comprehensive display with all pipeline stages
        
        Args:
            result: Processing result dictionary
            
        Returns:
            Combined display image
        """
        frame = result['frame']
        h, w = frame.shape[:2]
        
        # Create main display area
        if result['detection_viz'] is not None:
            display = result['detection_viz'].copy()
        else:
            display = frame.copy()
        
        # Add status overlay
        self._add_status_overlay(display, result)
        
        # Create side panels for aligned plate and info
        if result['aligned_plate'] is not None:
            # Resize aligned plate for display
            plate_display = cv2.resize(result['aligned_plate'], (400, 100))
            
            # Create info panel
            info_panel = self._create_info_panel(result, 400, 300)
            
            # Stack vertically
            right_panel = np.vstack([plate_display, info_panel])
            
            # Combine with main display
            # Resize main display to match height
            target_height = right_panel.shape[0]
            aspect = w / h
            target_width = int(target_height * aspect)
            display_resized = cv2.resize(display, (target_width, target_height))
            
            # Combine horizontally
            final_display = np.hstack([display_resized, right_panel])
        else:
            final_display = display
        
        return final_display
    
    def _add_status_overlay(self, frame: np.ndarray, result: dict):
        """Add status information overlay to frame"""
        y_offset = 30
        
        # Detection status
        if result['plate_detected']:
            cv2.putText(frame, "PLATE DETECTED", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SCANNING...", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        y_offset += 30
        
        # OCR result
        if result['ocr_result']:
            text = result['ocr_result']['text']
            conf = result['ocr_result']['confidence']
            cv2.putText(frame, f"OCR: {text} ({conf:.0f}%)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Confirmation status
        if result['confirmation_status']:
            status = result['confirmation_status']
            color = (0, 255, 0) if status['confirmed'] else (255, 255, 0)
            cv2.putText(frame, f"Confirm: {status['progress']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if status['confirmed']:
                y_offset += 30
                cv2.putText(frame, "CONFIRMED!", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _create_info_panel(self, result: dict, width: int, height: int) -> np.ndarray:
        """Create information panel with processing details"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_offset = 30
        line_height = 25
        
        # Title
        cv2.putText(panel, "ANPR Pipeline Status", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height * 2
        
        # OCR Result
        if result['ocr_result']:
            ocr = result['ocr_result']
            cv2.putText(panel, f"Text: {ocr['text']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(panel, f"Confidence: {ocr['confidence']:.1f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Validation
        if result['validation']:
            val = result['validation']
            status = "VALID" if val['valid'] else "INVALID"
            color = (0, 255, 0) if val['valid'] else (0, 0, 255)
            cv2.putText(panel, f"Validation: {status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height
            
            if val['errors']:
                for error in val['errors'][:2]:  # Show max 2 errors
                    cv2.putText(panel, f"  - {error}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    y_offset += line_height
        
        # Statistics
        y_offset += line_height
        cv2.putText(panel, "Statistics:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        
        stats_text = [
            f"Frames: {self.stats['frames_processed']}",
            f"Detected: {self.stats['plates_detected']}",
            f"Confirmed: {self.stats['plates_confirmed']}",
            f"OCR Rate: {self.stats['ocr_successes']}/{self.stats['ocr_attempts']}"
        ]
        
        for text in stats_text:
            cv2.putText(panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += line_height
        
        return panel
    
    def run(self):
        """Run the ANPR system with camera capture"""
        print("="*60)
        print("ANPR System Starting")
        print("="*60)
        print(f"Camera Index: {self.camera_index}")
        print(f"Save Directory: {self.save_directory}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset confirmation")
        print("="*60)
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"✗ Error: Cannot open camera {self.camera_index}")
            print("  Try different camera index (0, 1, 2, ...)")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Camera opened successfully")
        print("\nPress 'q' to quit\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("✗ Error: Cannot read frame")
                    break
                
                frame_count += 1
                
                # Process frame
                result = self.process_frame(frame)
                
                # Create display
                display = self.create_display(result)
                
                # Show result
                cv2.imshow('ANPR System', display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.save_directory, f"frame_{timestamp}.jpg")
                    cv2.imwrite(filename, display)
                    print(f"✓ Frame saved: {filename}")
                elif key == ord('r'):
                    # Reset confirmation
                    self.confirmation.reset()
                    print("✓ Confirmation reset")
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames | " +
                          f"Detected: {self.stats['plates_detected']} | " +
                          f"Confirmed: {self.stats['plates_confirmed']}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "="*60)
            print("Session Statistics")
            print("="*60)
            for key, value in self.stats.items():
                print(f"  {key}: {value}")
            print("="*60)
            print(f"Results saved to: {self.csv_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ANPR - Automatic Number Plate Recognition')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (0=default, 1=external USB camera)')
    parser.add_argument('--save-dir', type=str, default='../data',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create and run ANPR system
    anpr = ANPRSystem(camera_index=args.camera, save_directory=args.save_dir)
    anpr.run()


if __name__ == "__main__":
    main()