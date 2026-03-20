#!/usr/bin/env python3
"""
Main ANPR with Verbose Output
Same as main.py but prints detailed debug info without needing --debug flag
"""

import cv2
import numpy as np
import csv
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from src.detect import PlateDetector
from src.align import PlateAligner
from src.ocr import PlateOCR
from src.validate import PlateValidator, PlateConfirmation

def run_verbose():
    """Run ANPR with verbose debugging output"""
    
    print("="*60)
    print("ANPR System - VERBOSE DEBUG MODE")
    print("="*60)
    
    # Setup
    camera_index = 0
    save_dir = "data"
    debug_dir = os.path.join(save_dir, "debug")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Camera: {camera_index}")
    print(f"Save Dir: {save_dir}")
    print(f"Debug Dir: {debug_dir}")
    print("\nControls: 'q' = Quit, 's' = Save frame")
    print("="*60)
    
    # Initialize components
    detector = PlateDetector()
    aligner = PlateAligner()
    ocr = PlateOCR()
    validator = PlateValidator()
    confirmation = PlateConfirmation(required_confirmations=3)  # Lowered to 3
    
    # Stats
    stats = {
        'frames': 0,
        'detections': 0,
        'ocr_attempts': 0,
        'ocr_successes': 0,
        'validations_passed': 0,
        'confirmations': 0
    }
    
    # CSV setup
    csv_file = os.path.join(save_dir, "plates_verbose.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Plate', 'Confidence', 'Valid'])
    
    # Camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("✗ Cannot open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera opened\n")
    
    detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            stats['frames'] += 1
            
            # Detect
            plate_roi, edges, corners = detector.detect(frame)
            
            if plate_roi is not None and corners is not None:
                detection_count += 1
                stats['detections'] += 1
                
                print(f"\n{'='*60}")
                print(f"🔍 DETECTION #{detection_count}")
                print(f"{'='*60}")
                print(f"Frame: {stats['frames']}")
                print(f"ROI size: {plate_roi.shape[1]}x{plate_roi.shape[0]}")
                
                # Save detected region
                det_path = os.path.join(debug_dir, f"det_{detection_count}.jpg")
                cv2.imwrite(det_path, plate_roi)
                print(f"Saved: {det_path}")
                
                # Align
                aligned = aligner.align(frame, corners)
                
                if aligned is not None:
                    align_path = os.path.join(debug_dir, f"aligned_{detection_count}.jpg")
                    cv2.imwrite(align_path, aligned)
                    print(f"Saved: {align_path}")
                    
                    # Preprocess
                    preprocessed = aligner.preprocess_for_ocr(aligned)
                    
                    if preprocessed is not None:
                        prep_path = os.path.join(debug_dir, f"prep_{detection_count}.jpg")
                        cv2.imwrite(prep_path, preprocessed)
                        print(f"Saved: {prep_path}")
                        
                        # OCR
                        stats['ocr_attempts'] += 1
                        print("\n📝 Running OCR...")
                        
                        try:
                            plate_text, confidence = ocr.extract_with_confidence(preprocessed)
                            
                            print(f"Raw result: '{plate_text}'")
                            print(f"Confidence: {confidence:.1f}%")
                            
                            if plate_text and len(plate_text) >= 2:
                                stats['ocr_successes'] += 1
                                print(f"✓ OCR SUCCESS!")
                                
                                # Validate
                                print("\n🔍 Validating...")
                                validation = validator.validate(plate_text)
                                
                                print(f"Result: {validation['valid']}")
                                if validation['valid']:
                                    stats['validations_passed'] += 1
                                    print(f"✓ VALIDATION PASSED!")
                                    print(f"Pattern: {validation['pattern_matched']}")
                                    
                                    # Confirm
                                    print("\n⏳ Checking confirmation...")
                                    corrected = validator.format_plate(plate_text)
                                    conf_status = confirmation.add_detection(corrected)
                                    
                                    print(f"Progress: {conf_status['progress']}")
                                    
                                    if conf_status['confirmed']:
                                        stats['confirmations'] += 1
                                        print(f"\n🎉 PLATE CONFIRMED: {corrected}")
                                        
                                        # Save to CSV
                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        with open(csv_file, 'a', newline='') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([timestamp, corrected, f"{confidence:.2f}", True])
                                        print(f"✓ Saved to CSV")
                                else:
                                    print(f"✗ VALIDATION FAILED")
                                    print(f"Errors: {validation['errors']}")
                                    print(f"\n💡 TIP: This plate format isn't recognized.")
                                    print(f"   Edit src/validate.py to add this pattern,")
                                    print(f"   OR use validate_lenient.py")
                            else:
                                print(f"✗ OCR failed - no text extracted")
                                print(f"   Text too short: '{plate_text}'")
                        
                        except Exception as e:
                            print(f"✗ OCR ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Draw detection
                viz = detector.draw_detection(frame.copy(), corners)
                cv2.imshow('ANPR Verbose', viz)
            else:
                cv2.imshow('ANPR Verbose', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = os.path.join(save_dir, f"frame_{stats['frames']}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"✓ Saved frame: {save_path}")
            
            # Print stats every 100 frames
            if stats['frames'] % 100 == 0:
                print(f"\nStats @ {stats['frames']} frames:")
                print(f"  Detections: {stats['detections']}")
                print(f"  OCR success: {stats['ocr_successes']}/{stats['ocr_attempts']}")
                print(f"  Validations: {stats['validations_passed']}")
                print(f"  Confirmations: {stats['confirmations']}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("="*60)
        print(f"\nResults: {csv_file}")
        print(f"Debug images: {debug_dir}/")

if __name__ == "__main__":
    run_verbose()