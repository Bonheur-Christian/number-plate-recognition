# ANPR - Automatic Number Plate Recognition System

## Overview
This project implements a complete Number Plate Recognition pipeline based on the three-step approach:
1. **Detection** - Locate the license plate in the frame
2. **Alignment** - Correct perspective distortion
3. **OCR** - Extract characters using Tesseract

## Reference
Implementation based on:
**"Car Number Plate Extraction in Three Steps — Detection, Alignment, and OCR"**
By Gabriel Baziramwabo

## Features
- Real-time camera capture (supports external USB cameras)
- Automatic plate detection using contour analysis
- Perspective correction and alignment
- OCR using Tesseract
- Plate format validation
- Multi-frame confirmation (reduces false positives)
- Automatic saving of confirmed plates to CSV

## System Requirements
- Python 3.7+
- External USB camera or webcam
- Tesseract OCR installed on system

## Installation

### 1. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python src/main.py
```

### With Specific Camera
```bash
python src/main.py --camera 1
```

### Controls
- **Press 'q'** - Quit the application
- **Press 's'** - Save current frame
- **Press 'r'** - Reset confirmation counter

## Project Structure
```
anpr-project/
│
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── src/
│   ├── detect.py         # Plate detection logic
│   ├── align.py          # Perspective correction
│   ├── ocr.py            # Character extraction
│   ├── validate.py       # Plate format validation
│   └── main.py           # Main pipeline
│
├── data/
│   └── plates.csv        # Saved confirmed plates
│
└── screenshots/          # Example results
    ├── detection.png
    ├── alignment.png
    └── ocr.png
```

## Pipeline Explanation

### Step 1: Detection
- Convert frame to grayscale
- Apply bilateral filter to reduce noise while preserving edges
- Use Canny edge detection
- Find contours and filter by area and aspect ratio
- Select candidates with rectangular shape (4 corners)

### Step 2: Alignment
- Extract the four corner points of the detected plate
- Order points (top-left, top-right, bottom-right, bottom-left)
- Calculate destination dimensions
- Apply perspective transformation using cv2.getPerspectiveTransform()
- Warp image to obtain aligned, rectangular plate

### Step 3: OCR
- Preprocess aligned plate (grayscale, threshold, denoise)
- Apply Tesseract OCR with configuration optimized for license plates
- Parse and clean the extracted text
- Validate format against expected patterns

### Validation & Confirmation
- Check plate format (letters, numbers, length)
- Require multiple consecutive detections (default: 5 frames)
- Save only confirmed plates to avoid false positives

## Testing Results

### Test Vehicle 1
- **Detection**: ✓ Success
- **Alignment**: ✓ Success  
- **OCR Result**: RAD123B
- **Validation**: ✓ Passed
- **Frames to Confirm**: 5/5

### Test Vehicle 2
- **Detection**: ✓ Success
- **Alignment**: ✓ Success
- **OCR Result**: RBA456C
- **Validation**: ✓ Passed
- **Frames to Confirm**: 5/5

## Troubleshooting

### Camera not detected
- Check camera index (try 0, 1, 2)
- Ensure camera is properly connected
- Check camera permissions

### Poor OCR results
- Ensure good lighting conditions
- Adjust camera distance (30cm - 2m optimal)
- Clean the license plate if dirty
- Check Tesseract installation

### No plates detected
- Adjust detection thresholds in detect.py
- Ensure plate is clearly visible
- Check minimum area settings

## Configuration

Edit `src/detect.py` to adjust detection parameters:
- `MIN_AREA`: Minimum contour area (default: 500)
- `MAX_AREA`: Maximum contour area (default: 50000)
- `MIN_ASPECT_RATIO`: Minimum width/height ratio (default: 2.0)
- `MAX_ASPECT_RATIO`: Maximum width/height ratio (default: 6.0)

Edit `src/validate.py` to adjust validation patterns:
- Modify `PLATE_PATTERNS` for different country formats

## License
MIT License - Educational Project

## Author
Year 3 Computer Science Student

## Acknowledgments
- Reference book by Gabriel Baziramwabo
- OpenCV community
- Tesseract OCR project