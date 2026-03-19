# Quick Start Guide

## 5-Minute Setup

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
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or use the automated setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Test the System

```bash
python test_system.py
```

This will:
- ✓ Test all components
- ✓ Generate example screenshots
- ✓ Verify Tesseract installation
- ✓ Check camera availability

### 4. Run the ANPR System

**With default camera (built-in webcam):**
```bash
python src/main.py
```

**With external USB camera:**
```bash
python src/main.py --camera 1
```

**With custom save directory:**
```bash
python src/main.py --save-dir ./my_results
```

---

## Usage

### Controls
- **'q'** - Quit the application
- **'s'** - Save current frame as screenshot
- **'r'** - Reset confirmation counter

### What You'll See

1. **Main Window** - Live camera feed with detection overlay
2. **Detection Status** - Shows "PLATE DETECTED" when found
3. **OCR Result** - Displays extracted text with confidence
4. **Confirmation Progress** - Shows progress (e.g., "3/5")
5. **Aligned Plate** - Preview of corrected plate image
6. **Statistics Panel** - Real-time processing stats

### Typical Workflow

1. **Position camera** - Point at license plate (1-2m distance)
2. **Wait for detection** - System will show "PLATE DETECTED"
3. **Hold steady** - Let system confirm (5 frames)
4. **See confirmation** - "CONFIRMED!" appears on screen
5. **Check results** - Plate saved to `data/plates.csv`

---

## Testing with Real Vehicles

### Preparation

1. **Get Permission**
   - Ask vehicle owner before testing
   - Explain it's a school project
   - Only test on plates, not personal info

2. **Choose Location**
   - School parking area recommended
   - Good lighting conditions
   - Stable camera position

3. **Camera Setup**
   - Connect external USB camera (recommended)
   - Position camera stable (tripod or laptop)
   - Distance: 0.5m - 2m from plate
   - Ensure plate is clearly visible

### Best Practices

**Lighting:**
- ✓ Natural daylight (best)
- ✓ Well-lit indoor areas
- ✗ Direct sunlight (causes glare)
- ✗ Very low light (poor contrast)

**Distance:**
- ✓ 1-2 meters (optimal)
- ✓ 0.5-1 meter (close-up)
- ✗ > 3 meters (too far)

**Angle:**
- ✓ Front view (0-15°)
- ✓ Slight angle (15-30°)
- ✗ Extreme angle (> 45°)

**Plate Condition:**
- ✓ Clean plates (best results)
- ~ Slightly dirty (still works)
- ✗ Very dirty/damaged (may fail)

---

## Understanding Results

### CSV Output Format

Location: `data/plates.csv`

```csv
Timestamp,Plate,Confidence,Valid
2024-03-19 10:30:45,RAD123B,89.50,True
2024-03-19 10:35:22,RBA456C,92.30,True
```

**Columns:**
- **Timestamp**: When plate was confirmed
- **Plate**: Extracted plate number
- **Confidence**: OCR confidence (0-100%)
- **Valid**: Passed format validation (True/False)

### Statistics Display

```
Frames: 1250          # Total frames processed
Detected: 145         # Plates detected in frames
Confirmed: 12         # Plates confirmed and saved
OCR Rate: 145/145     # Successful OCR attempts
```

---

## Troubleshooting

### Camera Issues

**Problem:** "Cannot open camera 0"
```bash
# Try different camera index
python src/main.py --camera 1
python src/main.py --camera 2
```

**Problem:** Camera works but blurry
- Check camera focus
- Clean camera lens
- Adjust distance to plate

### OCR Issues

**Problem:** Wrong characters detected
- Improve lighting
- Clean the license plate
- Adjust camera distance
- Ensure plate is flat/aligned

**Problem:** No text extracted
- Check Tesseract installation: `tesseract --version`
- Verify plate is visible in detection
- Try different lighting

### Detection Issues

**Problem:** No plate detected
- Ensure plate is fully visible
- Check MIN_AREA and MAX_AREA in `src/detect.py`
- Adjust camera angle
- Improve lighting

**Problem:** False detections
- Tighten aspect ratio constraints
- Increase confirmation threshold
- Add more validation patterns

---

## Project Submission Checklist

### GitHub Repository

- [ ] All code files committed
- [ ] README.md complete
- [ ] requirements.txt included
- [ ] Screenshots in screenshots/ folder
- [ ] Test results documented

### Required Files

```
anpr-project/
├── README.md ✓
├── requirements.txt ✓
├── src/
│   ├── detect.py ✓
│   ├── align.py ✓
│   ├── ocr.py ✓
│   ├── validate.py ✓
│   └── main.py ✓
├── data/
│   └── plates.csv ✓
└── screenshots/
    ├── detection.png ✓
    ├── alignment.png ✓
    └── ocr.png ✓
```

### Testing Documentation

Include in README:
- [ ] Multiple test vehicles
- [ ] Different lighting conditions
- [ ] Various angles
- [ ] Success/failure examples


## Next Steps

1. **Test Thoroughly**
   - Test on 5+ different vehicles
   - Try different conditions
   - Document results

2. **Document Results**
   - Take screenshots
   - Record statistics
   - Note any issues

3. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial ANPR implementation"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```


## Support

### Common Questions

**Q: How long should confirmation take?**
A: With 5-frame requirement at 30 FPS, about 0.2 seconds when plate is stable.

**Q: Can I adjust confirmation threshold?**
A: Yes, edit `PlateConfirmation(required_confirmations=5)` in main.py

**Q: What plate formats are supported?**
A: Currently Rwanda formats (RAD123B). Edit PLATE_PATTERNS in validate.py for others.

**Q: Can I use a phone camera?**
A: Yes, if you can connect it as a webcam (apps like DroidCam work).

### Getting Help

- Check IMPLEMENTATION.md for detailed explanations
- Review test_system.py output for diagnostics
- Verify Tesseract installation
- Check camera permissions

---

## Good Luck! 🚀

Remember:
- Test on real vehicles (with permission)
- Document your results

Your system is ready to detect, align, and recognize license plates!