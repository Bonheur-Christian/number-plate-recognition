import cv2
import pytesseract
import re

# Set your Tesseract path here if needed on Windows:
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


def clean_text(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def read_plate_text(plate_img):
    processed = preprocess_for_ocr(plate_img)

    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(processed, config=config)
    text = clean_text(raw_text)

    return text, processed