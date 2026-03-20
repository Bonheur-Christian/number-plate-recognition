import re

PLATE_PATTERNS = [
    r"[A-Z]{3}[0-9]{3}[A-Z]?",   
    r"[A-Z]{2}[0-9]{3}[A-Z]{2}", 
]

def extract_plate(text: str) -> str:
    if not text:
        return ""
    for pattern in PLATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return text

def is_valid_plate(text: str) -> bool:
    if not text:
        return False
    
    for pattern in PLATE_PATTERNS:
        if re.search(pattern, text):
            return True

    return False