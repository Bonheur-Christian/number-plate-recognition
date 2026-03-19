"""
Validation Module
Validates extracted license plate text against expected formats
Implements format checking and consistency validation
"""

import re
from typing import Optional, Dict


class PlateValidator:
    """Validates license plate format and consistency"""
    
    def __init__(self):
        # Rwanda license plate patterns
        # Common formats:
        # - RAD123B (3 letters + 3 digits + 1 letter)
        # - RAD1234B (3 letters + 4 digits + 1 letter)
        # - RBA456C (3 letters + 3 digits + 1 letter)
        # Add more patterns as needed based on Rwanda regulations
        
        self.PLATE_PATTERNS = [
            r'^[A-Z]{3}\d{3}[A-Z]$',      # RAD123B format
            r'^[A-Z]{3}\d{4}[A-Z]$',      # RAD1234B format
            r'^[A-Z]{2}\d{3,4}[A-Z]{1,2}$',  # More flexible format
            r'^[A-Z]{3}\d{3}$',           # 3 letters + 3 digits
        ]
        
        # Minimum and maximum lengths
        self.MIN_LENGTH = 6
        self.MAX_LENGTH = 9
        
    def validate(self, plate_text: str) -> Dict[str, any]:
        """
        Validate plate text against patterns
        
        Args:
            plate_text: Extracted plate text
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'plate': str,
                'pattern_matched': str or None,
                'errors': list
            }
        """
        result = {
            'valid': False,
            'plate': plate_text,
            'pattern_matched': None,
            'errors': []
        }
        
        # Check if text exists
        if not plate_text:
            result['errors'].append("Empty plate text")
            return result
        
        # Clean and uppercase
        cleaned = plate_text.strip().upper()
        result['plate'] = cleaned
        
        # Check length
        if len(cleaned) < self.MIN_LENGTH:
            result['errors'].append(f"Plate too short ({len(cleaned)} < {self.MIN_LENGTH})")
        
        if len(cleaned) > self.MAX_LENGTH:
            result['errors'].append(f"Plate too long ({len(cleaned)} > {self.MAX_LENGTH})")
        
        # Check if contains both letters and numbers
        has_letters = bool(re.search(r'[A-Z]', cleaned))
        has_digits = bool(re.search(r'\d', cleaned))
        
        if not has_letters:
            result['errors'].append("No letters found")
        
        if not has_digits:
            result['errors'].append("No digits found")
        
        # Check against patterns
        for i, pattern in enumerate(self.PLATE_PATTERNS):
            if re.match(pattern, cleaned):
                result['valid'] = True
                result['pattern_matched'] = pattern
                result['errors'] = []  # Clear errors if pattern matched
                break
        
        if not result['valid'] and not result['errors']:
            result['errors'].append("Does not match any known plate pattern")
        
        return result
    
    def is_valid(self, plate_text: str) -> bool:
        """
        Simple boolean validation check
        
        Args:
            plate_text: Extracted plate text
            
        Returns:
            True if valid, False otherwise
        """
        result = self.validate(plate_text)
        return result['valid']
    
    def correct_common_errors(self, plate_text: str) -> str:
        """
        Attempt to correct common OCR errors
        
        Args:
            plate_text: Raw OCR text
            
        Returns:
            Corrected text
        """
        if not plate_text:
            return plate_text
        
        corrected = plate_text.upper()
        
        # Common OCR mistakes for license plates
        # These corrections are context-dependent
        
        # At the beginning (usually letters):
        # 0 -> O, 1 -> I
        first_three = corrected[:3] if len(corrected) >= 3 else corrected
        first_three = first_three.replace('0', 'O').replace('1', 'I')
        
        # In the middle (usually digits):
        # O -> 0, I -> 1, S -> 5, B -> 8
        if len(corrected) > 3:
            middle = corrected[3:-1] if len(corrected) > 4 else corrected[3:]
            middle = middle.replace('O', '0').replace('I', '1')
            middle = middle.replace('S', '5').replace('B', '8')
            
            # Last character (usually letter):
            last = corrected[-1]
            last = last.replace('0', 'O').replace('1', 'I')
            
            corrected = first_three + middle + last
        else:
            corrected = first_three
        
        return corrected
    
    def format_plate(self, plate_text: str) -> str:
        """
        Format plate text for consistent display
        
        Args:
            plate_text: Raw plate text
            
        Returns:
            Formatted plate text
        """
        if not plate_text:
            return ""
        
        # Remove spaces and special characters
        formatted = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        
        # Apply correction
        formatted = self.correct_common_errors(formatted)
        
        return formatted


class PlateConfirmation:
    """Handles multi-frame confirmation of plate detections"""
    
    def __init__(self, required_confirmations: int = 5, max_variation: int = 1):
        """
        Initialize confirmation tracker
        
        Args:
            required_confirmations: Number of times plate must be seen
            max_variation: Maximum character differences allowed between readings
        """
        self.required_confirmations = required_confirmations
        self.max_variation = max_variation
        self.detections = []
        self.current_plate = None
        self.confirmation_count = 0
        
    def add_detection(self, plate_text: str) -> Dict[str, any]:
        """
        Add a new detection and check for confirmation
        
        Args:
            plate_text: Detected plate text
            
        Returns:
            Dictionary with confirmation status
        """
        if not plate_text:
            return self._get_status()
        
        # Add to detections list
        self.detections.append(plate_text)
        
        # Keep only recent detections (sliding window)
        if len(self.detections) > self.required_confirmations * 2:
            self.detections = self.detections[-(self.required_confirmations * 2):]
        
        # Check for consistent plate
        if self._check_consistency(plate_text):
            self.confirmation_count += 1
            self.current_plate = plate_text
        else:
            # Reset if new plate detected
            if self.current_plate and self._levenshtein_distance(plate_text, self.current_plate) > self.max_variation:
                self.reset()
                self.current_plate = plate_text
                self.confirmation_count = 1
        
        return self._get_status()
    
    def _check_consistency(self, plate_text: str) -> bool:
        """Check if plate_text is consistent with current detections"""
        if not self.current_plate:
            return True
        
        # Check Levenshtein distance
        distance = self._levenshtein_distance(plate_text, self.current_plate)
        return distance <= self.max_variation
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def is_confirmed(self) -> bool:
        """Check if plate is confirmed"""
        return self.confirmation_count >= self.required_confirmations
    
    def get_confirmed_plate(self) -> Optional[str]:
        """Get confirmed plate if available"""
        if self.is_confirmed():
            return self.current_plate
        return None
    
    def reset(self):
        """Reset confirmation tracker"""
        self.detections = []
        self.current_plate = None
        self.confirmation_count = 0
    
    def _get_status(self) -> Dict[str, any]:
        """Get current confirmation status"""
        return {
            'plate': self.current_plate,
            'count': self.confirmation_count,
            'required': self.required_confirmations,
            'confirmed': self.is_confirmed(),
            'progress': f"{self.confirmation_count}/{self.required_confirmations}"
        }


def test_validator():
    """Test the validation module"""
    print("Testing Plate Validator...")
    
    validator = PlateValidator()
    
    # Test valid plates
    valid_plates = ["RAD123B", "RBA456C", "RAC7890D"]
    for plate in valid_plates:
        result = validator.validate(plate)
        status = "✓" if result['valid'] else "✗"
        print(f"  {status} {plate}: {result['valid']}")
    
    # Test invalid plates
    invalid_plates = ["ABC", "12345", "TOOLONG123456"]
    for plate in invalid_plates:
        result = validator.validate(plate)
        status = "✓" if not result['valid'] else "✗"
        print(f"  {status} {plate}: invalid (expected)")
    
    # Test confirmation
    print("\nTesting Plate Confirmation...")
    confirmation = PlateConfirmation(required_confirmations=3)
    
    for i in range(5):
        status = confirmation.add_detection("RAD123B")
        print(f"  Detection {i+1}: {status['progress']} - Confirmed: {status['confirmed']}")
    
    return True


if __name__ == "__main__":
    test_validator()