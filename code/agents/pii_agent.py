import re
from typing import List
from code.utils.logger import get_logger
from code.schemas import PIIResult

logger = get_logger(__name__)

class PIIAgent:
    """
    PII Detection Agent responsible for identifying and redacting 
    personally identifiable information deterministically.
    """
    
    PII_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]{2,}",
        "PHONE": r"(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
        "SSN": r"\b\d{3}[\s-]?\d{2}[\s-]?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{15,16}\b", 
        "API_KEY": r"(?i)(?:api[_-]?key|token|secret)[\s:=]+[\"']?[A-Za-z0-9\-_]{16,}[\"']?",
        "BANK_ACCOUNT": r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b", # Basic IBAN format
        "ADDRESS": r"(?i)\b\d+\s+(?:[\w\s.,-]*?\b(?:Avenue|Ave|Street|St|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Circle|Cir)\b|[\w\s.,-]*?(?:Straße|strasse)\b|(?:Calle|Rue)\s+[\w\s.,-]+)\b"
    }

    def __init__(self):
        # Sort to guarantee deterministic execution order
        self.compiled_patterns = {
            name: re.compile(pattern) for name, pattern in sorted(self.PII_PATTERNS.items())
        }

    def redact(self, text: str) -> PIIResult:
        """
        Scans text for PII, masks it, and returns the result.
        Replaces matched PII with semantic placeholders like [REDACTED_EMAIL].
        """
        if not text:
            return PIIResult(pii_detected=False, detected_pii_types=[], redacted_text="")

        detected_types = set()
        redacted_text = text
        
        # Apply regex replacements
        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(redacted_text)
            has_match = False
            
            def replace_func(match):
                nonlocal has_match
                has_match = True
                return f"[REDACTED_{pii_type}]"
                
            redacted_text = pattern.sub(replace_func, redacted_text)
            if has_match:
                detected_types.add(pii_type)
                
        pii_detected = len(detected_types) > 0
        
        if pii_detected:
            logger.info(f"PII detected: {list(detected_types)}")
            
        return PIIResult(
            pii_detected=pii_detected,
            detected_pii_types=sorted(list(detected_types)),
            redacted_text=redacted_text
        )
