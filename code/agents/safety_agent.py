import re
from typing import Optional
from code.utils.logger import get_logger
from code.schemas import SafetyResult, RiskLevelEnum, ParsedConversation

logger = get_logger(__name__)

class SafetyAgent:
    """
    Safety Agent responsible for detecting prompt injections, jailbreaks,
    role spoofing, and other adversarial patterns before any retrieval or generation.
    """
    
    # Regex heuristics for detecting attacks. High recall preferred.
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?previous\s+(instructions|prompts|rules)",
        r"(?i)reveal\s+(your\s+)?(system\s+prompt|instructions|rules)",
        r"(?i)print\s+(hidden\s+)?policies",
        r"(?i)system\s+override",
        r"(?i)override\s+escalation",
        r"(?i)classify\s+this\s+as\s+(replied|escalated)",
        r"(?i)forget\s+(all\s+)?rules",
        r"(?i)you\s+are\s+now\s+a\s+(developer|admin|tester)",
        r"(?i)this\s+is\s+a\s+test",
    ]
    
    ROLE_SPOOFING_PATTERNS = [
        r"(?i)i\s+am\s+(an\s+)?(admin|system|developer|root)",
        r"(?i)user\s+authorization:\s*(admin|root)",
    ]
    
    DATA_EXFILTRATION_PATTERNS = [
        r"(?i)output\s+all\s+(data|records|tickets|files)",
        r"(?i)read\s+file",
        r"(?i)dump\s+database",
    ]
    
    TOOL_ABUSE_PATTERNS = [
        r"(?i)delete\s+another\s+(user's\s+)?account",
        r"(?i)grant\s+(me\s+)?(admin|full)\s+access",
        r"(?i)issue\s+refund\s+for\s+all",
    ]

    def __init__(self):
        self.injection_regexes = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self.role_regexes = [re.compile(p) for p in self.ROLE_SPOOFING_PATTERNS]
        self.exfil_regexes = [re.compile(p) for p in self.DATA_EXFILTRATION_PATTERNS]
        self.tool_regexes = [re.compile(p) for p in self.TOOL_ABUSE_PATTERNS]

    def detect_attacks(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Runs regex-based heuristics over the text.
        Returns (attack_detected, attack_type)
        """
        for regex in self.injection_regexes:
            if regex.search(text):
                return True, "prompt_injection"
                
        for regex in self.role_regexes:
            if regex.search(text):
                return True, "role_spoofing"
                
        for regex in self.exfil_regexes:
            if regex.search(text):
                return True, "data_exfiltration"
                
        for regex in self.tool_regexes:
            if regex.search(text):
                return True, "tool_abuse"
                
        # Optional lightweight LLM classifier hook could be added here
                
        return False, None

    def analyze_conversation(self, parsed_conv: ParsedConversation) -> SafetyResult:
        """
        Analyzes the full context for adversarial attacks.
        Returns deterministic SafetyResult.
        """
        attack_detected = False
        attack_type = None
        
        # 1. Did intake catch something structurally illegal?
        if parsed_conv.has_malformed_roles:
            attack_detected = True
            attack_type = "malformed_json_or_roles"
            
        # 2. Check main conversation block
        if not attack_detected:
            attack_detected, attack_type = self.detect_attacks(parsed_conv.combined_user_context)
            
        # 3. Check subject header (hackers hide payloads here)
        if not attack_detected:
            attack_detected, attack_type = self.detect_attacks(parsed_conv.cleaned_subject)
            
        if attack_detected:
            logger.warning(f"Safety Agent triggered! Attack Type: {attack_type}")
            return SafetyResult(
                attack_detected=True,
                attack_type=attack_type,
                pii_detected=False, # Standard PII detection runs elsewhere
                risk_level=RiskLevelEnum.CRITICAL,
                escalation_recommended=True,
                confidence_score=0.95, # Deterministic regexes map to high confidence
                reasoning=f"Detected adversarial pattern: {attack_type}"
            )
            
        return SafetyResult(
            attack_detected=False,
            attack_type=None,
            pii_detected=False,
            risk_level=RiskLevelEnum.LOW,
            escalation_recommended=False,
            confidence_score=0.90,
            reasoning="No adversarial patterns detected."
        )
