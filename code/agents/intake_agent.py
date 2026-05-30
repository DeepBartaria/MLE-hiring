import json
import re
import unicodedata
from typing import Dict, Any, List, Tuple
from code.utils.logger import get_logger
from code.schemas import ParsedConversation

logger = get_logger(__name__)

class IntakeAgent:
    """
    Intake Agent responsible for safe loading, parsing, and normalization
    of raw support tickets. Resistant to adversarial inputs, fake roles, 
    unicode attacks, and malformed JSON.
    """
    
    MAX_MESSAGE_LENGTH = 100000 # Prevent OOM on extremely long messages
    ALLOWED_ROLES = {"user", "assistant", "system"}
    
    def __init__(self):
        pass

    def sanitize_text(self, text: Any) -> str:
        """
        Sanitize unicode, strip invisible characters, and truncate extremely long texts.
        """
        if text is None:
            return ""
            
        if not isinstance(text, str):
            text = str(text)
            
        # Truncate early to prevent DOS
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(f"Message truncated from {len(text)} to {self.MAX_MESSAGE_LENGTH} chars.")
            text = text[:self.MAX_MESSAGE_LENGTH]
            
        # Normalize unicode to NFKC
        text = unicodedata.normalize('NFKC', text)
        
        # Remove invisible/control characters (keeping basic whitespace like \n, \t)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()

    def parse_issue(self, issue_raw: Any) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Parse the JSON safely.
        Returns (parsed_issue_list, has_malformed_roles_or_json)
        """
        has_error = False
        parsed = []
        
        if not issue_raw or not isinstance(issue_raw, str):
            if issue_raw:
                has_error = True
            return [], has_error
            
        try:
            parsed = json.loads(issue_raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Malformed JSON detected: {e}")
            has_error = True
            # Fallback: treat the entire raw string as a single user message
            return [{"role": "user", "content": issue_raw}], True
            
        if not isinstance(parsed, list):
            logger.warning("Parsed issue is not a list. Fallback triggered.")
            return [{"role": "user", "content": str(parsed)}], True
            
        normalized = []
        for turn in parsed:
            if not isinstance(turn, dict):
                has_error = True
                continue
            
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            if not isinstance(role, str) or not isinstance(content, str):
                has_error = True
                continue
                
            role = role.lower().strip()
            
            # Reject fake roles by coercing them to 'user'
            if role not in self.ALLOWED_ROLES:
                logger.warning(f"Fake or invalid role detected: {role}. Coercing to 'user'.")
                role = "user"
                has_error = True
                
            normalized.append({"role": role, "content": content})
            
        return normalized, has_error

    def process_row(self, row: Dict[str, Any]) -> ParsedConversation:
        """
        Process a single CSV row containing 'issue', 'subject', 'company'.
        """
        # Handle potential capitalization differences in CSV headers
        raw_issue = row.get("issue", row.get("Issue", ""))
        raw_subject = row.get("subject", row.get("Subject", ""))
        raw_company = row.get("company", row.get("Company", "None"))
        
        # 1. Parse JSON safely
        original_turns, has_malformed = self.parse_issue(raw_issue)
        
        # 2. Sanitize and combine
        latest_user_message = ""
        combined_user_context = ""
        is_multi_turn = len(original_turns) > 1
        
        for turn in original_turns:
            role = turn["role"]
            content = self.sanitize_text(turn["content"])
            
            if role == "user":
                latest_user_message = content
                if combined_user_context:
                    combined_user_context += "\\n" + content
                else:
                    combined_user_context = content
                    
        cleaned_subject = self.sanitize_text(raw_subject)
        cleaned_company = self.sanitize_text(raw_company)
        
        # Fallback if there is no user message
        if not latest_user_message and cleaned_subject:
            latest_user_message = cleaned_subject
            combined_user_context = cleaned_subject
            
        return ParsedConversation(
            raw_subject=str(raw_subject) if raw_subject is not None else "",
            raw_company=str(raw_company) if raw_company is not None else "None",
            cleaned_subject=cleaned_subject,
            cleaned_company=cleaned_company,
            is_multi_turn=is_multi_turn,
            latest_user_message=latest_user_message,
            combined_user_context=combined_user_context,
            has_malformed_roles=has_malformed,
            original_turns=original_turns
        )
