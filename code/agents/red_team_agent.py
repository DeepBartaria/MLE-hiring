import os
import re
from typing import List, Optional
from code.schemas import FinalTicketOutput, StatusEnum, ToolAction
from code.utils.logger import get_logger

logger = get_logger(__name__)

class RedTeamAgent:
    """
    Audits the generated FinalTicketOutput before it's finalized.
    If violations are found, it deterministically forces an escalation
    and sanitizes the output.
    """
    
    # Common PII regexes just for final output check
    EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_REGEX = re.compile(r'\b\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    
    # System prompt leakages
    PROMPT_LEAK_KEYWORDS = [
        "I am an AI",
        "system prompt",
        "ignore previous instructions",
        "as an AI language model",
        "OpenAI",
        "Anthropic"
    ]
    
    DESTRUCTIVE_TOOLS = {"issue_refund", "modify_subscription", "lock_account"}
    
    def __init__(self):
        self.fallback_message = "I apologize, but I am unable to fully resolve your request at this time. I have escalated your ticket to our human support team, and someone will reach out to you shortly."
        
    def audit(self, output: FinalTicketOutput) -> FinalTicketOutput:
        """
        Runs all auditing checks. If any fail, mutates the output to escalate safely.
        """
        violations = []
        
        # 1. Check PII Leakage
        if self._detect_pii(output.response):
            violations.append("Leaked PII detected in response.")
            
        # 2. Prompt Leakage
        if self._detect_prompt_leakage(output.response):
            violations.append("Prompt injection or system leakage detected in response.")
            
        # 3. Citation Hallucinations
        hallucinations = self._detect_hallucinated_citations(output.source_documents)
        if hallucinations:
            violations.append(f"Hallucinated citations detected: {hallucinations}")
            
        # 4. Unsafe Tool Calls
        unsafe_tools = self._detect_unsafe_tool_calls(output.actions_taken)
        if unsafe_tools:
            violations.append(f"Unsafe or unauthorized tool calls detected: {unsafe_tools}")
            
        if violations:
            logger.warning(f"Red Team Audit FAILED. Violations: {violations}")
            return self._force_escalation(output, violations)
            
        logger.info("Red Team Audit PASSED.")
        return output
        
    def _detect_pii(self, text: str) -> bool:
        if self.EMAIL_REGEX.search(text) and "[REDACTED" not in text:
            # Note: We assume if real emails are found without redaction it's a leak.
            # In a real app we'd need to distinguish user email vs contact support email.
            return True
        if self.PHONE_REGEX.search(text) and "[REDACTED" not in text:
            return True
        return False
        
    def _detect_prompt_leakage(self, text: str) -> bool:
        text_lower = text.lower()
        for kw in self.PROMPT_LEAK_KEYWORDS:
            if kw.lower() in text_lower:
                return True
        return False
        
    def _detect_hallucinated_citations(self, source_documents: str) -> List[str]:
        if not source_documents or source_documents == "None":
            return []
            
        hallucinated = []
        paths = [p.strip() for p in source_documents.split("|") if p.strip()]
        for path in paths:
            if not os.path.exists(path):
                # For testing and avoiding false positives if the agent didn't prefix right,
                # we just strictly enforce the rule. If path doesn't exist, it's a hallucination.
                # In production, you might resolve relative paths to repo root.
                if not os.path.exists(os.path.abspath(path)):
                    hallucinated.append(path)
        return hallucinated
        
    def _detect_unsafe_tool_calls(self, actions: List[ToolAction]) -> List[str]:
        unsafe = []
        has_verification = False
        
        for action in actions:
            if action.action == "verify_identity":
                has_verification = True
            elif action.action in self.DESTRUCTIVE_TOOLS:
                if not has_verification:
                    unsafe.append(action.action)
        return unsafe
        
    def _force_escalation(self, output: FinalTicketOutput, violations: List[str]) -> FinalTicketOutput:
        output.status = StatusEnum.ESCALATED
        output.response = self.fallback_message
        output.justification += f" | Red Team Forced Escalation: {', '.join(violations)}"
        output.actions_taken = []
        return output
