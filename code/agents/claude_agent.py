from typing import List
from code.agents.domain_base import DomainAgent
from code.schemas import ParsedConversation, ClassificationResult, RiskLevelEnum, RequestTypeEnum

class ClaudeAgent(DomainAgent):
    KEYWORDS = {
        "claude": 0.5, "anthropic": 0.4, "opus": 0.4, "sonnet": 0.4,
        "haiku": 0.4, "model": 0.2, "prompt": 0.3, "hallucination": 0.3, "token": 0.2
    }
    
    def get_company_name(self) -> str:
        return "Claude"
        
    def match_score(self, conversation: ParsedConversation) -> float:
        text = conversation.combined_user_context
        boost = 0.2 if conversation.cleaned_company.lower() in ["claude", "anthropic"] else 0.0
        return min(1.0, self._calculate_keyword_score(text, self.KEYWORDS) + boost)
        
    def classify(self, conversation: ParsedConversation, confidence: float) -> ClassificationResult:
        text = conversation.combined_user_context.lower()
        
        req_type = RequestTypeEnum.PRODUCT_ISSUE
        if "feature" in text or "support for" in text:
            req_type = RequestTypeEnum.FEATURE_REQUEST
        elif "bug" in text or "error" in text:
            req_type = RequestTypeEnum.BUG
            
        product_area = "Model Inference"
        if "console" in text or "billing" in text:
            product_area = "Account Management"
            
        return ClassificationResult(
            product_area=product_area,
            request_type=req_type,
            language="en",
            inferred_company=self.get_company_name(),
            confidence_score=confidence
        )
        
    def assess_risk(self, conversation: ParsedConversation) -> RiskLevelEnum:
        text = conversation.combined_user_context.lower()
        if "jailbreak" in text or "ignore previous" in text or "system prompt" in text:
            return RiskLevelEnum.CRITICAL # Also handled by SafetyAgent, but good domain backup
        if "overcharged" in text or "billing" in text:
            return RiskLevelEnum.MEDIUM
        return RiskLevelEnum.LOW
        
    def get_retrieval_queries(self, conversation: ParsedConversation) -> List[str]:
        return [f"Claude model behavior: {conversation.latest_user_message}"]
