import re
from typing import List
from code.agents.domain_base import DomainAgent
from code.schemas import ParsedConversation, ClassificationResult, RiskLevelEnum, RequestTypeEnum

class VisaAgent(DomainAgent):
    KEYWORDS = {
        "visa": 0.5, "credit card": 0.4, "transaction": 0.3, "payment": 0.3,
        "charge": 0.2, "dispute": 0.4, "merchant": 0.3, "card": 0.2
    }
    
    def get_company_name(self) -> str:
        return "Visa"
        
    def match_score(self, conversation: ParsedConversation) -> float:
        text = conversation.combined_user_context
        # Boost if raw company actually matches
        boost = 0.2 if conversation.cleaned_company.lower() == "visa" else 0.0
        return min(1.0, self._calculate_keyword_score(text, self.KEYWORDS) + boost)
        
    def classify(self, conversation: ParsedConversation, confidence: float) -> ClassificationResult:
        text = conversation.combined_user_context.lower()
        
        # Determine request type
        req_type = RequestTypeEnum.PRODUCT_ISSUE
        if "fraud" in text or "stolen" in text or "unauthorized" in text:
            req_type = RequestTypeEnum.BUG
        
        # Determine product area
        product_area = "Payments"
        if "dispute" in text:
            product_area = "Disputes"
        elif "merchant" in text:
            product_area = "Merchant Services"
            
        return ClassificationResult(
            product_area=product_area,
            request_type=req_type,
            language="en", # Defaulting to en for deterministic simplicity
            inferred_company=self.get_company_name(),
            confidence_score=confidence
        )
        
    def assess_risk(self, conversation: ParsedConversation) -> RiskLevelEnum:
        text = conversation.combined_user_context.lower()
        if any(w in text for w in ["fraud", "stolen", "unauthorized", "scam"]):
            return RiskLevelEnum.CRITICAL
        if "dispute" in text or "chargeback" in text:
            return RiskLevelEnum.HIGH
        return RiskLevelEnum.LOW
        
    def get_retrieval_queries(self, conversation: ParsedConversation) -> List[str]:
        # Specialize query for visa corpus
        return [f"Visa payment issue: {conversation.latest_user_message}"]
