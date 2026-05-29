from typing import List
from code.agents.domain_base import DomainAgent
from code.schemas import ParsedConversation, ClassificationResult, RiskLevelEnum, RequestTypeEnum

class DevPlatformAgent(DomainAgent):
    KEYWORDS = {
        "api key": 0.4, "endpoint": 0.3, "webhook": 0.4, "rate limit": 0.4,
        "sdk": 0.3, "server": 0.2, "timeout": 0.3, "500 error": 0.4, "devplatform": 0.5
    }
    
    def get_company_name(self) -> str:
        return "DevPlatform"
        
    def match_score(self, conversation: ParsedConversation) -> float:
        text = conversation.combined_user_context
        boost = 0.2 if conversation.cleaned_company.lower() == "devplatform" else 0.0
        return min(1.0, self._calculate_keyword_score(text, self.KEYWORDS) + boost)
        
    def classify(self, conversation: ParsedConversation, confidence: float) -> ClassificationResult:
        text = conversation.combined_user_context.lower()
        
        req_type = RequestTypeEnum.PRODUCT_ISSUE
        if "bug" in text or "500" in text or "down" in text:
            req_type = RequestTypeEnum.BUG
        elif "feature" in text:
            req_type = RequestTypeEnum.FEATURE_REQUEST
            
        product_area = "API Infrastructure"
        if "sdk" in text:
            product_area = "Client SDKs"
        elif "webhook" in text:
            product_area = "Webhooks"
            
        return ClassificationResult(
            product_area=product_area,
            request_type=req_type,
            language="en",
            inferred_company=self.get_company_name(),
            confidence_score=confidence
        )
        
    def assess_risk(self, conversation: ParsedConversation) -> RiskLevelEnum:
        text = conversation.combined_user_context.lower()
        if "production down" in text or "outage" in text or "breach" in text or "leaked key" in text:
            return RiskLevelEnum.CRITICAL
        if "timeout" in text or "rate limit" in text:
            return RiskLevelEnum.MEDIUM
        return RiskLevelEnum.LOW
        
    def get_retrieval_queries(self, conversation: ParsedConversation) -> List[str]:
        return [f"DevPlatform API issue: {conversation.latest_user_message}"]
