from abc import ABC, abstractmethod
from typing import List
from code.schemas import ParsedConversation, ClassificationResult, RiskLevelEnum

class DomainAgent(ABC):
    @abstractmethod
    def get_company_name(self) -> str:
        pass
        
    @abstractmethod
    def match_score(self, conversation: ParsedConversation) -> float:
        """Calculate confidence [0.0, 1.0] that this ticket belongs to this domain."""
        pass
        
    @abstractmethod
    def classify(self, conversation: ParsedConversation, confidence: float) -> ClassificationResult:
        """Classify product area and request type."""
        pass
        
    @abstractmethod
    def assess_risk(self, conversation: ParsedConversation) -> RiskLevelEnum:
        """Domain-specific risk assessment."""
        pass
        
    @abstractmethod
    def get_retrieval_queries(self, conversation: ParsedConversation) -> List[str]:
        """Generate domain-specialized retrieval queries."""
        pass
        
    def _calculate_keyword_score(self, text: str, keywords: dict[str, float]) -> float:
        """Helper to calculate a bounded score based on keyword hits."""
        text_lower = text.lower()
        score = 0.0
        for kw, weight in keywords.items():
            if kw in text_lower:
                score += weight
        return min(1.0, score)
