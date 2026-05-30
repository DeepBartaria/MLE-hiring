from typing import Optional
from code.schemas import (
    SafetyResult, 
    ClassificationResult, 
    RetrievalResult, 
    EscalationDecision, 
    StatusEnum, 
    RiskLevelEnum
)
from code.utils.logger import get_logger

logger = get_logger(__name__)

class EscalationAgent:
    """
    Evaluates safety, classification, and retrieval signals to 
    deterministically decide whether a ticket should be replied to or escalated.
    """
    
    # Thresholds for heuristics
    RETRIEVAL_MIN_GROUNDING = 0.3
    RETRIEVAL_CONFLICT_DELTA = 0.05
    CLASSIFICATION_MIN_CONFIDENCE = 0.2
    
    def evaluate(
        self, 
        safety: SafetyResult, 
        classification: ClassificationResult, 
        retrieval: RetrievalResult
    ) -> EscalationDecision:
        
        reasons = []
        should_escalate = False
        
        # 1. Safety Checks (Fraud, Legal threats, Compromise, Injection)
        if safety.escalation_recommended:
            should_escalate = True
            reasons.append(f"Safety constraint violated (Risk: {safety.risk_level.value}). Reason: {safety.reasoning}")
            
        if safety.risk_level in [RiskLevelEnum.CRITICAL, RiskLevelEnum.HIGH]:
            should_escalate = True
            reasons.append(f"High risk identified: {safety.risk_level.value}")
            
        # 2. Classification Checks (Ambiguity / Unsupported)
        if classification.confidence_score < self.CLASSIFICATION_MIN_CONFIDENCE:
            should_escalate = True
            reasons.append(f"Ambiguous or unsupported request (Confidence: {classification.confidence_score:.2f})")
            
        # 3. Retrieval Checks (Insufficient grounding, Conflicting evidence)
        top_score = 0.0
        if not retrieval.retrieval_scores:
            should_escalate = True
            reasons.append("Zero relevant documents retrieved (No grounding).")
        else:
            top_score = retrieval.retrieval_scores[0]
            if top_score < self.RETRIEVAL_MIN_GROUNDING:
                should_escalate = True
                reasons.append(f"Insufficient corpus grounding (Top score: {top_score:.2f})")
                
            # Conflicting evidence heuristic (top 2 scores are very close but might represent conflicting docs)
            if len(retrieval.retrieval_scores) > 1:
                second_score = retrieval.retrieval_scores[1]
                delta = top_score - second_score
                
                top_doc = retrieval.source_documents[0]
                second_doc = retrieval.source_documents[1]
                
                # Only consider it conflicting if they come from DIFFERENT source documents
                # and the score delta is extremely small (e.g. < 0.01) indicating a near tie.
                if top_doc != second_doc:
                    if top_score > 0.8 and delta < 0.01:
                        should_escalate = True
                        reasons.append(f"Conflicting retrieval evidence detected (Delta: {delta:.3f})")
                    
        # 4. Finalizing the Decision
        if should_escalate:
            status = StatusEnum.ESCALATED
            justification = " | ".join(reasons)
        else:
            status = StatusEnum.REPLIED
            justification = "Safe to reply. High confidence and sufficient grounding."
            
        logger.info(f"Escalation Decision: {status.value} - {justification}")
        
        return EscalationDecision(
            should_escalate=should_escalate,
            status=status,
            justification=justification
        )
