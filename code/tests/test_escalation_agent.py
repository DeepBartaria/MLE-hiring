import pytest
from code.schemas import (
    SafetyResult, 
    ClassificationResult, 
    RetrievalResult, 
    RiskLevelEnum,
    RequestTypeEnum,
    StatusEnum
)
from code.agents.escalation_agent import EscalationAgent

@pytest.fixture
def base_safety():
    return SafetyResult(
        attack_detected=False,
        risk_level=RiskLevelEnum.LOW,
        escalation_recommended=False,
        confidence_score=1.0,
        reasoning="Safe"
    )
    
@pytest.fixture
def base_classification():
    return ClassificationResult(
        product_area="Payments",
        request_type=RequestTypeEnum.PRODUCT_ISSUE,
        language="en",
        inferred_company="Visa",
        confidence_score=0.9
    )
    
@pytest.fixture
def base_retrieval():
    return RetrievalResult(
        retrieved_chunks=["Here is the solution."],
        source_documents=["doc.txt"],
        retrieval_scores=[0.8, 0.5] # No conflict
    )

def test_happy_path(base_safety, base_classification, base_retrieval):
    agent = EscalationAgent()
    decision = agent.evaluate(base_safety, base_classification, base_retrieval)
    
    assert decision.should_escalate is False
    assert decision.status == StatusEnum.REPLIED
    assert "Safe to reply" in decision.justification

def test_safety_escalation(base_safety, base_classification, base_retrieval):
    base_safety.escalation_recommended = True
    base_safety.risk_level = RiskLevelEnum.CRITICAL
    base_safety.reasoning = "Prompt Injection detected"
    
    agent = EscalationAgent()
    decision = agent.evaluate(base_safety, base_classification, base_retrieval)
    
    assert decision.should_escalate is True
    assert decision.status == StatusEnum.ESCALATED
    assert "Prompt Injection" in decision.justification

def test_insufficient_grounding(base_safety, base_classification, base_retrieval):
    base_retrieval.retrieval_scores = [0.2] # Below 0.3 threshold
    
    agent = EscalationAgent()
    decision = agent.evaluate(base_safety, base_classification, base_retrieval)
    
    assert decision.should_escalate is True
    assert "Insufficient corpus grounding" in decision.justification

def test_conflicting_evidence(base_safety, base_classification, base_retrieval):
    # Top two scores are 0.80 and 0.78 (delta 0.02) -> Conflict!
    base_retrieval.retrieval_scores = [0.80, 0.78]
    
    agent = EscalationAgent()
    decision = agent.evaluate(base_safety, base_classification, base_retrieval)
    
    assert decision.should_escalate is True
    assert "Conflicting retrieval evidence" in decision.justification

def test_ambiguous_classification(base_safety, base_classification, base_retrieval):
    base_classification.confidence_score = 0.1 # Below 0.2
    
    agent = EscalationAgent()
    decision = agent.evaluate(base_safety, base_classification, base_retrieval)
    
    assert decision.should_escalate is True
    assert "Ambiguous" in decision.justification
