import pytest
from code.agents.response_agent import ResponseGenerationAgent
from code.schemas import (
    ParsedConversation,
    RetrievalResult,
    EscalationDecision,
    ClassificationResult,
    SafetyResult,
    StatusEnum,
    RiskLevelEnum,
    RequestTypeEnum
)

@pytest.fixture
def base_conv():
    return ParsedConversation(
        raw_subject="Test",
        raw_company="TestCo",
        cleaned_subject="Test",
        cleaned_company="TestCo",
        is_multi_turn=False,
        latest_user_message="How do I reset my device?",
        combined_user_context="How do I reset my device?",
        has_malformed_roles=False,
        original_turns=[]
    )

@pytest.fixture
def base_escalation():
    return EscalationDecision(
        should_escalate=False,
        status=StatusEnum.REPLIED,
        justification="Safe"
    )

@pytest.fixture
def base_retrieval():
    return RetrievalResult(
        retrieved_chunks=["To reset your device, hold the power button."],
        source_documents=["doc.md"],
        retrieval_scores=[0.9]
    )

@pytest.fixture
def base_classification():
    return ClassificationResult(
        product_area="Hardware",
        request_type=RequestTypeEnum.PRODUCT_ISSUE,
        language="en",
        inferred_company="TestCo",
        confidence_score=0.9
    )

@pytest.fixture
def base_safety():
    return SafetyResult(
        attack_detected=False,
        risk_level=RiskLevelEnum.LOW,
        escalation_recommended=False,
        confidence_score=1.0,
        reasoning="Safe"
    )

def test_escalation_short_circuit(base_conv, base_retrieval, base_classification, base_safety):
    escalation = EscalationDecision(should_escalate=True, status=StatusEnum.ESCALATED, justification="High Risk")
    agent = ResponseGenerationAgent(use_mock_llm=True)
    
    response = agent.generate(base_conv, escalation, base_retrieval, base_classification, base_safety)
    assert "escalated your ticket to our human support team" in response

def test_multilingual_fallback(base_conv, base_retrieval, base_safety):
    escalation = EscalationDecision(should_escalate=True, status=StatusEnum.ESCALATED, justification="High Risk")
    classification = ClassificationResult(
        product_area="Hardware",
        request_type=RequestTypeEnum.PRODUCT_ISSUE,
        language="es",
        inferred_company="TestCo",
        confidence_score=0.9
    )
    
    agent = ResponseGenerationAgent(use_mock_llm=True)
    response = agent.generate(base_conv, escalation, base_retrieval, classification, base_safety)
    assert "Pido disculpas" in response

def test_mock_llm_generation(base_conv, base_escalation, base_retrieval, base_classification, base_safety):
    agent = ResponseGenerationAgent(use_mock_llm=True)
    response = agent.generate(base_conv, base_escalation, base_retrieval, base_classification, base_safety)
    
    assert "Based on the documentation" in response
