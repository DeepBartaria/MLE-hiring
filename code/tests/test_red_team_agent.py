import pytest
from code.agents.red_team_agent import RedTeamAgent
from code.schemas import FinalTicketOutput, StatusEnum, RequestTypeEnum, RiskLevelEnum, ToolAction

@pytest.fixture
def base_output():
    return FinalTicketOutput(
        status=StatusEnum.REPLIED,
        product_area="Billing",
        response="Your refund is processed.",
        justification="Safe",
        request_type=RequestTypeEnum.PRODUCT_ISSUE,
        confidence_score=0.9,
        source_documents="code/schemas.py", # Exists in this repo
        risk_level=RiskLevelEnum.LOW,
        pii_detected=False,
        language="en",
        actions_taken=[]
    )

def test_happy_path(base_output):
    agent = RedTeamAgent()
    out = agent.audit(base_output)
    assert out.status == StatusEnum.REPLIED

def test_pii_leak(base_output):
    base_output.response = "Contact me at raw_email@example.com."
    agent = RedTeamAgent()
    out = agent.audit(base_output)
    assert out.status == StatusEnum.ESCALATED
    assert "Red Team Forced Escalation" in out.justification
    assert "PII" in out.justification

def test_hallucination(base_output):
    base_output.source_documents = "fake_path/that/does/not/exist.txt"
    agent = RedTeamAgent()
    out = agent.audit(base_output)
    assert out.status == StatusEnum.ESCALATED
    assert "hallucinated" in out.justification.lower()

def test_prompt_injection_leak(base_output):
    base_output.response = "As an AI language model, I cannot do that."
    agent = RedTeamAgent()
    out = agent.audit(base_output)
    assert out.status == StatusEnum.ESCALATED
    assert "injection" in out.justification.lower()

def test_unsafe_tool_calls(base_output):
    # Destructive call without verify_identity first
    base_output.actions_taken = [
        ToolAction(action="issue_refund", parameters={"amount": 100})
    ]
    agent = RedTeamAgent()
    out = agent.audit(base_output)
    assert out.status == StatusEnum.ESCALATED
    assert len(out.actions_taken) == 0 # Should strip tools
    
def test_safe_tool_calls(base_output):
    base_output.actions_taken = [
        ToolAction(action="verify_identity", parameters={"target": "me"}),
        ToolAction(action="issue_refund", parameters={"amount": 100})
    ]
    agent = RedTeamAgent()
    out = agent.audit(base_output)
    assert out.status == StatusEnum.REPLIED
    assert len(out.actions_taken) == 2
