import pytest
from pydantic import ValidationError
from code.schemas import (
    StatusEnum, RequestTypeEnum, RiskLevelEnum,
    TicketInput, ConversationTurn, ToolAction, FinalTicketOutput
)

def test_ticket_input_validation():
    """Test standard ticket parsing."""
    ticket = TicketInput(
        issue=[ConversationTurn(role="user", content="Hello")],
        subject="Login issue",
        company="DevPlatform"
    )
    assert ticket.subject == "Login issue"
    assert len(ticket.issue) == 1

def test_final_ticket_output_valid():
    """Test that a compliant agent prediction is serialized deterministically."""
    output = FinalTicketOutput(
        status=StatusEnum.REPLIED,
        product_area="Auth",
        response="Please reset your password.",
        justification="Standard FAQ",
        request_type=RequestTypeEnum.PRODUCT_ISSUE,
        confidence_score=0.95,
        source_documents="data/devplatform/auth.md",
        risk_level=RiskLevelEnum.LOW,
        pii_detected=False,
        language="en",
        actions_taken=[ToolAction(action="reset_pw", parameters={"user_id": "123"})]
    )
    
    # Verify standard values
    assert output.status == "replied"
    assert output.actions_taken[0].action == "reset_pw"

    # Test JSON-safe serialization
    csv_dict = output.to_csv_dict()
    assert csv_dict["status"] == "replied"
    assert csv_dict["request_type"] == "product_issue"
    assert csv_dict["risk_level"] == "low"
    assert isinstance(csv_dict["actions_taken"], list)
    assert csv_dict["actions_taken"][0]["parameters"]["user_id"] == "123"

def test_strict_enum_validation():
    """Test that invalid classifications immediately raise ValidationErrors."""
    
    # Test invalid Status
    with pytest.raises(ValidationError) as exc:
        FinalTicketOutput(
            status="resolved", # invalid status
            product_area="Auth",
            response="done",
            justification="reason",
            request_type=RequestTypeEnum.PRODUCT_ISSUE,
            confidence_score=1.0,
            source_documents="",
            risk_level=RiskLevelEnum.LOW,
            pii_detected=False,
            language="en"
        )
    assert "Input should be 'replied' or 'escalated'" in str(exc.value)
        
    # Test invalid Confidence bounds
    with pytest.raises(ValidationError) as exc:
        FinalTicketOutput(
            status=StatusEnum.REPLIED,
            product_area="Auth",
            response="done",
            justification="reason",
            request_type=RequestTypeEnum.PRODUCT_ISSUE,
            confidence_score=1.5, # Invalid bounds (>1.0)
            source_documents="",
            risk_level=RiskLevelEnum.LOW,
            pii_detected=False,
            language="en"
        )
    assert "Input should be less than or equal to 1" in str(exc.value)
