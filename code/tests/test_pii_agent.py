import pytest
from code.agents.pii_agent import PIIAgent

@pytest.fixture
def agent():
    return PIIAgent()

def test_email_redaction(agent):
    text = "Contact me at john@example.com."
    res = agent.redact(text)
    assert res.pii_detected
    assert "EMAIL" in res.detected_pii_types
    assert res.redacted_text == "Contact me at [REDACTED_EMAIL]."

def test_multilingual_address(agent):
    text1 = "My office is at 123 Main Street."
    text2 = "Mon bureau est a 45 Rue de la Paix."
    text3 = "Mein Haus ist in der 10 Hauptstraße."
    
    res1 = agent.redact(text1)
    assert "ADDRESS" in res1.detected_pii_types
    
    res2 = agent.redact(text2)
    assert "ADDRESS" in res2.detected_pii_types
    
    res3 = agent.redact(text3)
    assert "ADDRESS" in res3.detected_pii_types

def test_credit_card(agent):
    text = "My card is 4111111111111111."
    res = agent.redact(text)
    assert res.pii_detected
    assert "CREDIT_CARD" in res.detected_pii_types
    assert res.redacted_text == "My card is [REDACTED_CREDIT_CARD]."
    
    text_spaces = "My card is 4111 1111 1111 1111."
    res_spaces = agent.redact(text_spaces)
    assert res_spaces.pii_detected
    assert "CREDIT_CARD" in res_spaces.detected_pii_types
    assert res_spaces.redacted_text == "My card is [REDACTED_CREDIT_CARD]."

def test_multiple_pii(agent):
    text = "Call 555-019-9238 or email test@test.com"
    res = agent.redact(text)
    assert res.pii_detected
    assert "EMAIL" in res.detected_pii_types
    assert "PHONE" in res.detected_pii_types
    assert "[REDACTED_PHONE]" in res.redacted_text
    assert "[REDACTED_EMAIL]" in res.redacted_text

def test_no_pii(agent):
    text = "How do I log in?"
    res = agent.redact(text)
    assert not res.pii_detected
    assert not res.detected_pii_types
    assert res.redacted_text == "How do I log in?"

def test_ssn_redaction(agent):
    text = "My SSN is 123-45-6789."
    res = agent.redact(text)
    assert "SSN" in res.detected_pii_types
    assert res.redacted_text == "My SSN is [REDACTED_SSN]."

def test_api_key_redaction(agent):
    text = "Here is my token: abcdef1234567890xyz"
    res = agent.redact(text)
    assert "API_KEY" in res.detected_pii_types
    assert res.redacted_text == "Here is my [REDACTED_API_KEY]"
