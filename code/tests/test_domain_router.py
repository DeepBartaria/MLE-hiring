import pytest
from code.schemas import ParsedConversation
from code.agents.domain_router import DomainRouter
from code.agents.visa_agent import VisaAgent
from code.agents.claude_agent import ClaudeAgent
from code.agents.devplatform_agent import DevPlatformAgent

def create_conversation(text: str, company: str = "None") -> ParsedConversation:
    return ParsedConversation(
        raw_subject="Test",
        raw_company=company,
        cleaned_subject="Test",
        cleaned_company=company,
        is_multi_turn=False,
        latest_user_message=text,
        combined_user_context=text,
        has_malformed_roles=False,
        original_turns=[]
    )

def test_misleading_metadata():
    router = DomainRouter()
    
    # Text is clearly Claude, but metadata is Visa
    conv = create_conversation("My Claude Opus model is hallucinating prompts.", "Visa")
    agent, result = router.route(conv)
    
    assert isinstance(agent, ClaudeAgent)
    assert result.inferred_company == "Claude"
    assert result.confidence_score > 0.5

def test_compound_ticket():
    router = DomainRouter()
    
    # Mix of DevPlatform and Visa
    conv = create_conversation("My webhook is timing out and my credit card was charged twice.", "None")
    agent, result = router.route(conv)
    
    # Should pick one depending on exact weights, but should not fail
    assert agent is not None
    assert result.inferred_company in ["DevPlatform", "Visa"]
    assert result.confidence_score > 0.0

def test_ambiguous_ticket():
    router = DomainRouter()
    
    # Generic complaint with no keywords
    conv = create_conversation("This product is terrible and I want my money back.", "None")
    agent, result = router.route(conv)
    
    # Since no keywords match and company is None, it should fallback
    assert agent is None
    assert result.inferred_company == "Unknown"
    assert result.confidence_score < 0.1
