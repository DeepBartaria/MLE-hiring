from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from enum import Enum

class StatusEnum(str, Enum):
    """Allowed values for status."""
    REPLIED = "replied"
    ESCALATED = "escalated"

class RequestTypeEnum(str, Enum):
    """Allowed values for request_type."""
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"

class RiskLevelEnum(str, Enum):
    """Allowed values for risk_level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConversationTurn(BaseModel):
    """Single turn in a conversation history."""
    role: str
    content: str

class TicketInput(BaseModel):
    """Schema for incoming support tickets."""
    issue: List[ConversationTurn]
    subject: str = ""
    company: str = "None"

class ParsedConversation(BaseModel):
    """Intermediate schema for parsed conversation state."""
    raw_subject: str
    raw_company: str
    cleaned_subject: str
    cleaned_company: str
    is_multi_turn: bool
    latest_user_message: str
    combined_user_context: str
    has_malformed_roles: bool
    original_turns: List[Dict[str, Any]]

class PIIResult(BaseModel):
    """Schema for PII detection and redaction."""
    pii_detected: bool
    detected_pii_types: List[str]
    redacted_text: str

class SafetyResult(BaseModel):
    """Schema for safety and PII detection results."""
    attack_detected: bool
    attack_type: Optional[str] = None
    pii_detected: bool = False
    detected_pii_types: List[str] = Field(default_factory=list)
    risk_level: RiskLevelEnum
    escalation_recommended: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str

class RetrievalResult(BaseModel):
    """Schema for RAG retrieval results."""
    retrieved_chunks: List[str]
    source_documents: List[str]
    retrieval_scores: List[float]
    
    @property
    def formatted_sources(self) -> str:
        """Returns pipe-separated file paths for output compliance."""
        return "|".join(self.source_documents)

class ClassificationResult(BaseModel):
    """Schema for classification."""
    product_area: str
    request_type: RequestTypeEnum
    language: str
    inferred_company: str
    confidence_score: float = Field(ge=0.0, le=1.0)

class EscalationDecision(BaseModel):
    """Schema for routing decision."""
    should_escalate: bool
    status: StatusEnum
    justification: str

class ToolAction(BaseModel):
    """Schema for API tool calls."""
    action: str
    parameters: Dict[str, Any]
    
    model_config = ConfigDict(strict=True)

class FinalTicketOutput(BaseModel):
    """Schema for the final required output. Validates against exact enums and bounds."""
    status: StatusEnum
    product_area: str
    response: str
    justification: str
    request_type: RequestTypeEnum
    confidence_score: float = Field(ge=0.0, le=1.0, description="Calibrated confidence")
    source_documents: str
    risk_level: RiskLevelEnum
    pii_detected: bool
    language: str
    actions_taken: List[ToolAction] = Field(default_factory=list)
    
    def to_csv_dict(self) -> dict:
        """
        Returns a dict ready to be written to the output CSV.
        Serializes enums to strings and ensures actions_taken is JSON-safe.
        """
        return self.model_dump(mode="json")
