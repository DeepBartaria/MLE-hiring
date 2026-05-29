from typing import Tuple, Optional
from code.schemas import ParsedConversation, ClassificationResult, RequestTypeEnum
from code.agents.domain_base import DomainAgent
from code.agents.visa_agent import VisaAgent
from code.agents.claude_agent import ClaudeAgent
from code.agents.devplatform_agent import DevPlatformAgent
from code.utils.logger import get_logger

logger = get_logger(__name__)

class DomainRouter:
    """
    Evaluates incoming conversations and routes them to the specialized domain agent
    based on deterministic keyword analysis rather than trusting raw metadata.
    """
    def __init__(self):
        self.agents: list[DomainAgent] = [
            VisaAgent(),
            ClaudeAgent(),
            DevPlatformAgent()
        ]
        
    def route(self, conversation: ParsedConversation) -> Tuple[Optional[DomainAgent], ClassificationResult]:
        best_agent = None
        best_score = -1.0
        
        # Calculate scores
        scores = {}
        for agent in self.agents:
            score = agent.match_score(conversation)
            scores[agent.get_company_name()] = score
            if score > best_score:
                best_score = score
                best_agent = agent
                
        logger.info(f"Domain routing scores: {scores}")
        
        # Ambiguous fallback
        if best_score < 0.1:
            logger.warning("Ambiguous routing detected. Falling back to generic classification.")
            generic_result = ClassificationResult(
                product_area="General Support",
                request_type=RequestTypeEnum.PRODUCT_ISSUE,
                language="en",
                inferred_company="Unknown",
                confidence_score=0.0
            )
            return None, generic_result
            
        # Classify using the chosen agent
        classification = best_agent.classify(conversation, confidence=best_score)
        logger.info(f"Routed to {best_agent.get_company_name()} with confidence {best_score}")
        
        return best_agent, classification
