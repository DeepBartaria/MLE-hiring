import traceback
from typing import Dict, Any
from code.schemas import FinalTicketOutput, StatusEnum, RequestTypeEnum, RiskLevelEnum
from code.agents.intake_agent import IntakeAgent
from code.agents.safety_agent import SafetyAgent
from code.agents.pii_agent import PIIAgent
from code.agents.domain_router import DomainRouter
from code.agents.retrieval_agent import RetrievalAgent
from code.agents.tool_agent import ToolPlanningAgent
from code.agents.escalation_agent import EscalationAgent
from code.agents.response_agent import ResponseGenerationAgent
from code.agents.red_team_agent import RedTeamAgent
from code.utils.logger import get_logger

logger = get_logger(__name__)

class TriageOrchestrator:
    def __init__(self, use_mock_llm: bool = False):
        self.intake = IntakeAgent()
        self.safety = SafetyAgent()
        self.pii = PIIAgent()
        self.router = DomainRouter()
        self.retrieval = RetrievalAgent()
        self.tool_planner = ToolPlanningAgent()
        self.escalator = EscalationAgent()
        self.responder = ResponseGenerationAgent(use_mock_llm=use_mock_llm)
        self.red_team = RedTeamAgent()
        
    def process_ticket(self, row: Dict[str, str]) -> FinalTicketOutput:
        try:
            # 1. Intake
            parsed = self.intake.process_row(row)
            
            # 2. Safety
            safety_res = self.safety.analyze_conversation(parsed)
            
            # 3. PII (Update safety result with PII)
            pii_res = self.pii.redact(parsed.combined_user_context)
            if pii_res.pii_detected:
                safety_res.pii_detected = True
                safety_res.detected_pii_types.extend(pii_res.detected_pii_types)
                # Ensure the context going downstream is redacted
                parsed.combined_user_context = pii_res.redacted_text
                parsed.latest_user_message = self.pii.redact(parsed.latest_user_message).redacted_text
                
            # 4. Domain Routing
            agent, class_res = self.router.route(parsed)
            
            # 5. Retrieval
            if agent:
                queries = agent.get_retrieval_queries(parsed)
                retrieval_res = self.retrieval.retrieve(queries)
            else:
                retrieval_res = self.retrieval.retrieve([parsed.latest_user_message])
                
            # 6. Tool Planner (Heuristic mock proposed tools)
            # A real LLM might propose tools. For deterministic logic, we use heuristic triggers.
            proposed_tools = []
            if agent and class_res.request_type == RequestTypeEnum.PRODUCT_ISSUE:
                if "refund" in parsed.combined_user_context.lower():
                    proposed_tools.append({"action": "issue_refund", "parameters": {"transaction_id": "unknown", "amount": 0, "reason": "customer_request"}})
            
            tool_actions = self.tool_planner.validate_and_plan(proposed_tools)
            
            # 7. Escalation
            escalation_decision = self.escalator.evaluate(safety_res, class_res, retrieval_res)
            
            # 8. Response Generation
            response_text = self.responder.generate(parsed, escalation_decision, retrieval_res, class_res, safety_res)
            
            # Construct Final Output
            final_output = FinalTicketOutput(
                status=escalation_decision.status,
                product_area=class_res.product_area,
                response=response_text,
                justification=escalation_decision.justification,
                request_type=class_res.request_type,
                confidence_score=class_res.confidence_score,
                source_documents=retrieval_res.formatted_sources,
                risk_level=safety_res.risk_level,
                pii_detected=safety_res.pii_detected,
                language=class_res.language,
                actions_taken=tool_actions
            )
            
            # 9. Red Team Validation
            audited_output = self.red_team.audit(final_output)
            return audited_output
            
        except Exception as e:
            logger.error(f"Catastrophic failure processing ticket: {e}\n{traceback.format_exc()}")
            return self._build_safe_fallback()
            
    def _build_safe_fallback(self) -> FinalTicketOutput:
        return FinalTicketOutput(
            status=StatusEnum.ESCALATED,
            product_area="Unknown",
            response="I apologize, but I am unable to fully resolve your request at this time. I have escalated your ticket to our human support team.",
            justification="System exception during pipeline execution.",
            request_type=RequestTypeEnum.INVALID,
            confidence_score=0.0,
            source_documents="",
            risk_level=RiskLevelEnum.CRITICAL,
            pii_detected=False,
            language="en",
            actions_taken=[]
        )
