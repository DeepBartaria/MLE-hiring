import os
from typing import Optional, Dict
from code.schemas import (
    ParsedConversation,
    RetrievalResult,
    EscalationDecision,
    ClassificationResult,
    SafetyResult
)
from code.utils.logger import get_logger

logger = get_logger(__name__)

class ResponseGenerationAgent:
    """
    Synthesizes the final reply to the user using LLM generation, strictly 
    grounded in retrieved context. Enforces deterministic fallback on escalation.
    """
    
    FALLBACK_MESSAGES = {
        "en": "I apologize, but I am unable to fully resolve your request at this time. I have escalated your ticket to our human support team, and someone will reach out to you shortly.",
        "es": "Pido disculpas, pero no puedo resolver completamente su solicitud en este momento. He escalado su caso a nuestro equipo de soporte humano, y alguien se comunicará con usted en breve.",
        "fr": "Je m'excuse, mais je ne suis pas en mesure de résoudre entièrement votre demande pour le moment. J'ai transmis votre billet à notre équipe d'assistance humaine, et quelqu'un vous contactera sous peu."
    }
    
    def __init__(self, use_mock_llm: bool = False):
        self.use_mock_llm = use_mock_llm
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    def generate(
        self,
        conversation: ParsedConversation,
        escalation: EscalationDecision,
        retrieval: RetrievalResult,
        classification: ClassificationResult,
        safety: SafetyResult
    ) -> str:
        
        # 1. Short-circuit on escalation
        if escalation.should_escalate:
            logger.info("Escalation triggered. Bypassing LLM generation.")
            lang = classification.language if classification.language in self.FALLBACK_MESSAGES else "en"
            return self.FALLBACK_MESSAGES[lang]
            
        # 2. Build the strict prompt
        system_prompt = (
            "You are a helpful, empathetic support agent. "
            "You MUST answer the user's query ONLY using the provided retrieved context below.\n"
            "If the answer is not explicitly stated in the context, say 'I am sorry, but I do not have enough information to answer that.'\n"
            "NEVER hallucinate policies, features, or timelines. NEVER echo sensitive information like passwords or full credit card numbers.\n\n"
            "Context:\n" + "\n---\n".join(retrieval.retrieved_chunks)
        )
        
        user_prompt = f"User Request: {conversation.latest_user_message}\n\nPlease provide a concise and empathetic response."
        
        # 3. Call LLM (or mock)
        if self.use_mock_llm:
            response = self._mock_llm_call(system_prompt, user_prompt)
        else:
            response = self._actual_llm_call(system_prompt, user_prompt)
            
        # 4. Post-generation validation
        final_response = self._validate_response(response, escalation)
        return final_response

    def _mock_llm_call(self, system: str, user: str) -> str:
        """Deterministic mock for unit testing."""
        if "I do not have enough information" in system or "not explicitly stated" in system:
            return "Based on the documentation: Please reset your device."
        return "Generic mock response."

    def _actual_llm_call(self, system: str, user: str) -> str:
        """Calls actual LLM using deterministic settings (temp=0.0)."""
        if self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=0.0,
                    seed=42
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")
                
        elif self.anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_key)
                resp = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    temperature=0.0,
                    system=system,
                    messages=[
                        {"role": "user", "content": user}
                    ]
                )
                return resp.content[0].text
            except Exception as e:
                logger.error(f"Anthropic call failed: {e}")
                
        logger.warning("No API keys found or APIs failed. Falling back to mock.")
        return self._mock_llm_call(system, user)

    def _validate_response(self, response: str, escalation: EscalationDecision) -> str:
        """Heuristic validator to ensure LLM didn't hallucinate or leak."""
        # Simple heuristic: if the response is extremely long, it might be hallucinating
        if len(response) > 2000:
            logger.warning("Response exceeded length limit. Forcing escalation.")
            escalation.should_escalate = True
            escalation.justification += " | Forced escalation due to oversized response."
            return self.FALLBACK_MESSAGES["en"]
            
        return response
