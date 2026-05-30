# Support Triage Agent Architecture

This document outlines the architecture, logic, and design decisions behind the Multi-Agent Support Triage System built for the MLE Hiring Challenge.

## High-Level Architecture & Data Flow

The system uses a sequential, deterministic multi-agent pipeline orchestrated by `BatchProcessor` and `Orchestrator`. Each ticket passes through a specialized chain of agents, accumulating state in a shared context dictionary until a final status (`replied` or `escalated`) and response are synthesized.

### Agent Execution Graph:
1. **IntakeAgent**: Normalizes raw CSV input, lowercases headers, and structures the data into a `ParsedConversation` schema.
2. **SafetyAgent**: Screens queries against a highly sensitive adversarial heuristic system (prompt injection, prompt leaking, malformed JSON formats, hallucination triggers).
3. **PIIAgent**: Scans for emails, phone numbers, and SSNs. Extracts them for downstream masking.
4. **DomainRouter**: A deterministic heuristics-based classifier that categorizes the product area (Visa, Claude, DevPlatform) and assigns a confidence score.
5. **RetrievalAgent**: Fetches top-k relevant documents from the target product's markdown corpus. 
6. **ToolPlanningAgent**: Evaluates if internal tools (like `verify_identity` or `issue_refund`) should be conditionally staged for execution. Outputs schema-validated JSON actions.
7. **EscalationDecisionAgent**: The core router. Evaluates all accumulated context (safety, classification confidence, retrieval grounding, tool restrictions) to make a definitive binary decision: **Reply or Escalate**.
8. **ResponseGenerationAgent**: Calls the LLM (`DeepSeek V4 Flash`) to synthesize a grounded response. Includes robust fallback fail-safes.
9. **RedTeamAgent**: A final output validation layer that enforces compliance, checking for leaked PII, unauthorized tool calls, or unsafe outputs before finalizing the ticket.

## Retrieval Strategy

We implemented a **Hybrid Search Pipeline using Reciprocal Rank Fusion (RRF)**.
* **Dense Retrieval (FAISS + MiniLM-L6-v2):** Captures semantic meaning, synonyms, and user intent (e.g., "money back" -> "refund").
* **Sparse Retrieval (BM25):** Ensures precision for exact keyword matches, error codes, and strict policy terminology.

**Why RRF?**
RRF allows us to combine dense and sparse rankings mathematically without needing to manually tune weighted combinations (which are notoriously brittle across different domains). We retrieve the top `k*3` candidates from both indices, compute RRF scores, and then apply a novel mathematical normalization (`score /= (2.0 / (k + 1))`) to bind scores between `[0.0, 1.0]`. This bounded score is critical for the `EscalationDecisionAgent` to accurately measure corpus grounding.

## Safety & Adversarial Handling

The system treats safety as a distinct, isolated step at the very beginning (`SafetyAgent`) and the very end (`RedTeamAgent`) of the pipeline.
1. **Intake Check:** We employ heuristics to detect attempts to bypass system constraints (e.g., "ignore all previous instructions", "system:", "assistant:").
2. **Adversarial Roles:** The agent explicitly checks for malformed JSON or prompt leaking requests designed to extract internal variables.
3. **Graceful Failovers:** If an LLM rate-limit (`429`), network error, or invalid API key (`401/404`) occurs, the `ResponseAgent` catches the exception and falls back to a deterministic mock string, ensuring the batch process **never crashes**.

## Escalation Decision Logic

The `EscalationDecisionAgent` prioritizes safety over answering. A ticket is immediately flagged as `ESCALATED` if any of the following triggers hit:
* **Safety Violation:** The `SafetyAgent` identified critical risk.
* **Insufficient Grounding:** The top normalized RRF retrieval score is `< 0.3`. This prevents LLM hallucination.
* **Ambiguous Routing:** The `DomainRouter` confidence is `< 0.2`.
* **Conflicting Evidence:** If the top two retrieved documents are distinct files, but their RRF score delta is `< 0.01` AND they are highly grounded (`> 0.8`), it indicates extreme ambiguity in the knowledge base, warranting human intervention.

## Known Limitations and Failure Modes

* **Mock Fallback Dependency:** If API keys run out of quota (or are invalid), the system relies on a static mock string instead of generating dynamic context.
* **Heuristic Classification:** The `DomainRouter` currently relies on term frequencies and strict heuristics. While extremely fast and deterministic, highly complex multi-domain queries might confuse it compared to an LLM-based router.
* **Sync Bottlenecks:** The FAISS index loading and batch processing are currently executed using `ThreadPoolExecutor`. For significantly larger datasets (e.g., 100k+ tickets), migrating to `asyncio` with batch LLM endpoints would improve throughput.

## System Diagram (ASCII)

```text
 ┌─────────────────┐
 │   Input CSV     │
 └───────┬─────────┘
         │
 ┌───────▼─────────┐
 │  Intake Agent   │ (Normalization)
 └───────┬─────────┘
         │
 ┌───────▼─────────┐
 │  Safety Agent   │ (Adversarial Check) ──┐
 └───────┬─────────┘                       │
         │                                 │
 ┌───────▼─────────┐                       │
 │    PII Agent    │                       │
 └───────┬─────────┘                       │
         │                                 │
 ┌───────▼─────────┐                       │
 │  Domain Router  │                       │
 └───────┬─────────┘                       │
         │                                 │
 ┌───────▼─────────┐                       │ (Failsafe)
 │ Retrieval Agent │ (Hybrid RRF)          │
 └───────┬─────────┘                       │
         │                                 │
 ┌───────▼─────────┐                       │
 │ToolPlanningAgent│                       │
 └───────┬─────────┘                       │
         │                                 │
 ┌───────▼─────────┐                       │
 │ EscalationAgent │ ◄─────────────────────┘
 └───────┬─────────┘
         │ (If Replied)
 ┌───────▼─────────┐
 │ Response Agent  │ (LLM + Fallbacks)
 └───────┬─────────┘
         │
 ┌───────▼─────────┐
 │ Red Team Agent  │ (Output Validation)
 └───────┬─────────┘
         │
 ┌───────▼─────────┐
 │   Output CSV    │
 └─────────────────┘
```
