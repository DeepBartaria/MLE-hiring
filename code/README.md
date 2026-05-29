# Code Architecture

This directory contains the source code for the Support Triage Agent. 

## Structure Overview

- **`agents/`**: Contains specialized agents that perform distinct tasks (e.g., `RetrievalAgent`, `SafetyAgent`, `RoutingAgent`).
- **`retrieval/`**: Logic for reading, chunking, and retrieving information from the local `data/` support corpus.
- **`safety/`**: Critical guardrail mechanisms (PII detection, prompt injection shielding).
- **`orchestration/`**: Main agent workflow that coordinates safety checks, routing, and retrieval.
- **`tools/`**: Structured schema validation for required API actions.
- **`utils/`**: Shared utilities like deterministic seeding and centralized logging.
- **`schemas.py`**: Pydantic models enforcing rigid I/O contracts.
- **`config.py`**: Typed configuration managing secrets via environment variables securely.

## Execution Requirements

This pipeline is designed to be fully deterministic and safe against adversarial test vectors. No secrets are hardcoded.

```bash
pip install -r ../requirements.txt
python -m code.main --input ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv
```
