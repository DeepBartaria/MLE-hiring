# Support Ticket Triage Agent

This is a robust, deterministic, multi-agent system designed to automatically evaluate and triage support tickets while remaining highly resistant to prompt injections, role spoofing, and PII leakage.

## Architecture

The system executes a deterministic sequence of guardrail agents:
1. **Intake Agent**: Sanitizes adversarial payloads, unicode attacks, and malformed JSON.
2. **Safety Agent**: Runs regex heuristics to detect jailbreaks and data exfiltration.
3. **PII Agent**: Deterministically redacts sensitive entities (Email, Phone, Credit Cards, etc.).
4. **Domain Router**: Ignores adversarial metadata and routes tickets based on semantic content (Visa, Claude, DevPlatform).
5. **Retrieval Agent**: Fetches highly relevant contextual chunks using a Hybrid BM25/FAISS index.
6. **Tool Planning Agent**: Validates deterministic schemas to block unauthorized destructive tool usage.
7. **Escalation Agent**: A decision-tree engine that forces ticket escalation on low confidence, poor grounding, or high risk.
8. **Response Agent**: Synthesizes the final empathetic LLM response exclusively from retrieved context.
9. **Red Team Agent**: Audits the final output schema for hallucinated citations, PII leaks, or prompt leakage, forcing an escalation if any occur.

## Setup & Installation

1. Create a Python virtual environment and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API Keys (for live LLM generation):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

## Running the Agent

You can run the agent over any CSV file containing `issue`, `subject`, and `company` headers. 
By default, the script processes `support_tickets/support_tickets.csv` and writes to `support_tickets/output.csv`.

To run the agent using the Mock LLM (for fast offline testing):
```bash
PYTHONPATH=. python code/main.py --input support_tickets/sample_support_tickets.csv --output support_tickets/output.csv --mock-llm
```

To run the agent in Production Mode (calls OpenAI/Anthropic APIs):
```bash
PYTHONPATH=. python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

### Advanced Arguments
- `--workers <int>`: Defines the number of concurrent threads used to process tickets (Default: 10). Increase this for faster batch processing on powerful machines.
- `--mock-llm`: Bypasses external API calls and uses deterministic placeholder strings.
