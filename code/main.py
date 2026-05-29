import argparse
import sys
from code.utils.seed import set_seed
from code.utils.logger import get_logger
from code.config import settings

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    parser.add_argument("--input", type=str, default="support_tickets/support_tickets.csv", 
                        help="Path to the input tickets CSV file")
    parser.add_argument("--output", type=str, default="support_tickets/output.csv", 
                        help="Path to write the output predictions CSV")
    return parser.parse_args()

def main():
    """
    Main entrypoint for the CLI-based triage agent.
    """
    args = parse_args()
    
    # 1. Guarantee determinism
    set_seed(settings.random_seed)
    logger.info("Initializing Support Triage Agent...")
    logger.info(f"Using random seed: {settings.random_seed}")
    
    # 2. Scaffolding for Component Initialization
    # TODO: Instantiate RetrievalAgent, SafetyAgent, RoutingAgent, Orchestrator
    
    # 3. Data Ingestion
    logger.info(f"Loading input tickets from {args.input}")
    # TODO: Load dataset and parse into schemas.SupportTicket objects
    
    # 4. Processing Loop
    logger.info("Starting processing loop...")
    # TODO: Iterate through tickets, invoke Orchestrator, aggregate results
    
    # 5. Output Generation
    logger.info(f"Writing structured output to {args.output}")
    # TODO: Convert results to DataFrame and save as CSV
    
    logger.info("Evaluation run complete.")

if __name__ == "__main__":
    main()
