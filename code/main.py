import argparse
import sys
import os
from code.orchestration.batch_processor import BatchProcessor
from code.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="MLE Hiring Challenge - Triage Agent")
    parser.add_argument("--input", default="support_tickets/support_tickets.csv", help="Path to input CSV")
    parser.add_argument("--output", default="support_tickets/output.csv", help="Path to output CSV")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM for testing")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    processor = BatchProcessor(
        input_file=args.input,
        output_file=args.output,
        use_mock_llm=args.mock_llm,
        max_workers=args.workers
    )
    
    logger.info("Initializing Retrieval Corpus from data/ ...")
    processor.orchestrator.retrieval.initialize_corpus(["data/"])
    
    processor.process_all()
    
if __name__ == "__main__":
    main()
