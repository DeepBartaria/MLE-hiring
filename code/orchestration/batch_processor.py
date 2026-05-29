import csv
import concurrent.futures
import json
from typing import List, Dict
from code.utils.logger import get_logger
from code.orchestration.orchestrator import TriageOrchestrator

logger = get_logger(__name__)

class BatchProcessor:
    def __init__(self, input_file: str, output_file: str, use_mock_llm: bool = False, max_workers: int = 10):
        self.input_file = input_file
        self.output_file = output_file
        self.max_workers = max_workers
        self.orchestrator = TriageOrchestrator(use_mock_llm=use_mock_llm)
        
    def process_all(self):
        logger.info(f"Starting batch processing of {self.input_file}")
        rows = self._read_csv()
        
        results = [None] * len(rows)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.orchestrator.process_ticket, row): i 
                for i, row in enumerate(rows)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    res = future.result(timeout=60)
                    results[index] = res.to_csv_dict()
                except Exception as e:
                    logger.error(f"Row {index} failed or timed out: {e}")
                    results[index] = self.orchestrator._build_safe_fallback().to_csv_dict()
                    
        self._write_csv(results)
        logger.info(f"Finished batch processing. Wrote {len(results)} rows to {self.output_file}")
        
    def _read_csv(self) -> List[Dict[str, str]]:
        rows = []
        try:
            with open(self.input_file, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            logger.error(f"Failed to read input CSV: {e}")
        return rows
        
    def _write_csv(self, results: List[Dict]):
        if not results:
            return
            
        fieldnames = list(results[0].keys())
        try:
            with open(self.output_file, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for res in results:
                    # actions_taken must be serialized to JSON string for CSV
                    if isinstance(res.get("actions_taken"), list):
                        res["actions_taken"] = json.dumps(res["actions_taken"])
                    writer.writerow(res)
        except Exception as e:
            logger.error(f"Failed to write output CSV: {e}")
