import os
import json
from typing import List, Dict, Any, Optional
from code.utils.logger import get_logger
from code.schemas import ToolAction

logger = get_logger(__name__)

class ToolPlanningAgent:
    """
    Validates proposed tool actions against internal_tools.json schema,
    enforces prerequisites, and returns a sanitized JSON-safe ToolAction array.
    """
    DESTRUCTIVE_TOOLS = {"issue_refund", "modify_subscription", "lock_account"}
    
    def __init__(self, schema_path: str = "data/api_specs/internal_tools.json"):
        # Since we might be running tests from different directories,
        # fallback to an absolute path search if needed, but relative is fine.
        self.schema_path = schema_path
        self.allowed_tools = {}
        self._load_schema()
        
    def _load_schema(self):
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                tools = json.load(f)
                for tool in tools:
                    self.allowed_tools[tool["name"]] = tool
            logger.info(f"Loaded {len(self.allowed_tools)} tools from schema.")
        except Exception as e:
            logger.error(f"Failed to load tool schema from {self.schema_path}: {e}")
            raise
            
    def _validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """Validates parameters against the required fields in the schema."""
        schema = self.allowed_tools.get(tool_name)
        if not schema:
            return False
            
        required_fields = schema.get("parameters", {}).get("required", [])
        for field in required_fields:
            if field not in parameters:
                logger.warning(f"Tool {tool_name} missing required field: {field}")
                return False
        return True

    def validate_and_plan(self, proposed_actions: List[Dict[str, Any]], is_identity_verified: bool = False, target_email: str = "user@example.com") -> List[ToolAction]:
        """
        Validates proposed actions, filters unsupported ones, and enforces prerequisites.
        """
        valid_actions = []
        needs_verification = False
        
        for action in proposed_actions:
            tool_name = action.get("action")
            parameters = action.get("parameters", {})
            
            if not tool_name or tool_name not in self.allowed_tools:
                logger.warning(f"Rejected unsupported tool: {tool_name}")
                continue
                
            if not self._validate_parameters(tool_name, parameters):
                continue
                
            # Enforce prerequisites
            if tool_name in self.DESTRUCTIVE_TOOLS and not is_identity_verified:
                logger.info(f"Destructive tool '{tool_name}' blocked. Identity verification required.")
                needs_verification = True
                continue
                
            try:
                valid_actions.append(ToolAction(action=tool_name, parameters=parameters))
            except Exception as e:
                logger.error(f"Failed to instantiate ToolAction for {tool_name}: {e}")
                
        # If any destructive action was blocked due to unverified identity, prepend verify_identity
        if needs_verification:
            verify_action = ToolAction(
                action="verify_identity", 
                parameters={"method": "email_otp", "target": target_email}
            )
            # Place verification first
            valid_actions.insert(0, verify_action)
            
        return valid_actions
