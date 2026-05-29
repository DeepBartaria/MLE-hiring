import os
import json
import pytest
from code.agents.tool_agent import ToolPlanningAgent

@pytest.fixture
def temp_schema(tmp_path):
    schema = [
        {
            "name": "issue_refund",
            "parameters": {"required": ["transaction_id", "amount", "reason"]}
        },
        {
            "name": "verify_identity",
            "parameters": {"required": ["method", "target"]}
        },
        {
            "name": "reset_password",
            "parameters": {"required": ["user_email"]}
        }
    ]
    schema_path = tmp_path / "internal_tools.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f)
    return str(schema_path)

def test_valid_tool_action(temp_schema):
    agent = ToolPlanningAgent(schema_path=temp_schema)
    proposed = [{"action": "reset_password", "parameters": {"user_email": "test@test.com"}}]
    
    actions = agent.validate_and_plan(proposed)
    assert len(actions) == 1
    assert actions[0].action == "reset_password"

def test_unsupported_tool_dropped(temp_schema):
    agent = ToolPlanningAgent(schema_path=temp_schema)
    proposed = [{"action": "delete_database", "parameters": {"force": True}}]
    
    actions = agent.validate_and_plan(proposed)
    assert len(actions) == 0

def test_destructive_action_requires_verification(temp_schema):
    agent = ToolPlanningAgent(schema_path=temp_schema)
    proposed = [{"action": "issue_refund", "parameters": {"transaction_id": "123", "amount": 10, "reason": "fraud"}}]
    
    # Not verified
    actions = agent.validate_and_plan(proposed, is_identity_verified=False)
    assert len(actions) == 1
    assert actions[0].action == "verify_identity" # Injected
    
    # Verified
    actions = agent.validate_and_plan(proposed, is_identity_verified=True)
    assert len(actions) == 1
    assert actions[0].action == "issue_refund"

def test_missing_parameters(temp_schema):
    agent = ToolPlanningAgent(schema_path=temp_schema)
    proposed = [{"action": "issue_refund", "parameters": {"amount": 10}}] # Missing transaction_id and reason
    
    actions = agent.validate_and_plan(proposed, is_identity_verified=True)
    assert len(actions) == 0
