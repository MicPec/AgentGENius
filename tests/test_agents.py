import json

import pytest
from pydantic import ValidationError

from agentgenius.agents import AgentDef, AgentParams


@pytest.fixture
def basic_agent_params():
    return AgentParams(
        result_type=str,
        retries=2,
    )


@pytest.fixture
def basic_agent_def(basic_agent_params):
    return AgentDef(
        model="openai:gpt-4o",
        name="TestAgent",
        system_prompt="You are a test assistant.",
        params=basic_agent_params,
    )


class TestAgentParams:
    def test_create_basic_params(self):
        """Test creating basic AgentParams with minimal required fields"""
        params = AgentParams(result_type=str)
        assert params.result_type is str

    def test_create_full_params(self):
        """Test creating AgentParams with all fields"""
        params = AgentParams(
            result_type=str,
            deps_type=int,
            retries=5,
            result_tool_name="custom_result",
            result_tool_description="Custom result tool",
            result_retries=2,
            defer_model_check=True,
            end_strategy="exhaustive",
        )
        assert params.result_type is str
        assert params.deps_type is int
        assert params.retries == 5
        assert params.result_tool_name == "custom_result"
        assert params.result_tool_description == "Custom result tool"
        assert params.result_retries == 2
        assert params.defer_model_check is True
        assert params.end_strategy == "exhaustive"

    def test_invalid_end_strategy(self):
        """Test that invalid end_strategy raises ValidationError"""
        with pytest.raises(ValidationError):
            AgentParams(result_type=str, end_strategy="invalid")

    def test_serialization(self, basic_agent_params):
        """Test JSON serialization and deserialization"""
        json_str = basic_agent_params.model_dump_json()
        data = json.loads(json_str)

        # Check serialized data
        assert data["result_type"] == "str"
        assert data["retries"] == 2

        # Test deserialization
        params2 = AgentParams.model_validate_json(json_str)
        assert params2.result_type is str
        assert params2.retries == 2


class TestAgentDef:
    def test_create_basic_agent(self, basic_agent_def):
        """Test creating basic AgentDef with minimal required fields"""
        assert basic_agent_def.model == "openai:gpt-4o"
        assert basic_agent_def.name == "TestAgent"
        assert basic_agent_def.system_prompt == "You are a test assistant."
        assert isinstance(basic_agent_def.params, AgentParams)

    def test_create_agent_without_params(self):
        """Test creating AgentDef without params"""
        agent = AgentDef(model="openai:gpt-4o", name="TestAgent", system_prompt="Test prompt")
        assert agent.params is None

    def test_invalid_model(self):
        """Test that invalid model raises ValidationError"""
        with pytest.raises(ValidationError):
            AgentDef(model="invalid:model", name="TestAgent", system_prompt="Test prompt")

    def test_serialization(self, basic_agent_def):
        """Test JSON serialization and deserialization"""
        json_str = basic_agent_def.model_dump_json()
        data = json.loads(json_str)

        # Check serialized data
        assert data["model"] == "openai:gpt-4o"
        assert data["name"] == "TestAgent"
        assert data["system_prompt"] == "You are a test assistant."
        assert "params" in data

        # Test deserialization
        agent2 = AgentDef.model_validate_json(json_str)
        assert agent2.model == basic_agent_def.model
        assert agent2.name == basic_agent_def.name
        assert agent2.system_prompt == basic_agent_def.system_prompt
        assert isinstance(agent2.params, AgentParams)

    def test_get_system_prompt(self, basic_agent_def):
        """Test getting system prompt"""
        assert basic_agent_def.system_prompt == "You are a test assistant."

    @pytest.mark.parametrize("model", ["openai:gpt-4o-mini", "openai:gpt-4o", "test"])
    def test_valid_models(self, model):
        """Test all valid model options"""
        agent = AgentDef(model=model, name="TestAgent", system_prompt="Test prompt")
        assert agent.model == model
