import json
import pytest
from agentgenius.agents import BaseAgent
from agentgenius.tools import ToolSet
from pydantic_ai import RunContext


@pytest.fixture
def mock_tool():
    def tool(ctx: RunContext[str], question: str) -> str:
        """Mock tool that returns a fixed response"""
        return "mock response"

    return tool


@pytest.fixture
def agent(mock_tool):
    tools = ToolSet(mock_tool)
    result = BaseAgent(name="test", model="test", system_prompt="You are a helpful assistant.", toolset=tools)
    yield result
    result = None


def test_agent_creation(agent):
    """Test that agent is created with correct attributes"""
    assert agent.model == "test"
    assert agent.get_system_prompt() == "You are a helpful assistant."
    assert isinstance(agent.toolset, ToolSet)


def test_agent_run_sync(agent):
    """Test synchronous run method"""
    result = agent.run_sync("test question")
    assert result is not None


def test_agent_serialization(agent):
    """Test JSON serialization/deserialization"""
    json_str = agent.to_json()
    data = json.loads(json_str)

    # Check serialized data
    assert data["model"] == "test"
    assert data["system_prompt"] == "You are a helpful assistant."
    assert isinstance(data["toolset"], list)

    # Test deserialization
    namespace = {"tool": agent.toolset.toolset[0]}  # Pass the original tool in namespace
    agent2 = BaseAgent.from_json(json_str, namespace=namespace)

    assert agent2.model == agent.model
    assert agent2.get_system_prompt() == agent.get_system_prompt()
    assert isinstance(agent2.toolset, ToolSet)


@pytest.mark.asyncio
async def test_agent_run_async(agent):
    """Test asynchronous run method"""
    result = await agent.run("test question")
    assert result is not None
