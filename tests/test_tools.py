import json
from typing import Any, Dict

import pytest
from pydantic_ai import RunContext

from agentgenius.builtin_tools import get_datetime
from agentgenius.tools import ToolDef, ToolSet


@pytest.fixture
def sample_tool():
    def sample_tool(question: str) -> str:
        """Sample tool that returns a fixed response"""
        return f"Response to: {question}"

    return sample_tool


@pytest.fixture
def sample_tool_with_deps():
    def sample_tool_with_deps(ctx: RunContext[Dict[str, Any]], question: str, deps: Dict[str, Any]) -> Dict[str, Any]:
        """Sample tool that uses dependencies"""
        return {"question": question, "deps": deps}

    return sample_tool_with_deps


@pytest.fixture
def basic_toolset(sample_tool):
    return ToolSet([sample_tool])


class TestToolDef:
    def test_create_basic_tool(self, sample_tool):
        """Test creating a basic ToolDef"""
        tool_def = ToolDef(name=sample_tool.__name__)
        assert tool_def.name == sample_tool.__name__

    def test_serialization(self, sample_tool):
        """Test JSON serialization and deserialization"""
        tool_def = ToolDef(name=sample_tool.__name__)
        json_str = tool_def.model_dump_json()
        data = json.loads(json_str)

        # Check serialized data
        assert data["name"] == sample_tool.__name__

        # Test deserialization
        tool_def2 = ToolDef.model_validate_json(json_str)
        assert tool_def2.name == tool_def.name


class TestToolSet:
    def test_create_empty_toolset(self):
        """Test creating an empty ToolSet"""
        toolset = ToolSet()
        assert len(toolset.tools) == 0

    def test_create_toolset_with_tool(self, sample_tool):
        """Test creating a ToolSet with a single tool"""
        toolset = ToolSet([sample_tool])
        assert len(toolset.tools) == 1
        assert toolset.tools[0].name == sample_tool.__name__

    def test_create_toolset_with_multiple_tools(self, sample_tool, sample_tool_with_deps):
        """Test creating a ToolSet with multiple tools"""
        toolset = ToolSet([sample_tool, sample_tool_with_deps])
        assert len(toolset.tools) == 2
        tool_names = {tool.name for tool in toolset.tools}
        assert tool_names == {sample_tool.__name__, sample_tool_with_deps.__name__}

    def test_add_tool(self, basic_toolset, sample_tool_with_deps):
        """Test adding a tool to ToolSet"""
        basic_toolset.add(sample_tool_with_deps)
        assert len(basic_toolset.tools) == 2
        assert any(t.name == sample_tool_with_deps.__name__ for t in basic_toolset.tools)

    def test_remove_tool(self, basic_toolset, sample_tool):
        """Test removing a tool from ToolSet"""
        basic_toolset.remove(sample_tool.__name__)
        assert len(basic_toolset.tools) == 0

    def test_get_tool(self, basic_toolset, sample_tool):
        """Test getting a tool by name"""
        tool = basic_toolset.get(sample_tool.__name__)
        assert tool.__name__ == sample_tool.__name__

    def test_get_nonexistent_tool(self, basic_toolset):
        """Test getting a nonexistent tool returns None"""
        assert basic_toolset.get("nonexistent") is None

    def test_contains(self, basic_toolset, sample_tool):
        """Test checking if a tool is in ToolSet"""
        assert sample_tool.__name__ in basic_toolset
        assert "nonexistent" not in basic_toolset

    def test_serialization(self, basic_toolset):
        """Test JSON serialization and deserialization"""
        json_str = basic_toolset.model_dump_json()
        data = json.loads(json_str)

        # Check serialized data
        assert len(data["tools"]) == 1

        # Test deserialization
        toolset2 = ToolSet.model_validate_json(json_str)
        assert len(toolset2.tools) == len(basic_toolset.tools)
        assert toolset2.tools[0].name == basic_toolset.tools[0].name

    def test_builtin_get_datetime_tool(self):
        """Test the built-in get_time tool"""
        toolset = ToolSet([get_datetime])
        assert len(toolset.tools) == 1
        assert toolset.tools[0].name == "get_datetime"

    def test_duplicate_tool_name(self, sample_tool):
        """Test that adding a tool with a duplicate name raises ValueError"""
        with pytest.raises(ValueError, match=f"Tool {sample_tool.__name__} already exists in ToolSet"):
            ToolSet([sample_tool, sample_tool])  # Try to add two tools with same name

        toolset = ToolSet([sample_tool])
        with pytest.raises(ValueError, match=f"Tool {sample_tool.__name__} already exists in ToolSet"):
            toolset.add(sample_tool)  # Try to add second tool with same name
