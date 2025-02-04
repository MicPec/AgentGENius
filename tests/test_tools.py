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

    def test_tool_duplicate_direct(self, basic_toolset):
        """Test replacing an existing tool with a new one"""

        def tool():
            return "first"

        def new_tool():
            return "second"

        # Add first tool
        basic_toolset.add(tool)
        original_length = len(basic_toolset.tools)

        # Add tool with same name
        new_tool.__name__ = tool.__name__  # Make names match
        basic_toolset.add(new_tool)

        # Verify tool was replaced
        assert len(basic_toolset.tools) == original_length
        assert basic_toolset.get(tool.__name__)() == "first"

    def test_tool_duplicate_in_list(self, basic_toolset):
        """Test replacing tools when adding from a list"""

        def tool1():
            return "original1"

        def tool2():
            return "original2"

        # Add original tools
        basic_toolset.add([tool1, tool2])
        original_length = len(basic_toolset.tools)

        # Create replacement tools with same names
        def new_tool1():
            return "replaced1"

        def new_tool2():
            return "replaced2"

        new_tool1.__name__ = tool1.__name__
        new_tool2.__name__ = tool2.__name__

        # Replace tools
        basic_toolset.add([new_tool1, new_tool2])

        # Verify tools were replaced
        assert len(basic_toolset.tools) == original_length
        assert basic_toolset.get(tool1.__name__)() == "original1"
        assert basic_toolset.get(tool2.__name__)() == "original2"

    def test_tool_duplicate_in_dict(self, basic_toolset):
        """Test replacing tools when adding from a dictionary"""

        def tool1():
            return "original1"

        def tool2():
            return "original2"

        # Add original tools
        basic_toolset.add({"t1": tool1, "t2": tool2})
        original_length = len(basic_toolset.tools)

        # Create replacement tools with same names
        def new_tool1():
            return "replaced1"

        def new_tool2():
            return "replaced2"

        new_tool1.__name__ = tool1.__name__
        new_tool2.__name__ = tool2.__name__

        # Replace tools
        basic_toolset.add({"new_t1": new_tool1, "new_t2": new_tool2})

        # Verify tools were replaced
        assert len(basic_toolset.tools) == original_length
        assert basic_toolset.get(tool1.__name__)() == "original1"
        assert basic_toolset.get(tool2.__name__)() == "original2"

    def test_tool_duplicate_in_nested_structure(self, basic_toolset):
        """Test replacing tools in nested structure"""

        def tool1():
            return "original1"

        def tool2():
            return "original2"

        def tool3():
            return "original3"

        # Add original tools
        basic_toolset.add({"level1": [tool1, tool2], "level2": {"inner": tool3}})
        original_length = len(basic_toolset.tools)

        # Create replacement tools with same names
        def new_tool1():
            return "replaced1"

        def new_tool2():
            return "replaced2"

        def new_tool3():
            return "replaced3"

        new_tool1.__name__ = tool1.__name__
        new_tool2.__name__ = tool2.__name__
        new_tool3.__name__ = tool3.__name__

        # Replace tools
        basic_toolset.add({"new_level1": {"inner": [new_tool1, new_tool2]}, "new_level2": new_tool3})

        # Verify tools were replaced
        assert len(basic_toolset.tools) == original_length
        assert basic_toolset.get(tool1.__name__)() == "original1"
        assert basic_toolset.get(tool2.__name__)() == "original2"
        assert basic_toolset.get(tool3.__name__)() == "original3"

    def test_add_tool_callable(self, basic_toolset):
        """Test adding a tool as a callable"""

        def new_tool():
            """A new test tool"""
            return "new tool result"

        basic_toolset.add(new_tool)
        assert new_tool.__name__ in basic_toolset
        assert len(basic_toolset.tools) == 2

        # Verify we can get and call the tool
        tool_func = basic_toolset.get(new_tool.__name__)
        assert tool_func() == "new tool result"

    def test_add_tool_list_of_callables(self, basic_toolset):
        """Test adding multiple tools as a list of callables"""

        def tool1():
            return "tool1"

        def tool2():
            return "tool2"

        def tool3():
            return "tool3"

        basic_toolset.add([tool1, tool2, tool3])
        assert len(basic_toolset.tools) == 4
        assert all(t.__name__ in basic_toolset for t in [tool1, tool2, tool3])

    def test_add_tool_dict_of_callables(self, basic_toolset):
        """Test adding tools from a dictionary of callables"""

        def tool1():
            return "tool1"

        def tool2():
            return "tool2"

        tools_dict = {"key1": tool1, "key2": tool2}
        basic_toolset.add(tools_dict)
        assert len(basic_toolset.tools) == 3
        assert all(t.__name__ in basic_toolset for t in [tool1, tool2])

    def test_add_tool_nested_structures(self, basic_toolset):
        """Test adding tools with nested structures"""

        def tool1():
            return "tool1"

        def tool2():
            return "tool2"

        def tool3():
            return "tool3"

        nested_structure = {"level1": [tool1, tool2], "level2": {"inner": tool3}}
        basic_toolset.add(nested_structure)
        assert len(basic_toolset.tools) == 4
        assert all(t.__name__ in basic_toolset for t in [tool1, tool2, tool3])

    def test_add_invalid_tool_type(self, basic_toolset):
        """Test adding a tool with invalid type"""
        with pytest.raises(ValueError, match="Tool must be a callable or a string"):
            basic_toolset.add(123)  # Try to add an integer

        with pytest.raises(ValueError, match="Tool must be a callable or a string"):
            basic_toolset.add(None)  # Try to add None

    def test_add_empty_structures(self, basic_toolset):
        """Test adding empty structures"""
        original_length = len(basic_toolset.tools)

        basic_toolset.add([])  # Empty list
        assert len(basic_toolset.tools) == original_length

        basic_toolset.add({})  # Empty dict
        assert len(basic_toolset.tools) == original_length

    def test_add_duplicate_in_structure(self, basic_toolset):
        """Test adding duplicate tools within a structure"""

        def new_tool():
            return "new tool"

        # Add tool first time
        basic_toolset.add(new_tool)
        original_length = len(basic_toolset.tools)

        # Add duplicates in structure
        basic_toolset.add(["new_tool", "new_tool"])  # Duplicate in list
        assert len(basic_toolset.tools) == original_length  # Length should not change

        basic_toolset.add(
            {
                "key1": new_tool,
                "key2": new_tool,  # Duplicate in dict values
            }
        )
        assert len(basic_toolset.tools) == original_length  # Length should not change

    def test_duplicate_tool_direct_add(self, basic_toolset):
        """Test adding the same tool directly multiple times"""

        def duplicate_tool():
            return "duplicate"

        # Add first time
        basic_toolset.add(duplicate_tool)
        original_length = len(basic_toolset.tools)

        # Add same tool again
        basic_toolset.add(duplicate_tool)
        assert len(basic_toolset.tools) == original_length  # Should not add duplicate
        assert basic_toolset.get(duplicate_tool.__name__)() == "duplicate"

    def test_duplicate_tool_different_reference(self, basic_toolset):
        """Test adding tools with same name but different references"""

        def tool():
            return "first"

        # Add first version
        basic_toolset.add(tool)
        original_length = len(basic_toolset.tools)

        # Create and add second version with same name
        def tool():  # noqa: F811
            return "second"

        basic_toolset.add(tool)
        assert len(basic_toolset.tools) == original_length  # Length should not change
        assert basic_toolset.get("tool")() == "first"  # Should use latest version

    def test_duplicate_tool_in_nested_structure(self, basic_toolset):
        """Test adding duplicate tools in nested structure"""

        def nested_tool():
            return "nested"

        # Add tool first time
        basic_toolset.add(nested_tool)
        original_length = len(basic_toolset.tools)

        # Add duplicates in nested structure
        nested_structure = {"level1": [nested_tool], "level2": {"inner1": nested_tool, "inner2": [nested_tool]}}

        basic_toolset.add(nested_structure)
        assert len(basic_toolset.tools) == original_length  # Length should not change
        assert basic_toolset.get(nested_tool.__name__)() == "nested"

    def test_duplicate_tool_mixed_types(self, basic_toolset):
        """Test adding duplicate tools using different container types"""

        def mixed_tool():
            return "mixed"

        # Add as direct tool first
        basic_toolset.add(mixed_tool)
        original_length = len(basic_toolset.tools)

        # Try adding in list
        basic_toolset.add([mixed_tool])
        assert len(basic_toolset.tools) == original_length

        # Try adding in dict
        basic_toolset.add({"key": mixed_tool})
        assert len(basic_toolset.tools) == original_length

        # Try adding in nested structure
        basic_toolset.add({"level1": {"inner": [mixed_tool]}})
        assert len(basic_toolset.tools) == original_length

        # Verify final state
        assert basic_toolset.get(mixed_tool.__name__)() == "mixed"
