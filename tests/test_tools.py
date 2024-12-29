import pytest
from agentgenius.tools import ToolSet


@pytest.fixture
def sample_function():
    def func(x: str) -> str:
        """Sample function docstring"""
        return f"Hello {x}"

    return func


@pytest.fixture
def sample_async_function():
    async def async_func(x: str) -> str:
        """Sample async function docstring"""
        return f"Hello {x}"

    return async_func


class TestToolSet:
    def test_toolset_creation_empty(self):
        """Test creating an empty ToolSet"""
        toolset = ToolSet()
        assert len(toolset.tools) == 0

    def test_toolset_creation_with_single_tool(self, sample_function):
        """Test creating a ToolSet with a single tool"""
        toolset = ToolSet(sample_function)
        assert len(toolset.tools) == 1
        assert toolset.tools[0] == sample_function

    def test_toolset_creation_with_multiple_tools(self, sample_function, sample_async_function):
        """Test creating a ToolSet with multiple tools"""
        toolset = ToolSet([sample_function, sample_async_function])
        assert len(toolset.tools) == 2
        assert sample_function in toolset.tools
        assert sample_async_function in toolset.tools

    def test_toolset_iteration(self, sample_function, sample_async_function):
        """Test that ToolSet is iterable"""
        toolset = ToolSet([sample_function, sample_async_function])
        tools = list(toolset)
        assert len(tools) == 2
        assert sample_function in tools
        assert sample_async_function in tools

    def test_toolset_add_remove(self, sample_function):
        """Test adding and removing tools from ToolSet"""
        toolset = ToolSet()

        # Test add
        toolset.add(sample_function)
        assert len(toolset.tools) == 1
        assert sample_function in toolset.tools

        # Test remove
        toolset.remove(sample_function.__name__)
        assert len(toolset.tools) == 0

    def test_toolset_get(self, sample_function):
        """Test getting a tool by name"""
        toolset = ToolSet(sample_function)
        retrieved_tool = toolset.get(sample_function.__name__)
        assert retrieved_tool == sample_function

    def test_toolset_get_nonexistent(self):
        """Test getting a nonexistent tool returns None"""
        toolset = ToolSet()
        assert toolset.get("nonexistent") is None

    def test_toolset_contains(self, sample_function, sample_async_function):
        """Test checking if a tool is in ToolSet"""
        toolset = ToolSet(sample_function)
        assert sample_function in toolset
        assert sample_function.__name__ in toolset
        assert sample_async_function not in toolset
        assert "nonexistent" not in toolset
        with pytest.raises(ValueError, match="value must be a type of string or callable"):
            0 in toolset

    def test_toolset_serialization(self, sample_function):
        """Test JSON serialization and deserialization of ToolSet"""
        toolset = ToolSet(sample_function)

        # Test to_dict and to_json
        json_str = toolset.to_json()
        dict_data = toolset.to_dict()

        assert isinstance(json_str, str)
        assert isinstance(dict_data, list)
        assert sample_function.__name__ in dict_data

        # Test from_json and from_dict
        namespace = {"func": sample_function}

        toolset2 = ToolSet.from_json(json_str, namespace=namespace)
        assert len(toolset2.tools) == 1
        assert toolset2.tools[0] == sample_function

    def test_toolset_repr(self, sample_function):
        """Test string representation of ToolSet"""
        toolset = ToolSet(sample_function)
        expected = f"ToolSet(['{sample_function.__name__}'])"
        assert repr(toolset) == expected

    def test_duplicate_tool_name(self, sample_function):
        """Test that adding a tool with a duplicate name raises ValueError"""
        with pytest.raises(ValueError, match=f"Tool {sample_function.__name__} already exists in ToolSet"):
            toolset = ToolSet([sample_function, sample_function])  # Try to add two tools with same name

        toolset = ToolSet(sample_function)
        with pytest.raises(ValueError, match=f"Tool {sample_function.__name__} already exists in ToolSet"):
            toolset.add(sample_function)  # Try to add second tool with same name

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, sample_async_function):
        """Test that async tools can be added and executed"""
        toolset = ToolSet(sample_async_function)
        tool = toolset.tools[0]
        result = await tool("World")
        assert result == "Hello World"
