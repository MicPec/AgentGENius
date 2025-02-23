import json

import pytest
from pydantic import ValidationError

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.builtin_tools import get_datetime, get_user_name
from agentgenius.tasks import Task, TaskDef, ToolSet


@pytest.fixture
def sample_agent_def():
    return AgentDef(
        model="test",
        name="TestAgent",
        system_prompt="You are a test assistant.",
        params=AgentParams(
            result_type=str,
            retries=3,
        ),
    )


@pytest.fixture
def sample_toolset():
    return ToolSet([get_datetime])


@pytest.fixture
def basic_task_def(sample_agent_def, sample_toolset):
    return TaskDef(
        name="TestTask",
        query="What time is it?",
        priority=1,
        agent_def=sample_agent_def,
        toolset=sample_toolset,
    )


class TestTaskDef:
    def test_create_basic_task_def(self, basic_task_def):
        """Test creating basic TaskDef with all fields"""
        assert basic_task_def.name == "TestTask"
        assert basic_task_def.query == "What time is it?"
        assert basic_task_def.priority == 1
        assert isinstance(basic_task_def.agent_def, AgentDef)
        assert isinstance(basic_task_def.toolset, ToolSet)

    def test_create_minimal_task_def(self):
        """Test creating TaskDef with only required fields"""
        task_def = TaskDef(
            name="MinimalTask",
            query="Test question",
            priority=1,
        )
        assert task_def.name == "MinimalTask"
        assert task_def.query == "Test question"
        assert task_def.priority == 1
        assert task_def.agent_def is None
        assert isinstance(task_def.toolset, ToolSet)

    def test_invalid_priority(self):
        """Test that negative priority raises ValidationError"""
        with pytest.raises(ValidationError):
            TaskDef(
                name="TestTask",
                query="Test question",
                priority=-1,
            )

    def test_serialization(self, basic_task_def):
        """Test JSON serialization and deserialization"""
        json_str = basic_task_def.model_dump_json()
        data = json.loads(json_str)

        # Check serialized data
        assert data["name"] == "TestTask"
        assert data["query"] == "What time is it?"
        assert data["priority"] == 1
        assert "agent_def" in data and data["agent_def"] is not None
        assert "toolset" in data and data["toolset"] is not None

        # Test deserialization
        task_def2 = TaskDef.model_validate_json(json_str)
        assert task_def2.name == basic_task_def.name
        assert task_def2.query == basic_task_def.query
        assert task_def2.priority == basic_task_def.priority
        assert isinstance(task_def2.agent_def, AgentDef)
        assert isinstance(task_def2.toolset, ToolSet)


class TestTask:
    def test_create_task(self, basic_task_def):
        """Test creating Task from TaskDef"""
        task = Task(task_def=basic_task_def)
        assert task.task_def == basic_task_def
        assert isinstance(task.agent_def, AgentDef)
        assert isinstance(task.toolset, ToolSet)

    def test_create_task_with_overrides(self, basic_task_def, sample_agent_def, sample_toolset):
        """Test creating Task with agent_def and toolset overrides"""
        new_agent = AgentDef(
            model="test",
            name="OverrideAgent",
            system_prompt="Override prompt",
            params=AgentParams(result_type=int),
        )
        new_toolset = ToolSet()

        task = Task(
            task_def=basic_task_def,
            agent_def=new_agent,
            toolset=new_toolset,
        )
        assert task.agent_def == new_agent
        assert task.toolset == sample_toolset | new_toolset

    def test_serialization(self, basic_task_def):
        """Test JSON serialization and deserialization"""
        task = Task(task_def=basic_task_def)
        json_str = task.model_dump_json()
        data = json.loads(json_str)

        # Check serialized data
        assert "task_def" in data
        assert "agent_def" in data
        assert "toolset" in data

        # Test deserialization
        task2 = Task.model_validate_json(json_str)
        assert task2.task_def == task.task_def
        assert isinstance(task2.agent_def, AgentDef)
        assert isinstance(task2.toolset, ToolSet)

    def test_toolset_merging_edge_cases(self, basic_task_def):
        """Test various edge cases for toolset merging"""
        from agentgenius.builtin_tools import get_user_name  # Adding a different tool for testing

        # Create a different toolset for testing
        different_toolset = ToolSet([get_user_name])

        # Case 1: Merging different toolsets
        task = Task(task_def=basic_task_def, toolset=different_toolset)
        assert task.toolset is not None
        assert len(task.toolset) == len(basic_task_def.toolset) + len(different_toolset)

        # Case 2: task_def has None toolset
        basic_task_def.toolset = None
        task = Task(task_def=basic_task_def, toolset=different_toolset)
        assert task.toolset is not None
        assert len(task.toolset) == len(different_toolset)

        # Case 3: Constructor toolset is None
        basic_task_def.toolset = different_toolset
        task = Task(task_def=basic_task_def, toolset=None)
        assert task.toolset is not None
        assert len(task.toolset) == len(different_toolset)

        # Case 4: Both toolsets are None
        basic_task_def.toolset = None
        task = Task(task_def=basic_task_def, toolset=None)
        assert task.toolset is not None
        assert len(task.toolset) == 0

    def test_toolset_merging_with_dict_input(self, sample_agent_def):
        """Test toolset merging when task_def is provided as a dict"""
        from agentgenius.builtin_tools import get_datetime, get_user_name

        # Create different toolsets for testing
        toolset1 = ToolSet([get_datetime])
        toolset2 = ToolSet([get_user_name])

        # Create task_def as a dict
        task_def_dict = {
            "name": "TestTask",
            "query": "Test query",
            "priority": 1,
            "agent_def": sample_agent_def,
            "toolset": toolset1
        }

        # Case 1: Dict task_def with different toolset
        task = Task(task_def=task_def_dict, toolset=toolset2)
        assert task.toolset is not None
        assert len(task.toolset) == len(toolset1) + len(toolset2)

        # Case 2: Dict task_def with None toolset
        task_def_dict["toolset"] = None
        task = Task(task_def=task_def_dict, toolset=toolset2)
        assert task.toolset is not None
        assert len(task.toolset) == len(toolset2)

        # Case 3: Dict task_def without toolset key
        del task_def_dict["toolset"]
        task = Task(task_def=task_def_dict, toolset=toolset2)
        assert task.toolset is not None
        assert len(task.toolset) == len(toolset2)
