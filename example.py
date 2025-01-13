from typing import List

import logfire
from dotenv import load_dotenv
from pydantic import TypeAdapter
from pydantic_ai import ModelRetry, RunContext, Tool
from rich import print

from agentgenius import AgentDef, AgentParams, Task, TaskDef
from agentgenius.builtin_tools import get_datetime, get_location_by_ip, get_user_ip
from agentgenius.tasks import TaskList
from agentgenius.tools import ToolSet

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")


planner = Task(
    task_def=TaskDef(name="planner", question="make a short plan how to archive this task", priority=1),
    agent_def=AgentDef(
        # model="openai:gpt-4o",
        model="openai:gpt-4o",
        name="planner",
        system_prompt="""You are a planner. your goal is to make a step by step plan for other agents. 
        Do not answer the user questions and NEVER call available tools. Just make a very short plan how to do this. 
        AlWAYS MAKE SURE TO ADD APPROPRIATE AGENTS AND TOOLS TO THE PLAN. 
        You can get the list of available tools by calling 'get_available_tools'.
        Efficiently is a priority, so don't waste time on things that are not necessary.
        LESS STEPS IS BETTER (up to 3 steps), so make it as short as possible.
        Tell an agent to use the tools if available. ALWAYS USE THE USER'S LANGUAGE""",
        params=AgentParams(
            result_type=TaskDef,
            # deps_type=TaskDef,
            retries=3,
        ),
    ),
    # toolset=ToolSet([get_datetime, get_user_ip, get_location_by_ip]),
)


@planner._agent.system_prompt
def get_available_tools():
    """Return a list of available tool names. DO NOT CALL THESE TOOLS.
    Just pass them to the other agents and let them to use them."""
    toolset = ToolSet([get_datetime, get_user_ip, get_location_by_ip])
    planner.register_toolset(toolset)
    return f"Available tools: {', '.join(toolset.all())}"


@planner._agent.result_validator
def validate_result(ctx: RunContext, result: TaskDef):
    if result.agent_def is None:
        raise ModelRetry(f"Agent is not defined in {result}")
    if result.toolset == []:
        raise ModelRetry(f"Toolset is not defined in {result}")
    return result


# result = planner.run_sync("what time is it at my location?")
# result = planner.run_sync("how to get my location by IP?")
result = planner.run_sync("Jaka jest moja lokacja i godzina? Kim jesteś?")

print(result.data)
task = Task(task_def=result.data)
result = task.run_sync()
print(result.data)

# for task in result.data:
#     ctx = []
#     ctx.append(task.run_sync(deps=ctx).data)
#     print(ctx)
