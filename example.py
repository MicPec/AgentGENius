from os import name

import logfire
from dotenv import load_dotenv
from rich import print

from agentgenius import AgentDef, Task, TaskDef
from agentgenius.builtin_tools import get_datetime, get_user_ip_and_location
from agentgenius.tasks import TaskList
from agentgenius.tools import ToolSet

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")

planner = Task(
    task=TaskDef(name="planner", question="make a short plan how to archive this task", priority=1),
    agent_def=AgentDef(
        model="openai:gpt-4o",
        name="planner",
        system_prompt="""You are a planner. your goal is to make a step by step plan for other agents. 
        Do not answer the user questions. Just make a very short plan how to do this. 
        AlWAYS MAKE SURE TO ADD APPROPRIATE TOOLS TO THE PLAN. You can get the list of available tools by calling 'get_available_tools'.
        Efficiently is a priority, so don't waste time on things that are not necessary.
        LESS STEPS IS BETTER (up to 3 steps), so make it as short as possible.""",
        params={
            "result_type": TaskList,
            "retries": 3,
        },
    ),
    # toolset=ToolSet(["get_datetime", "get_user_ip_and_location", "get_installed_packages"]),
)


@planner.agent.system_prompt
def get_available_tools():
    """Return a list of available tools. Do not use these tools.
    Just let the other agents to use them."""
    tools = ToolSet([get_datetime, get_user_ip_and_location]).init(namespace=globals())
    return f"Available tools: {', '.join(tools)}"


result = planner.run_sync("what time is it?")
# result = planner.run_sync("how to get my location by IP?")
print(result.data)
plan = result.data[0].run_sync()
print(plan.data)
