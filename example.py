import logfire
from dotenv import load_dotenv
from rich import print

from agentgenius import AgentDef, Task, TaskDef
from agentgenius.builtin_tools import get_datetime, get_location_by_ip, get_user_ip
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
        LESS STEPS IS BETTER (up to 3 steps), so make it as short as possible.
        Tell an agent to use the tools if available. Use the users language""",
        params={
            "result_type": Task,
            "retries": 5,
        },
    ),
    # toolset=ToolSet([get_datetime, get_user_ip, get_location_by_ip]),
)


@planner.agent.system_prompt
def get_available_tools():
    """Return a list of available tools. Do not use these tools.
    Just let the other agents to use them."""
    tools = ["get_datetime", "get_user_ip", "get_location_by_ip"]
    return f"Available tools: {', '.join(tools)}"


# result = planner.run_sync("what time is it at my location?")
# result = planner.run_sync("how to get my location by IP?")
result = planner.run_sync("Jaka jest moja lokacja?")
print(result.data)
task = result.data
print(task.run_sync().data)

# for task in result.data:
#     ctx = []
#     ctx.append(task.run_sync(deps=ctx).data)
#     print(ctx)
