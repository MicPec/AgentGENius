from types import NoneType
from typing import Union

from pydantic_ai import RunContext

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.history import History, TaskHistory
from agentgenius.tasks import Task, TaskDef
from agentgenius.tools import ToolDef

TaskDefList = list[TaskDef]
SimpleResponse = str


class QuestionAnalyzer:
    def __init__(self, model: str):
        self.agent_def = AgentDef(
            model=model,
            name="task analyzer",
            system_prompt="""You are an expert at breaking down complex tasks into smaller, manageable pieces.
Think step by step, what are the steps to solve this task and what information are needed to do it?
Also take into account the conversation history. User can ask for information about the previous task.
Focus on creating clear, detailed, effective, and actionable subtasks that can be executed independently by the AI agent. Optimal is 2-3 subtasks.
Keep in mind that the results of the previous subtasks are available for use in the current task, so do not duplicate tools. Task are sorted by priority.
In the field 'query', put the command for an AI agent, not question.
Be proactive with the task analysis, if you lack of any information create a new task.
For easy queries that do not require any additional information (like translation, welcome message, previous queries, etc.), you can return None


Examples:
Query: What time is it?
Expected output: [TaskDef(name="time_info", agent_def=AgentDef(...), query="get the current time")]

Query: What movies are playing today in my local cinema?
Expected output:
[TaskDef(name="find_location", agent_def=AgentDef(...), query="Identify the user's location")
TaskDef(name="search_web", agent_def=AgentDef(...), query="find cinema in user's location and schedule")]

Query: Search for file in my home directory
Expected output:
[TaskDef(name="os_info", agent_def=AgentDef(...), query="identify user's operating system")
TaskDef(name="user_info", agent_def=AgentDef(...), query="get user's name and home directory")
TaskDef(name="search_file", agent_def=AgentDef(...), query="use user's operating system to search for the file")]

Query: Hello! How are you?
Expected output: None
""",
            params=AgentParams(result_type=Union[TaskDefList, NoneType], deps_type=History),
        )

        self.task = Task(
            task_def=TaskDef(
                name="task_analysis",
                agent_def=self.agent_def,
                query="Analyze query and generate a list of subtasks. Query",
            )
        )

        @self.task._agent.system_prompt
        def get_history(ctx: RunContext[History]) -> str:
            """Prepare query by adding task history to the query."""
            if ctx.deps:
                history = [
                    {"user_query": item.user_query, "final_result": item.final_result}
                    for item in ctx.deps.items
                    if item.final_result is not None
                ]
            else:
                history = []
            result = f"Conversation history: {history}"
            # print(result)
            return result

    async def analyze(self, *, query: str, deps: History) -> Union[SimpleResponse, TaskDefList]:
        result = await self.task.run(query, deps=deps)
        if isinstance(result.data, str):
            return result.data
        return sorted(result.data, key=lambda x: x.priority)

    def analyze_sync(self, *, query: str, deps: History) -> Union[SimpleResponse, TaskDefList]:
        result = self.task.run_sync(query, deps=deps)
        if isinstance(result.data, NoneType):
            return None
        return sorted(result.data, key=lambda x: x.priority)


class TaskRunner:
    def __init__(self, model: str, task_def: TaskDef, toolset: list[ToolDef]):
        self.agent_def = AgentDef(
            model=model,
            name="Task solver",
            system_prompt="""You are an expert in task solving. You always try to solve the task using available tools and history.
Return clear, concise and detailed answer based on the information you have provided.""",
        )

        if task_def.agent_def is None:
            task_def.agent_def = self.agent_def
        task_def.agent_def.params = AgentParams(deps_type=TaskHistory)
        self.task = Task(task_def=task_def, agent_def=task_def.agent_def, toolset=toolset)

        @self.task._agent.system_prompt
        def get_history(ctx: RunContext[TaskHistory]) -> str:
            """Prepare query by adding task history to the query."""
            return f"Task History: {ctx.deps}"

    async def run(self, *, deps: TaskHistory):
        return await self.task.run(deps=deps)

    def run_sync(self, *, deps: TaskHistory):
        return self.task.run_sync(deps=deps)
