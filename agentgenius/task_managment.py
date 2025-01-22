from agentgenius.agents import AgentDef, AgentParams
from agentgenius.history import TaskHistory
from agentgenius.tasks import Task, TaskDef
from agentgenius.tools import ToolDef


class QuestionAnalyzer:
    def __init__(self, model: str):
        self.agent_def = AgentDef(
            model=model,
            name="task analyzer",
            system_prompt="""You are an expert at breaking down complex tasks into smaller, manageable pieces.
Think step by step, what are the steps to solve this task and what information are needed to do it?
Focus on creating clear, detailed, effective, and actionable subtasks that can be executed independently by the AI agent.
Keep in mind that the results of the previous subtasks are available for use in the current task, so do not duplicate tools. Task are sorted by priority.
In the field 'query', put the command for an AI agent, not question
Example:
Question: What movies are playing today in my local cinema?
Steps:
1. Find the user location and current time.
2. Get the cinema location in the user's area.
3. Get the movie schedule for the local cinema.
Question: Search for file in my home directory
Steps:
1. Identify the user's operating system.
2. Get the user's name and home directory.
3. Use the user's operating system to search for the file.""",
            params=AgentParams(result_type=list[TaskDef], deps_type=TaskHistory),
        )

    async def analyze(self, query: str, deps: TaskHistory) -> list[TaskDef]:
        result = await Task(task_def=TaskDef(name="task_analysis", agent_def=self.agent_def, query=query)).run(
            deps=deps
        )
        return sorted(result.data, key=lambda x: x.priority)

    def analyze_sync(self, query: str, deps: TaskHistory) -> list[TaskDef]:
        result = Task(task_def=TaskDef(name="task_analysis", agent_def=self.agent_def, query=query)).run_sync(deps=deps)
        return sorted(result.data, key=lambda x: x.priority)


class TaskRunner:
    def __init__(self, model: str, task_def: TaskDef, toolset: list[ToolDef]):
        self.agent_def = AgentDef(
            model=model,
            name="Task solver",
            system_prompt="You are an expert in task solving. You always try to solve the task using available informations and tools.",
        )
        self.task_def = task_def
        if self.task_def.agent_def is None:
            self.task_def.agent_def = self.agent_def
        self.task_def.agent_def.params = AgentParams(deps_type=TaskHistory)
        self.task_def.toolset = toolset

    async def run(self, task_history: TaskHistory):
        return await Task(task_def=self.task_def).run(deps=task_history)

    def run_sync(self, task_history: TaskHistory):
        return Task(task_def=self.task_def).run_sync(deps=task_history)
