from agentgenius.agents import AgentDef, AgentParams
from agentgenius.tasks import Task, TaskDef


class QuestionAnalyzer:
    def __init__(self, query: str):
        self.query = query
        self._agent = AgentDef(
            model="openai:gpt-4o",
            name="task analyzer",
            system_prompt="""You are an expert at breaking down complex tasks into smaller, manageable pieces.
Think step by step, what are the steps to solve this task and what information are needed to do it?
Focus on creating clear, detailed, effective, and actionable subtasks that can be executed independently.
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
            params=AgentParams(
                result_type=list[TaskDef],
            ),
        )

    async def analyze(self) -> list[TaskDef]:
        result = await Task(task_def=TaskDef(name="task_analysis", agent_def=self._agent, query=self.query)).run()
        return sorted(result.data, key=lambda x: x.priority)

    def analyze_sync(self) -> list[TaskDef]:
        result = Task(task_def=TaskDef(name="task_analysis", agent_def=self._agent, query=self.query)).run_sync()
        return sorted(result.data, key=lambda x: x.priority)
