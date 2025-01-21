from agentgenius.agents import AgentDef, AgentParams
from agentgenius.history import TaskHistory
from agentgenius.tasks import Task, TaskDef


class Aggregator:
    def __init__(self, history: TaskHistory):
        self.history = history
        self.task_def = TaskDef(
            name="aggregator",
            query="Analyze full history of tasks and results and generate a final answer. Answer in language of the user first query.",
            priority=10,
            agent_def=AgentDef(
                model="openai:gpt-4o",
                name="aggregator",
                system_prompt="You are an expert at synthesizing information and providing clear, direct answers.",
                params=AgentParams(
                    result_type=str,
                    deps_type=TaskHistory,
                ),
            ),
        )

        self.task = Task(task_def=self.task_def)

    async def analyze(self):
        return (await self.task.run(self.history)).data

    def analyze_sync(self):
        print(self.history)
        return self.task.run_sync(self.history).data
