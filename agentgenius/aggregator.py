from agentgenius.agents import AgentDef, AgentParams
from agentgenius.history import TaskHistory
from agentgenius.tasks import Task, TaskDef


class Aggregator:
    def __init__(self, model: str, history: TaskHistory):
        self.history = history
        self.task_def = TaskDef(
            name="aggregator",
            query="Analyze the task results and provide a final answer that addresses all parts of the user's query. Your answer should be clear, concise, and in a natural conversational style. Use the task results to provide accurate information.",
            priority=10,
            agent_def=AgentDef(
                model=model,
                name="aggregator",
                system_prompt="""You are an expert at synthesizing information and providing clear, direct answers.
Your task is to:
1. Always respond in the language from the user's query ('user_query')
2. Look at all the task results in the history
3. Combine their information into a coherent response
4. Address all parts of the user's original query
5. Format the response in a clear, natural way
6. If there are any uncertainties or missing information, acknowledge them
7. Keep the tone helpful and conversational""",
                params=AgentParams(
                    result_type=str,
                    deps_type=TaskHistory,
                ),
            ),
        )

        self.task = Task(task_def=self.task_def)

    async def analyze(self) -> str:
        """Analyze task history and generate final response asynchronously."""
        result = await self.task.run(self.history)
        return result.data if result and result.data else "I apologize, but I couldn't generate a proper response."

    def analyze_sync(self) -> str:
        """Analyze task history and generate final response synchronously."""
        result = self.task.run_sync(self.history)
        return result.data if result and result.data else "I apologize, but I couldn't generate a proper response."
