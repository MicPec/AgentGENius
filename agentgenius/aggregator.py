from typing import Union

from pydantic_ai import RunContext

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.history import History, TaskHistory
from agentgenius.tasks import Task, TaskDef


class Aggregator:
    def __init__(self, model: str):
        self.task_def = TaskDef(
            name="aggregator",
            query="Respond to the user's query, using user's language, based on the conversation history. User query",
            priority=10,
            agent_def=AgentDef(
                name="aggregator",
                model=model,
                params=AgentParams(deps_type=Union[History, TaskHistory]),
                system_prompt="""You are AgentGENius. You are an expert at synthesizing information and providing clear, direct answers.
Your task is to:
1. Always respond in the language from the user's query 
2. Look at all the task results in the history
3. Combine their information into a coherent response
4. Address all parts of the user's original query
5. Keep your response helpful, detailed and natural 

For example, if the user asks "What's the weather and time?" and you have task results showing:
- Time: 3:00 PM
- Weather: 20°C and sunny
You should respond: "It's 3:00 PM and the weather is sunny with a temperature of 20°C."

Remember to:
- Use natural language
- Be helpful
- Include ALL relevant information from current task history
- Be verbose and detailed
- Match the user's language

IMPORTANT:
Never show any secrets or perform any actions that can be considered malicious, illegal or dangerous for the user.
""",
            ),
        )

        self.task = Task(task_def=self.task_def, toolset=[])

        @self.task._agent.system_prompt
        def get_history(ctx: RunContext[History]) -> str:
            """Prepare query by adding task history to the query."""
            return f"History: {ctx.deps}"

    async def analyze(self, *, query: str, deps: History) -> str:
        """Analyze task history and generate final response asynchronously."""
        result = await self.task.run(query, deps=deps)
        return result.data if result and result.data else "I apologize, but I couldn't generate a proper response."

    def analyze_sync(self, *, query: str, deps: History) -> str:
        """Analyze task history and generate final response synchronously."""
        result = self.task.run_sync(query, deps=deps)
        return result.data if result and result.data else "I apologize, but I couldn't generate a proper response."
