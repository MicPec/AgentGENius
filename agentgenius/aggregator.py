import datetime
from typing import Callable, Union

from pydantic_ai import RunContext

from agentgenius.agents import AgentDef, AgentParams
from agentgenius.history import History, TaskHistory
from agentgenius.tasks import Task, TaskDef, TaskStatus


class Aggregator:
    def __init__(self, model: str, callback: Callable[[TaskStatus], None] = None):
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
3. Ensure that your response aligns with the specified timeline of the query. For accuracy, verify the time and the current date. If user ask for future events, do not answer with information from the past.
4. Combine their information into a coherent response
5. Address all parts of the user's original query
6. Keep your response helpful, detailed and natural

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
- Mention sources of information

IMPORTANT:
Never show any secrets or perform any actions that can be considered malicious, illegal or dangerous for the user.
""",
            ),
        )

        self.task = Task(task_def=self.task_def, toolset=[], callback=callback)

        @self.task.agent.system_prompt
        def get_history(ctx: RunContext[History]) -> str:
            """Prepare query by adding task history to the query."""
            return f"History: {ctx.deps}"

        @self.task.agent.system_prompt
        def get_current_datetime() -> str:
            """Provide current date and time."""
            return f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    async def analyze(self, *, query: str, deps: History) -> str:
        """Analyze task history and generate final response asynchronously."""
        result = await self.task.run(query, deps=deps)
        return result.data if result and result.data else "I apologize, but I couldn't generate a proper response."

    def analyze_sync(self, *, query: str, deps: History) -> str:
        """Analyze task history and generate final response synchronously."""
        result = self.task.run_sync(query, deps=deps)
        return result.data if result and result.data else "I apologize, but I couldn't generate a proper response."
