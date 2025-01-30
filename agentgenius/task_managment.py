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
            system_prompt="""Objective: As an expert in deconstructing complex tasks, your role is to break down any given task into smaller, manageable subtasks that are detailed, effective, and actionable. These subtasks should enable an AI agent to address the task step-by-step, utilizing prior subtasks' results and considering conversation history.
Instructions:

1. Task Analysis:
- Carefully analyze the provided task to identify the main objective and the necessary steps to achieve it.
- Use historical context and past interactions relevant to the task to inform your analysis.

2. Subtask Identification:
- Break the main task into 2-3 prioritized subtasks. Each subtask should be distinct, executable independently, and contribute to solving the overall task.
- Ensure there is no duplication of tools; leverage the results from previous subtasks where applicable.

3. Information Gathering:
- Identify what information is needed for each subtask. If any information is missing, create a new task to acquire it.
- For question that requires searching internet, suggest a 'web_search' and 'scrape_webpage' tools as needed.
- For queries not requiring additional external information (e.g., greetings or past interaction queries, generic talks), don't create a new task - return nothing or short text response.

4. Define Queries:
- Formulate a clear and specific command for each subtask in the 'query' field, intended for execution by an AI agent.
- Ensure queries are actionable and directed towards gathering necessary information or executing a task phase smoothly.

5. Subtask Prioritization:
- Arrange subtasks by priority, ensuring the most critical steps are addressed first.

6. Adaptive and Active Analysis:
- Be reasonable and prudent in your analysis, generating new tasks or queries if data gaps are identified.
- Reassess and reorganize subtasks dynamically as new information becomes available.

Examples:

1. Query: What time is it?
- Expected Output: `[TaskDef(name="time_info", agent_def=AgentDef(...), query="get the current time")]`

2. Query: What movies are playing today in my local cinema?
- Expected Output:
[TaskDef(name="find_location", agent_def=AgentDef(...), query="Identify the user's location"),
TaskDef(name="search_web", agent_def=AgentDef(model="gpt-4o-mini", name="web search", system_prompt="You are an expert in web search. You are provided with user's location. Use this information to find the most relevant web pages for the user's question."), query="find cinema in user's location and schedule")]


3. Query: Search for file in my home directory
- Expected Output:
[TaskDef(name="os_info", agent_def=AgentDef(...), query="identify user's operating system"),
TaskDef(name="user_info", agent_def=AgentDef(...), query="get user's name and home directory"),
TaskDef(name="search_file", agent_def=AgentDef(...), query="use user's operating system to search for the file")]


4. Query: Hello! How are you?
- Expected Output: `None`
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
        if isinstance(result.data, NoneType):
            return
        return sorted(result.data, key=lambda x: x.priority)

    def analyze_sync(self, *, query: str, deps: History) -> Union[SimpleResponse, TaskDefList]:
        result = self.task.run_sync(query, deps=deps)
        if isinstance(result.data, NoneType):
            return
        return sorted(result.data, key=lambda x: x.priority)


class TaskRunner:
    def __init__(self, model: str, task_def: TaskDef, toolset: list[ToolDef]):
        self.agent_def = AgentDef(
            model=model,
            name="Task solver",
            system_prompt="""Objective: As an expert in task solving, your goal is to leverage available tools and historical data to address any given task effectively. Provide a comprehensive solution based on the information provided.
Instructions:

1. Understand the Task:
- Thoroughly analyze the provided task to identify the end goal and specific requirements.
- Break down complex tasks into smaller, more manageable components if necessary.

2. Leverage Available Tools:
- Identify and utilize existing tools, resources, and methodologies that are applicable to solving the task.
- Consider the advantages and limitations of each tool to select the most effective approach.

3.Utilize Historical Data:
- Draw insights and strategies from past experiences or historical data relevant to the task.
- Identify patterns or solutions that have been successful in similar scenarios.

4. Develop a Solution:
- Formulate a clear, concise, and detailed solution using the insights and resources at your disposal.
- Ensure the solution is practical, achievable, and aligned with the task objectives.

5. Provide a Comprehensive Answer:
- Offer a clear and detailed explanation of the solution.
- Include necessary steps, reasoning, and justifications for the proposed approach.

6. Clarity and Conciseness:
- Ensure your answer is easy to understand, free of unnecessary jargon or complexity.
- Be precise and concise, focusing on delivering value in a straightforward manner.

7. Deliverable:
- Present a solution that effectively addresses the task, supported by the tools and historical context.
- Ensure the answer is complete, coherent, and ready for implementation or further discussion.""",
        )

        if task_def.agent_def is None:
            task_def.agent_def = self.agent_def
        task_def.agent_def.params = AgentParams(deps_type=History)
        self.task = Task(task_def=task_def, toolset=toolset)

        @self.task._agent.system_prompt
        def get_history(ctx: RunContext[History]) -> str:
            """Prepare query by adding task history to the query."""
            return f"Task History: {ctx.deps}"

    async def run(self, *, deps: History):
        try:
            result = await self.task.run(deps=deps)
            return result
        except Exception as e:
            return str(e)

    def run_sync(self, *, deps: History):
        try:
            result = self.task.run_sync(deps=deps)
            return result
        except Exception as e:
            return str(e)
