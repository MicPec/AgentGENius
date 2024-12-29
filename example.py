from datetime import datetime

import nest_asyncio
from dotenv import load_dotenv
from pydantic_ai import RunContext, Tool

from agentgenius.core import AgentGENius
from agentgenius.tools import ToolSet

nest_asyncio.apply()
load_dotenv()


def ask_user_tool(ctx: RunContext[str], question: str) -> str:
    """Ask the user a question"""
    return input(question + " ")


tools = ToolSet(ask_user_tool)


agent = AgentGENius(name="assistant", model="openai:gpt-4o-mini", toolset=tools)


@agent.tool_plain
def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    message_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Agent: Goodbye!")
            break
        response = agent.run_sync(user_input, message_history=message_history)
        message_history += response.new_messages()
        if len(message_history) > 20:
            message_history = message_history[-20:]
        print(f"Agent ({len(message_history)}):{response.data}")


if __name__ == "__main__":
    main()
