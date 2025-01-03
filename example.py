import nest_asyncio
import requests
from dotenv import load_dotenv
from pydantic_ai import RunContext

from agentgenius.main import AgentGENius
from agentgenius.tools import ToolSet

nest_asyncio.apply()  # just to use with ipykernel
load_dotenv()


def ask_user_tool(ctx: RunContext[str], question: str) -> str:
    """Ask the user a question"""
    return input(question + " ")


def get_current_datetime(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current datetime as a string in the specified python format."""

    from datetime import datetime

    return datetime.now().strftime(format_str)


tools = ToolSet(ask_user_tool)
agent = AgentGENius(name="assistant", model="openai:gpt-4o-mini", toolset=tools)
agent.agent_store.load_agents()

agent.toolset.add(get_current_datetime)


@agent.tool_plain
def get_my_public_ip() -> str:
    """Get the public IP address of the current machine using an external service."""
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        if response.status_code == 200:
            return response.json().get("ip", "")
        else:
            return "Unable to fetch IP"
    except Exception as e:
        return f"Error: {e}"


@agent.tool
def get_location_by_ip(ctx: RunContext[str], ip_address: str) -> dict:
    """Get the geographical location of an IP address using an external API."""
    try:
        response = requests.get(f"https://ipinfo.io/{ip_address}/json", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Unable to fetch location"}
    except Exception as e:
        return f"Error: {e}"


# print(agent.toolset)


def main():
    message_history = []
    print(f"Agent: {agent.run_sync("Hello").data}")
    while True:
        user_input = input("You: ")

        if user_input.lower() == "bye":
            print("Agent: Goodbye!")
            break
        response = agent.run_sync(user_input, message_history=message_history)
        message_history += response.new_messages()
        if len(message_history) > 20:
            message_history = message_history[-20:]
        print(f"Agent: {response.data}")


if __name__ == "__main__":
    main()
