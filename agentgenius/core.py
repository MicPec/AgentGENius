from .agents import AgentSchema, BaseAgent
from .config import config
from .tools import ToolSchema, ToolSet


class AgentGENius(BaseAgent):
    def __init__(
        self,
        name: str,
        model: str | None = config.default_agent_model,
        system_prompt: str | None = config.default_agent_prompt,
        toolset: ToolSet | None = None,
    ):
        super().__init__(name, model, system_prompt, toolset)


# TODO: Add AgentGENius methods


class TaskGENius(BaseAgent):
    def __init__(
        self,
        name: str,
        model: str | None = config.default_agent_model,
        system_prompt: str | None = config.default_agent_prompt,
        toolset: ToolSet | None = None,
    ):
        super().__init__(name, model, system_prompt, toolset)


# TODO: Add TaskGENius methods


class ToolGENius(BaseAgent):
    def __init__(
        self,
        name: str,
        model: str | None = config.default_agent_model,
        system_prompt: str | None = config.default_agent_prompt,
        toolset: ToolSet | None = None,
    ):
        super().__init__(name, model, system_prompt, toolset)


# TODO: Add ToolGENius methods
