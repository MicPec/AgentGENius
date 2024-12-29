from .agents import BaseAgent, AgentSchema
from .tools import ToolSet, ToolSchema
from .config import config


class AgentGENius(BaseAgent):
    def __init__(
        self,
        name: str,
        model: str | None = config.default_agent_model,
        system_prompt: str | None = config.default_agent_prompt,
        toolset: ToolSet | None = None,
    ):
        super().__init__(name, model, system_prompt, toolset)

    def load_agent(self, json_str: str):
        data = AgentSchema.model_validate_json(json_str).model_dump()
        if data.get("toolset"):
            data["toolset"] = ToolSet.from_dict(data["toolset"])
        return BaseAgent(**data)

    def save_agent(self, agent: BaseAgent) -> str:
        config.agents_path.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member

        result = agent.to_json()
        with open(config.agents_path / f"{agent.name}.json", encoding="utf-8", mode="w") as f:
            f.write(result)
        return result
