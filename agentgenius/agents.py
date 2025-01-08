from dataclasses import field
from typing import Any, Dict, Literal

import logfire
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
from pydantic_ai.models import KnownModelName

load_dotenv()


KnownModelName = Literal[
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
]

AgentParams = Dict[str, Any]


@dataclass
class AgentDef:
    """Definition of an agent.

    Args:
        model (KnownModelName): The model to use for the agent. https://ai.pydantic.dev/api/models/base/
        name (str): The name of the agent.
        system_prompt (str): The system prompt for the agent.
        params (AgentParams, optional): The parameters for the agent. Defaults to {}.
            Possible params:
            result_type: type[ResultData] = str,
            deps_type: type[AgentDeps] = NoneType,
            model_settings: ModelSettings | None = None,
            retries: int = 1,
            result_tool_name: str = 'final_result',
            result_tool_description: str | None = None,
            result_retries: int | None = None,
            defer_model_check: bool = False,
            end_strategy: EndStrategy = 'early',
    """

    model: KnownModelName
    name: str
    system_prompt: str
    params: AgentParams = field(default_factory=dict, repr=True)
