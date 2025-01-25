import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import KnownModelName as kmn

KNOWN_MODELS = Literal[
    "openai:gpt-4o", "openai:gpt-4o-mini", "ollama:granite3.1-dense:8b-instruct-q6_K", "ollama:qwen2.5:14b", "test"
]


if not KNOWN_MODELS:
    KNOWN_MODELS = kmn


class AgentGENiusConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    tools_path: Path = Field(default=Path("tools"), description="Path to store tools")
    agents_path: Path = Field(default=Path("agents"), description="Path to store agents")
    default_agent_model: str = Field(default="openai:gpt-4o", description="Default model to use for agents")
    default_agent_prompt: str = Field(
        default="You are a helpful AI assistant.", description="Default system prompt to use for agents"
    )
    logs_path: Path = Field(default=Path("logs"), description="Path to store logs")
    log_level: str = Field(default="INFO", description="Log level to use")


config = AgentGENiusConfig()
Path.mkdir(config.logs_path, exist_ok=True)
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.logs_path / f"agent_genius_{datetime.now().strftime('%Y-%m-%d')}.log")],
)
