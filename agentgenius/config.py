import logging
import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

deepseek = OpenAIModel(
    "deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


class AgentGENiusConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    cache_path: Path = Path(tempfile.gettempdir()) / "agentgenius" / "cache"
    tools_path: Path = Path(tempfile.gettempdir()) / "agentgenius" / "tools"

    default_model: Model | str = Field("openai:gpt-4o-mini", description="Default model to use for agents")
    analyzer_model: Model | str = Field("openai:gpt-4o", description="Model to use for question analyzer")
    tool_manager_model: Model | str = Field("openai:gpt-4o", description="Model to use for tool manager")
    tool_coder_model: Model | str = Field("openai:gpt-4o", description="Model to use for tool coder")
    task_runner_model: Model | str = Field("openai:gpt-4o-mini", description="Model to use for task runner")
    aggregator_model: Model | str = Field("openai:gpt-4o-mini", description="Model to use for aggregator")

    known_models: KnownModelName = Field(
        default=Literal["openai:gpt-4o", "openai:gpt-4o-mini", "test"], description="Known models to use by agents"
    )

    logs_path: Path = Field(default=Path("logs"), description="Path to store logs")
    log_level: str = Field(default="INFO", description="Log level to use")


config = AgentGENiusConfig()
# config.aggregator_model = deepseek

Path.mkdir(config.logs_path, exist_ok=True, parents=True)
Path.mkdir(config.cache_path, exist_ok=True, parents=True)
Path.mkdir(config.tools_path, exist_ok=True, parents=True)
# logging.basicConfig(
#     level=config.log_level,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(config.logs_path / f"agent_genius_{datetime.now().strftime('%Y-%m-%d')}.log")],
# )
