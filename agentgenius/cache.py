import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentgenius.tasks import TaskDef
from agentgenius.tools import ToolSet


class CacheEntry(BaseModel):
    """Single cache entry with result and timestamp"""

    result: Any
    timestamp: datetime


class ToolCallCache(BaseModel):
    """Cache for tool call results"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cache: Dict[str, CacheEntry] = Field(default_factory=dict)
    max_size: int = Field(default=100)
    ttl_minutes: int = Field(default=60)

    def _make_key(self, tool_name: str, args: tuple, kwargs: dict) -> str:
        """Create a cache key from tool name and arguments"""
        key_parts = [tool_name, str(args), str(sorted(kwargs.items()))]
        return hashlib.md5(str(key_parts).encode()).hexdigest()

    def get(self, tool_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached result if it exists and is not expired"""
        key = self._make_key(tool_name, args, kwargs)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry.timestamp < timedelta(minutes=self.ttl_minutes):
                return entry.result
            del self.cache[key]
        return None

    def set(self, tool_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache a tool call result"""
        key = self._make_key(tool_name, args, kwargs)
        self.cache[key] = CacheEntry(result=result, timestamp=datetime.now())

        # Remove oldest entries if cache is too large
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)  # pylint: disable=no-member
            del self.cache[oldest_key]


class TaskResult(BaseModel):
    """Result from a single task execution"""

    task_def: TaskDef
    success: bool
    result: Any
    error: Optional[str] = None


class CachedToolSet(ToolSet):
    """ToolSet wrapper that caches tool results"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cache: ToolCallCache = Field(default_factory=ToolCallCache)

    def _wrap_tool(self, tool: Callable) -> Callable:
        """Wrap a tool function with caching"""

        @wraps(tool)
        def wrapped(*args, **kwargs):
            # Try to get from cache
            cached_result = self.cache.get(tool.__name__, args, kwargs)  # pylint: disable=no-member
            if cached_result is not None:
                return cached_result

            # Call the tool and cache result
            result = tool(*args, **kwargs)
            self.cache.set(tool.__name__, args, kwargs, result)  # pylint: disable=no-member
            return result

        return wrapped

    def add(self, tool: Callable):
        """Add a tool with caching wrapper"""
        wrapped_tool = self._wrap_tool(tool)
        super().add(wrapped_tool)
