import importlib
import inspect
import sys
from functools import cache, wraps
from pathlib import Path
from types import GenericAlias
from typing import Any, Dict

from pydantic import TypeAdapter

from agentgenius.config import config


class TypeAdapterMixin:
    @classmethod
    def model_dump_json(cls, instance: object, **kwargs):
        adapter = TypeAdapter(cls)
        return adapter.dump_json(instance, **kwargs)

    @classmethod
    def model_dump(cls, instance: object, **kwargs):
        adapter = TypeAdapter(cls)
        return adapter.dump_python(instance, **kwargs)


def custom_type_encoder(obj: Any) -> Any:
    if isinstance(obj, type):
        return obj.__name__  # Serialize type objects as their names
    if isinstance(obj, GenericAlias):
        return repr(obj)
    return obj.model_dump()  # Fallback to default Pydantic encoder


def search_frame(value: str, name: str = None) -> dict:
    frame = inspect.currentframe()
    while frame:
        if name:
            result = frame.f_globals.get(name)
            if result == value:
                return frame.f_globals
        result = frame.f_locals.get(value)
        if result is not None:
            return frame.f_locals
        result = frame.f_globals.get(value)
        if result is not None:
            return frame.f_globals
        frame = frame.f_back
    raise ValueError(f"'{value}' not found")


def save_history(filename: str = "task_history.json"):
    """Decorator factory that saves history after each task execution"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            result = await func(self, *args, **kwargs)

            # Save history to file
            if hasattr(self, "history"):
                history_path = Path("history") / filename
                history_path.parent.mkdir(exist_ok=True)
                history_path.write_text(self.history.model_dump_json(indent=2), encoding="utf-8")

            return result

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # Save history to file
            if hasattr(self, "history"):
                history_path = Path("history") / filename
                history_path.parent.mkdir(exist_ok=True)
                history_path.write_text(self.history.model_dump_json(indent=2), encoding="utf-8")

            return result

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


def load_generated_tools() -> Dict[str, Any]:
    """Load all generated tools from the temporary directory and add them to globals().

    Returns:
        Dict[str, Any]: Dictionary mapping tool names to their function objects
    """
    tools = {}
    temp_dir = config.tools_path

    if not temp_dir.exists():
        return tools

    for tool_file in temp_dir.glob("*.py"):
        try:
            # Create a unique module name
            module_name = f"generated_tool_{tool_file.stem}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, str(tool_file))
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get all functions from the module and add to globals
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and not attr_name.startswith("_"):
                    tools[attr_name] = attr
                    # Add to globals so search_frame can find it
                    globals()[attr_name] = attr

        except Exception as e:
            print(f"Error loading tool from {tool_file}: {e}")
            continue

    return tools


@cache
def load_builtin_tools() -> Dict[str, Any]:
    """Load all builtin tools from the builtin_tools module.

    Returns:
        Dict[str, Any]: Dictionary mapping tool names to their function objects
    """
    tools = {}

    try:
        from agentgenius import builtin_tools

        # Get all callable functions that don't start with _
        for attr_name in dir(builtin_tools):
            attr = getattr(builtin_tools, attr_name)
            if callable(attr) and not attr_name.startswith("_") and attr.__module__ == builtin_tools.__name__:
                tools[attr_name] = attr
                # Add to globals so search_frame can find it
                globals()[attr_name] = attr

    except Exception as e:
        print(f"Error loading builtin tools: {e}")

    return tools
