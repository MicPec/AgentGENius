import importlib
import inspect
import json
import sys
import tempfile
from functools import wraps
from pathlib import Path
from types import GenericAlias
from typing import Any, Dict
from functools import cache

from pydantic import TypeAdapter


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
            return frame.f_locals | frame.f_globals
        result = frame.f_globals.get(value)
        if result is not None:
            return frame.f_globals
        frame = frame.f_back
    raise ValueError(f"'{value}' not found")


def save_task_history(filename: str = "task_history.json"):
    """A decorator that saves task history to a JSON file after each task is processed."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, task_def, task_history, *args, **kwargs):
            # Call the original function
            func(self, task_def, task_history, *args, **kwargs)

            # Create history directory if it doesn't exist
            history_dir = Path("history")
            history_dir.mkdir(exist_ok=True)

            # Save history to JSON file
            history_file = history_dir / filename
            try:
                # Load existing history if file exists
                existing_history = []
                if history_file.exists():
                    try:
                        with open(history_file, "r") as f:
                            existing_history = json.load(f)
                    except json.JSONDecodeError:
                        existing_history = []

                # Convert current history to JSON-serializable format
                current_item = task_history.items[-1] if task_history.items else None
                if current_item:
                    history_item = {
                        "user_query": current_item.user_query,
                        "tasks": [{"query": task.query, "result": task.result} for task in current_item.tasks],
                        "final_result": current_item.final_result,
                    }

                    # Update or append history item
                    updated = False
                    for item in existing_history:
                        if item["user_query"] == history_item["user_query"]:
                            item["tasks"] = history_item["tasks"]
                            if history_item["final_result"]:
                                item["final_result"] = history_item["final_result"]
                            updated = True
                            break

                    if not updated:
                        existing_history.append(history_item)

                # Write updated history back to file
                with open(history_file, "w") as f:
                    json.dump(existing_history, f, indent=2)

            except Exception as e:
                print(f"Error saving task history: {e}")

        return wrapper

    return decorator


def save_history(filename: str = "task_history.json"):
    """A decorator that saves the final result to the history file."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Call the original function
            result = func(self, *args, **kwargs)

            # Create history directory if it doesn't exist
            history_dir = Path("history")
            history_dir.mkdir(exist_ok=True)

            # Save history to JSON file
            history_file = history_dir / filename
            try:
                # Load existing history if file exists
                existing_history = []
                if history_file.exists():
                    try:
                        with open(history_file, "r") as f:
                            existing_history = json.load(f)
                    except json.JSONDecodeError:
                        existing_history = []

                # Get current history item
                current_item = self.history.get_current_item()
                if current_item:
                    history_item = {
                        "user_query": current_item.user_query,
                        "tasks": [{"query": task.query, "result": task.result} for task in current_item.tasks],
                        "final_result": result,
                    }

                    # Update or append history item
                    updated = False
                    for item in existing_history:
                        if item["user_query"] == history_item["user_query"]:
                            item.update(history_item)
                            updated = True
                            break

                    if not updated:
                        existing_history.append(history_item)

                # Write updated history back to file
                with open(history_file, "w") as f:
                    json.dump(existing_history, f, indent=2)

            except Exception as e:
                print(f"Error saving final result: {e}")

            return result

        return wrapper

    return decorator


def load_generated_tools() -> Dict[str, Any]:
    """Load all generated tools from the temporary directory and add them to globals().

    Returns:
        Dict[str, Any]: Dictionary mapping tool names to their function objects
    """
    tools = {}
    temp_dir = Path(tempfile.gettempdir()) / "agentgenius_tools"

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
        # Import the builtin_tools module
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
