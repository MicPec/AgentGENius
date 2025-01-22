import inspect
from types import GenericAlias
from typing import Any
import json
from functools import wraps
from pathlib import Path
from datetime import datetime

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
            result = func(self, task_def, task_history, *args, **kwargs)

            # Create history directory if it doesn't exist
            history_dir = Path("history")
            history_dir.mkdir(exist_ok=True)

            # Prepare the full history data
            history_data = []
            history_file = history_dir / filename

            # Load existing history if file exists
            if history_file.exists():
                try:
                    with open(history_file, "r") as f:
                        history_data = json.load(f)
                except json.JSONDecodeError:
                    history_data = []

            # Add new task history with timestamp
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "task": task_def.name,
                "result": result.data if hasattr(result, "data") else str(result),
            }
            history_data.append(history_entry)

            # Save updated history to file
            try:
                with open(history_file, "w") as f:
                    json.dump(history_data, f, indent=2, default=str)
            except Exception as e:
                print(f"Failed to save task history: {e}")

            return result

        return wrapper

    return decorator
