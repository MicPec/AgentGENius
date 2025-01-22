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
                        "tasks": [
                            {
                                "name": task.name,
                                "result": task.result
                            }
                            for task in current_item.tasks
                        ],
                        "result": current_item.result
                    }
                    
                    # Check if this item already exists in history
                    exists = False
                    for item in existing_history:
                        if item["user_query"] == history_item["user_query"]:
                            item.update(history_item)
                            exists = True
                            break
                    
                    if not exists:
                        existing_history.append(history_item)
                
                with open(history_file, "w") as f:
                    json.dump(existing_history, f, indent=2)
            except Exception as e:
                print(f"Failed to save history: {e}")

        return wrapper

    return decorator


def save_final_result(filename: str = "task_history.json"):
    """A decorator that saves the final result to the history file."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, result, *args, **kwargs):
            # Call the original function
            func(self, result, *args, **kwargs)

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

                # Update the most recent history item with the final result
                if existing_history:
                    existing_history[-1]["result"] = result

                with open(history_file, "w") as f:
                    json.dump(existing_history, f, indent=2)
            except Exception as e:
                print(f"Failed to save history: {e}")

        return wrapper

    return decorator
