from pathlib import Path

# from .agents import AgentStore
# from .config import config
# from .tools import ToolSet


def get_all_agents() -> list[str]:
    """Get list of all available agents"""
    return AgentStore(config.agents_path).load_agents().list()


def get_all_tools() -> list[str]:
    """Get list of all available tools"""
    return ToolSet().list_all_tools()


def get_external_tools() -> list[str]:
    """Get list of all external tools"""
    return ToolSet().list_external_tools()


def get_builtin_tools() -> list[str]:
    """Get list of all builtin tools"""
    return ToolSet.list_builtin_tools()


def get_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current datetime as a string in the specified python format."""
    from datetime import datetime

    return datetime.now().strftime(format)


def get_user_ip_and_location() -> str:
    """Get the public IP address and location of the current machine using an external service."""
    import requests

    try:
        response = requests.get("https://ipinfo.io")
        return response.text
    except requests.RequestException as e:
        return f"Error: {str(e)}"


def get_installed_packages() -> str:
    """Get a list of all installed Python packages."""
    import pkg_resources

    installed_packages = pkg_resources.working_set
    sorted_packages = sorted([f"{i.key}" for i in installed_packages])
    return ", ".join(sorted_packages)
