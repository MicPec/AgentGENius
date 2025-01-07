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


def get_user_ip() -> str:
    """Get the public IP address of the current machine using an external service."""
    import requests

    try:
        response = requests.get("https://ifconfig.me")
        return response.text.strip()
    except requests.RequestException as e:
        return f"Error: {str(e)}"


def get_location_by_ip(ip_address: str) -> str:
    """Get the location (city, region, country, coordinates) of the given IP address."""
    import requests

    url = f"https://apip.cc/api-json/{ip_address}"
    response = requests.get(url)
    if response.status_code == 200:
        location_data = response.text.strip()
        return location_data
    else:
        return "Error: Unable to retrieve location data"
