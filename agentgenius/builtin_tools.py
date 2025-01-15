from pathlib import Path

# from .agents import AgentStore
# from .config import config
# from .tools import ToolSet


# def get_all_agents() -> list[str]:
#     """Get list of all available agents"""
#     return AgentStore(config.agents_path).load_agents().list()


# def get_all_tools() -> list[str]:
#     """Get list of all available tools"""
#     return ToolSet().list_all_tools()


# def get_external_tools() -> list[str]:
#     """Get list of all external tools"""
#     return ToolSet().list_external_tools()


# def get_builtin_tools() -> list[str]:
#     """Get list of all builtin tools"""
#     return ToolSet.list_builtin_tools()


def get_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current datetime as a string in the specified python format. '%Y-%m-%d %H:%M:%S'"""
    from datetime import datetime

    return datetime.now().strftime(format)


def get_user_ip() -> str:
    """Get the public IP address of the current machine using an external service."""
    import requests

    try:
        response = requests.get("https://ifconfig.me", timeout=10)
        return response.text.strip()
    except requests.RequestException as e:
        return f"Error: {str(e)}"


def get_location_by_ip(ip_address: str) -> str:
    """Get the location (city, region, country, coordinates) of the given IP address."""
    import requests

    url = f"https://apip.cc/api-json/{ip_address}"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        location_data = response.text.strip()
        return location_data
    else:
        return "Error: Unable to retrieve location data"


def get_installed_packages() -> str:
    """Get a list of all installed python packages and their versions in the current environment."""
    import pkg_resources

    return "\n".join([str(pkg) for pkg in pkg_resources.working_set])


def get_user_name() -> str:
    """Get the username of the current user."""
    import getpass

    return getpass.getuser()


def get_builtin_tools() -> list[str]:
    """Get list of all builtin tools"""
    return [
        func.__name__
        for func in globals().values()
        if callable(func) and func.__module__ == __name__ and not func.__name__.startswith("_")
    ]


def get_weather_forecast(latitude: float, longitude: float) -> str:
    """Get the current weather and forecast for the given latitude and longitude."""
    import requests

    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
        return "Error: Unable to retrieve weather data"


def search_web(query: str) -> str:
    """Search the web using duckduckgo API"""
    from duckduckgo_search import DDGS

    results = DDGS().text(query, max_results=5)
    return results
