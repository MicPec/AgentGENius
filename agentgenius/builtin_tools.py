from typing import Any, Dict, List, Optional


def get_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current datetime as a string in the specified python format."""
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


def get_installed_packages() -> List[str]:
    """Get a list of all installed python packages and their versions in the current environment."""
    import pkg_resources

    # return [pkg.key for pkg in pkg_resources.working_set]
    return [f"{str(pkg.key)} {pkg.version}" for pkg in pkg_resources.working_set]


def get_user_name() -> str:
    """Get the username of the current user."""
    import getpass

    return getpass.getuser()


def get_weather_forecast(latitude: float, longitude: float) -> str:
    """Get the current weather and forecast for the given latitude and longitude.

    Args:
        latitude: The latitude of the location
        longitude: The longitude of the location

    Returns:
        str: The weather data in JSON format
    """
    import requests

    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    return "Error: Unable to retrieve weather data"


def get_duckduckgo_zero_click(query):
    import requests

    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    response = requests.get(url, params=params)
    data = response.json()
    return data


def get_wikipedia_summary(title, language="en") -> str:
    """Get the summary of a Wikipedia page.

    Args:
        title (str): The title of the Wikipedia page.
        language (str): The language of the Wikipedia page (default: "en").

    Returns:
        str: The summary of the Wikipedia page.
    """
    import wikipediaapi

    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="AgentGENius", language=language, extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    page = wiki_wiki.page(title)

    if page.exists():
        return page.summary
    else:
        return "The page does not exist."


def get_wikipedia_page(title, language="en", max_length=1000) -> str:
    """Get the full text of a Wikipedia page.

    Args:
        title (str): The title of the Wikipedia page.
        language (str): The language of the Wikipedia page (default: "en").
        max_length (int): The maximum length of the page text (default: 1000).

    Returns:
        str: The full text of the Wikipedia page.
    """
    import wikipediaapi

    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="AgentGENius", language=language, extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    page = wiki_wiki.page(title)

    if page.exists():
        return page.text[:max_length] + "..." if len(page.text) > max_length else page.text
    else:
        return "The page does not exist."


def web_search(query: str, max_results: Optional[int] = 10) -> dict:
    """Search the web using Tavily API

    Args:
        query: The search query string
        max_results: Number of results to return (default: 10, max: 20)
    """
    import os

    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=api_key)
    try:
        response = tavily_client.search(query, max_results=max_results)
        return response
    except Exception as e:
        return f"Error searching web: {str(e)}"


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read content from a file. Returns the file content as a string."""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> str:
    """Write content to a file. Returns success message or error."""
    import os

    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


def list_directory(path: str, pattern: Optional[str] = None) -> List[str]:
    """List files and directories in the specified path. Optionally filter by pattern."""
    from pathlib import Path

    try:
        if pattern:
            return [str(p) for p in Path(path).glob(pattern)]
        return [str(p) for p in Path(path).iterdir()]
    except Exception as e:
        return [f"Error listing directory: {str(e)}"]


def read_json(file_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its content as a dictionary."""
    import json

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Error reading JSON file: {str(e)}"}


def write_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> str:
    """Write a dictionary to a JSON file. Returns success message or error."""
    import json
    import os

    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return f"Successfully wrote JSON to {file_path}"
    except Exception as e:
        return f"Error writing JSON file: {str(e)}"


def extract_text_from_url(url: str, max_chars: int = 2000) -> str:
    """Extract readable text content from a URL using readability algorithms.

    Args:
        url: The URL to extract text from
        max_chars: Maximum number of characters to return (default: 5000)
    """
    try:
        import trafilatura

        # Fetch URL content without timeout (trafilatura handles timeouts internally)
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return "Error: Could not download the webpage"

        # Extract text content
        text = trafilatura.extract(
            downloaded, include_links=True, include_images=False, include_tables=True, no_fallback=False
        )

        if text:
            # Truncate if necessary
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            return text

        return "No readable text content found"

    except Exception as e:
        return f"Error extracting text: {str(e)}"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get detailed information about a file."""
    from datetime import datetime
    from pathlib import Path

    try:
        path = Path(file_path)
        stat = path.stat()
        return {
            "name": path.name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "absolute_path": str(path.absolute()),
        }
    except Exception as e:
        return {"error": f"Error getting file info: {str(e)}"}


def get_home_directory() -> str:
    """Get the home directory of the current user."""
    from pathlib import Path

    try:
        return str(Path.home())
    except Exception as e:
        return f"Error getting home directory: {str(e)}"


def get_current_working_directory() -> str:
    """Get the current working directory (cwd)."""
    import os

    try:
        return os.getcwd()
    except Exception as e:
        return {"error": f"Error getting current working directory: {str(e)}"}


def get_operating_system() -> str:
    """Get the operating system of the current machine."""
    import platform

    try:
        return platform.system()
    except Exception as e:
        return {"error": f"Error getting operating system: {str(e)}"}


def open_with_default_application(file_path: str) -> dict:
    """
    Execute command to open a file using the default application associated with its file type in the user's system.

    Args:
        file_path (str): The path to the file.

    Returns:
        dict: A dictionary containing 'success' boolean and optional 'error' message
    """
    import os
    import platform
    import subprocess

    try:
        system_name = platform.system().lower()
        if system_name == "darwin":  # macOS
            result = subprocess.run(["open", file_path], capture_output=True, text=True)
        elif system_name == "windows":
            os.startfile(file_path)  # pylint: disable=no-member
            result = subprocess.CompletedProcess(args=[], returncode=0)
        else:  # Assuming Linux or other Unix-like OS
            result = subprocess.run(["xdg-open", file_path], capture_output=True, text=True)

        if result.returncode == 0:
            return {"success": True}
        else:
            return {
                "success": False,
                "error": f"Command failed with return code {result.returncode}. Error: {result.stderr}",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


def check_file_existence(file_path: str) -> bool:
    """
    Checks if a specified file exists at the given file path.

    Args:
        file_path (str): The path to the file whose existence is to be checked.

    Returns:
        bool: True if the file exists, False otherwise.

    Raises:
        ValueError: If the provided path is empty.
    """

    import os

    if not file_path:
        raise ValueError("The file path cannot be empty.")

    try:
        return os.path.isfile(file_path)
    except Exception as e:
        return False


def identify_language(query: str) -> str:
    """
    Detect the language of the user's query.

    Args:
        query (str): The query whose language needs to be detected.

    Returns:
        str: The ISO 639-1 code of the detected language.

    Raises:
        ValueError: If the language cannot be detected from the text.
    """

    from langdetect import detect, DetectorFactory, LangDetectException

    # Ensure reproducibility by setting the random seed
    DetectorFactory.seed = 0

    try:
        language_code = detect(query)
        return language_code
    except LangDetectException as e:
        return f"Could not detect the language: {e}"


def scrape_webpage(
    url: str,
    selectors: Optional[Dict[str, str]] = None,
    dynamic: bool = False,
    wait_time: int = 0,
    headers: Optional[Dict[str, str]] = None,
    extract_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Universal web scraping function designed for AI agent use.

    Args:
        url (str): The URL to scrape
        selectors (Dict[str, str], optional): CSS selectors to extract specific content
            Example: {"title": "h1.main-title", "price": "span.price"}
        dynamic (bool): Whether to use Selenium for JavaScript-rendered content
        wait_time (int): Seconds to wait for dynamic content to load (only used if dynamic=True)
        headers (Dict[str, str], optional): Custom headers for the request
        extract_metadata (bool): Whether to extract page metadata (title, description, etc.)

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'content': Main page content
            - 'selected_content': Content matching provided selectors (if any)
            - 'metadata': Page metadata (if extract_metadata=True)
            - 'status': Success/failure status
            - 'error': Error message if any
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        import json
        from urllib.parse import urljoin

        # Default headers to mimic a browser
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        headers = headers or default_headers

        result = {"content": "", "selected_content": {}, "metadata": {}, "status": "success", "error": None}

        if dynamic:
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC

                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")

                driver = webdriver.Chrome(options=chrome_options)
                driver.get(url)

                if wait_time > 0:
                    WebDriverWait(driver, wait_time)

                page_source = driver.page_source
                driver.quit()

            except Exception as e:
                return {
                    "content": "",
                    "selected_content": {},
                    "metadata": {},
                    "status": "error",
                    "error": f"Selenium error: {str(e)}",
                }
        else:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            page_source = response.text

        soup = BeautifulSoup(page_source, "html.parser")

        # Extract main content (remove scripts, styles, and other non-content elements)
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        result["content"] = soup.get_text(separator=" ", strip=True)

        # Extract content based on provided selectors
        if selectors:
            for key, selector in selectors.items():
                elements = soup.select(selector)
                if elements:
                    # If multiple elements found, return a list
                    if len(elements) > 1:
                        result["selected_content"][key] = [elem.get_text(strip=True) for elem in elements]
                    else:
                        result["selected_content"][key] = elements[0].get_text(strip=True)
                else:
                    result["selected_content"][key] = None

        # Extract metadata if requested
        if extract_metadata:
            metadata = {}

            # Title
            title_tag = soup.find("title")
            metadata["title"] = title_tag.string if title_tag else None

            # Meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            metadata["description"] = meta_desc.get("content") if meta_desc else None

            # Open Graph metadata
            og_tags = soup.find_all("meta", property=lambda x: x and x.startswith("og:"))
            metadata["og"] = {tag.get("property")[3:]: tag.get("content") for tag in og_tags}

            # Links
            links = soup.find_all("a", href=True)
            metadata["links"] = [urljoin(url, link["href"]) for link in links]

            result["metadata"] = metadata

        return result

    except Exception as e:
        return {"content": "", "selected_content": {}, "metadata": {}, "status": "error", "error": str(e)}


if __name__ == "__main__":
    pass
