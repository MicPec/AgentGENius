import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def get_installed_packages() -> List[str]:
    """Get a list of all installed python packages and their versions in the current environment."""
    import pkg_resources

    return [pkg.key for pkg in pkg_resources.working_set]
    # return [f"{str(pkg.key)}=={pkg.version}" for pkg in pkg_resources.working_set]


def get_user_name() -> str:
    """Get the username of the current user."""
    import getpass

    return getpass.getuser()


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


def scrape_webpage(url: str, selector: Optional[str] = None, retry_count: int = 3, timeout: int = 10) -> str:
    """Scrape text content from a webpage using advanced BeautifulSoup4 techniques.

    Args:
        url: The webpage URL to scrape
        selector: Optional CSS selector to get specific content
        retry_count: Number of retries for failed requests (default: 3)
        timeout: Request timeout in seconds (default: 10)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    from bs4 import BeautifulSoup
    import requests

    def clean_text(text: str) -> str:
        """Clean extracted text by removing extra whitespace and normalizing line breaks."""
        import re

        # Replace multiple newlines and spaces with single ones
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def is_content_relevant(element) -> bool:
        """Check if an element likely contains relevant content."""
        # List of classes/IDs that typically indicate non-content areas
        blacklist = {
            "nav",
            "navigation",
            "menu",
            "sidebar",
            "footer",
            "header",
            "banner",
            "ad",
            "advertisement",
            "social",
            "comment",
            "cookie",
            "popup",
            "modal",
        }

        # Check element's class and id attributes
        element_classes = set(element.get("class", []))
        element_id = element.get("id", "").lower()

        # Check if element's classes or id contain blacklisted terms
        return not (
            any(term in element_id for term in blacklist)
            or any(any(term in cls.lower() for term in blacklist) for cls in element_classes)
        )

    def extract_content(soup: BeautifulSoup) -> str:
        """Extract main content from the page using various heuristics."""
        # Remove unwanted elements
        for element in soup(["script", "style", "noscript", "iframe", "head", "meta", "link"]):
            element.decompose()

        if selector:
            elements = soup.select(selector)
            if not elements:
                return "No elements found matching the selector"
            return clean_text("\n".join(elem.get_text(strip=True) for elem in elements))

        # Try to find main content area
        content_tags = [
            "article",
            "main",
            '[role="main"]',
            '[role="article"]',
            "#content",
            ".content",
            "#main",
            ".main",
        ]

        for tag in content_tags:
            main_content = soup.select_one(tag)
            if main_content and is_content_relevant(main_content):
                return clean_text(main_content.get_text(strip=True))

        # If no main content found, try to find the largest text block
        paragraphs = []
        for p in soup.find_all(["p", "div"]):
            if is_content_relevant(p):
                text = clean_text(p.get_text(strip=True))
                if len(text) > 100:  # Only consider paragraphs with substantial content
                    paragraphs.append(text)

        if paragraphs:
            return "\n\n".join(paragraphs)

        # Fallback: get all text from body
        body = soup.find("body")
        if body:
            return clean_text(body.get_text(strip=True))

        return clean_text(soup.get_text(strip=True))

    # Main scraping logic with retries
    for attempt in range(retry_count):
        try:
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Try to detect and handle character encoding correctly
            if response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "html.parser", from_encoding=response.encoding)

            # Handle meta refresh redirects
            meta_refresh = soup.find("meta", attrs={"http-equiv": lambda x: x and x.lower() == "refresh"})
            if meta_refresh:
                content = meta_refresh.get("content", "")
                if content and "url=" in content.lower():
                    redirect_url = content.split("url=", 1)[1].strip("'").strip('"')
                    if redirect_url != url:  # Avoid infinite loops
                        response = session.get(redirect_url, headers=headers, timeout=timeout)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.text, "html.parser")

            return extract_content(soup)

        except requests.RequestException as e:
            if attempt == retry_count - 1:
                return f"Error scraping webpage (after {retry_count} retries): {str(e)}"
            time.sleep(1)  # Wait before retrying
        except Exception as e:
            return f"Error processing webpage: {str(e)}"


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """Read content from a file. Returns the file content as a string."""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> str:
    """Write content to a file. Returns success message or error."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


def list_directory(path: str, pattern: Optional[str] = None) -> List[str]:
    """List files and directories in the specified path. Optionally filter by pattern."""
    try:
        if pattern:
            return [str(p) for p in Path(path).glob(pattern)]
        return [str(p) for p in Path(path).iterdir()]
    except Exception as e:
        return [f"Error listing directory: {str(e)}"]


def read_json(file_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its content as a dictionary."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Error reading JSON file: {str(e)}"}


def write_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> str:
    """Write a dictionary to a JSON file. Returns success message or error."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return f"Successfully wrote JSON to {file_path}"
    except Exception as e:
        return f"Error writing JSON file: {str(e)}"


def extract_text_from_url(url: str, max_chars: int = 5000, timeout: int = 10) -> str:
    """Extract readable text content from a URL using readability algorithms.

    Args:
        url: The URL to extract text from
        max_chars: Maximum number of characters to return (default: 5000)
        timeout: Timeout in seconds for the request (default: 10)
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


def search_text(text: str, query: str, context_words: int = 50) -> List[str]:
    """Search for a query in text and return matching excerpts with context."""
    try:
        words = text.split()
        matches = []
        for i, word in enumerate(words):
            if query.lower() in word.lower():
                start = max(0, i - context_words)
                end = min(len(words), i + context_words + 1)
                context = " ".join(words[start:end])
                matches.append(f"...{context}...")
        return matches if matches else ["No matches found"]
    except Exception as e:
        return [f"Error searching text: {str(e)}"]


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get detailed information about a file."""
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
        return {"error": f"Error getting home directory: {str(e)}"}


def get_user_operating_system() -> str:
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
            os.startfile(file_path)
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


if __name__ == "__main__":
    # Test each function
    print("Testing scrape_webpage:")
    print(scrape_webpage("https://example.com"))

    print("\nTesting read_file:")
    print(read_file("test.txt"))

    print("\nTesting write_file:")
    print(write_file("test_output.txt", "This is a test content"))

    print("\nTesting list_directory:")
    print(list_directory("."))

    print("\nTesting read_json:")
    print(read_json("test.json"))

    print("\nTesting write_json:")
    print(write_json("test_output.json", {"key": "value"}))

    print("\nTesting extract_text_from_url:")
    print(extract_text_from_url("https://example.com"))

    print("\nTesting search_text:")
    print(search_text("This is a sample text for searching.", "sample"))

    print("\nTesting get_file_info:")
    print(get_file_info("README.md"))

    print("\nTesting get_home_directory:")
    print(get_home_directory())

    print("\nTesting get_user_operating_system:")
    print(get_user_operating_system())

    print("\nTesting open_with_default_application:")
    print(open_with_default_application("README.md"))
