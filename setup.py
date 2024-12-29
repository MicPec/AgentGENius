from setuptools import setup, find_packages

setup(
    name="agentgenius",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "pydantic-ai",
        "python-dotenv",
        "pytest",
        "pytest-asyncio",
    ],
)
