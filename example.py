#!/usr/bin/env python3
"""Simple CLI chat application using AgentGENius."""

import sys
from pathlib import Path

import logfire
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from agentgenius.main import AgentGENius

load_dotenv()
logfire.configure(send_to_logfire="if-token-present", console=False)


def main():
    console = Console()
    console.print("[bold blue]AgentGENius Chat[/bold blue]")
    console.print("Type 'bye' or press Ctrl+C to quit\n")

    # Initialize agent
    agent = AgentGENius(
        system_prompt="""You are a helpful AI assistant. You can help with various tasks like:
        - Getting current time and date
        - Checking system information
        - Reading and writing files
        - Web searches
        - And more!
        
        Always try to be concise and helpful in your responses."""
    )

    try:
        while True:
            # Get user input
            query = Prompt.ask("\n[bold green]You[/bold green]")

            # Check for exit command
            if query.lower() in ("exit", "quit", "bye"):
                console.print("\n[bold blue]Goodbye![/bold blue]")
                break

            # Process query and display response
            console.print("\n[bold purple]Assistant[/bold purple]")
            try:
                response = agent.ask_sync(query)
                console.print(Markdown(response))
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

    except KeyboardInterrupt:
        console.print("\n[bold blue]Goodbye![/bold blue]")
        sys.exit(0)


if __name__ == "__main__":
    main()
