#!/usr/bin/env python3
"""Simple CLI chat application using AgentGENius."""

import sys
from pathlib import Path

import logfire
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from agentgenius.main import AgentGENius, TaskStatus

load_dotenv()
# logfire.configure(send_to_logfire="if-token-present", console=False)


def status_callback(status: TaskStatus):
    """Callback function to display task status using Rich."""
    console = Console()
    status_text = f"[bold cyan]{status.task_name}[/bold cyan]: {status.status}"
    if status.progress is not None:
        status_text += f" ([bold green]{status.progress:.1f}%[/bold green])"
    console.print(" " * 80, end="\r")  # Clear the current line
    console.print(status_text, end="\r")


def main():
    console = Console()
    console.print("[bold blue]AgentGENius Chat[/bold blue]")
    console.print("Type 'bye' or press Ctrl+C to quit\n")

    # Initialize agent with callback
    agent = AgentGENius(model="openai:gpt-4", callback=status_callback)

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
