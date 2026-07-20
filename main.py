#!/usr/bin/env python3
"""Main CLI interface for the Hacker News Wires Agent."""
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.logging import RichHandler

import logging
import traceback

LOGGER_LEVEL = "WARNING" #DEBUG

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import NewsAgent
from config.settings import settings

# Load environment variables
load_dotenv()

console = Console()

def print_goodbye():
    """Print goodbye message."""
    console.print(Panel("Good bye!", border_style="green"))


def main():
    """Main CLI entry point."""

    init_logging()

    # Initialize agent
    try:
        agent = NewsAgent()
        console.print("Agent ready!", style="green")
    except Exception as e:
        console.print(f"Error initializing agent: {e}", style="red")
        console.print("Make sure Ollama is running and the database exists.", style="yellow")
        sys.exit(1)

    # Main conversation loop
    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]", default="")

            if user_input.lower() in ["quit", "exit", "q"]:
                print_goodbye()
                break

            if not user_input.strip():
                continue

            # Process the user's input
            console.print("[dim]Processing...[/dim]")
            result = agent.invoke(user_input)

            # Display retrieved candidates
            console.print()
            if result["retrieved"]:
                candidate_lines = [
                    f"{candidate['id']}: {candidate['title']} (score={candidate.get('similarity', 'N/A')})"
                    for candidate in result["retrieved"]
                ]
                console.print(Panel("\n".join(candidate_lines), title="Retrieved candidates", border_style="yellow"))
                console.print(f"Selected post ID: {result['selected_id']}", style="bold green")
            else:
                console.print(Panel("No candidates found.", title="Retrieved candidates", border_style="yellow"))


            if result["selected_post"]:
                post = result["selected_post"]
                selected_post = [
                    f"{post['id']}: \"{post['title']}\" by {post['author']}",
                    f"url: {post['url']}",
                    f"{post['wire']}"
                ]
                console.print()
                console.print(Panel("\n".join(selected_post), title="Selected Post", border_style="magenta"))

                keywords = ', '.join(
                    f"{keyword['keyword']} (ID: {keyword['id']})"
                    for keyword in result['keywords']
                )
                console.print(f"Keywords: [magenta]{keywords}[/magenta]", justify="right")

            console.print()
            if result["response"]:
                console.print(Panel(result["response"], title="Agent Response", border_style="cyan"))
                console.print()

        except KeyboardInterrupt:
            console.print("\nInterrupted by user.", style="yellow")
            print_goodbye()
            break
        
        # TODO: Implement more specific error handling with detailed logging.
        except Exception as e:
            console.print(f"Error: {e}", style="red")
            traceback.print_exc()
            console.print("Please try again.", style="yellow")

def init_logging():
    """Logging global configuration."""

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
    )

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    rich_handler.setLevel(LOGGER_LEVEL)
    rich_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(LOGGER_LEVEL)
    root_logger.handlers.clear()
    root_logger.addHandler(rich_handler)

if __name__ == "__main__":
    main()
