import json
import re
from typing import Any, Iterable
import logging

from models import OllamaLLM

logger = logging.getLogger(__name__)

class WriterAgent:
    """Agent responsible for write the final response."""

    def __init__(self, llm: OllamaLLM | None = None):
        self.llm = llm or OllamaLLM()

    def write(self, user_query: str, selected_post: dict[str, Any]) -> int | None:
        """Generate a final response based on the user's query and the selected post."""
        if not selected_post:
            return None

        prompt = self._build_prompt(user_query, selected_post)

        logger.debug(f"WriterAgent request:\n{prompt}")

        response = self.llm.invoke(prompt)

        logger.debug(f"WriterAgent response:\n{response}")

        return response

    def _build_prompt(self, user_query: str, selected_post: list[dict[str, Any]]) -> str:
        post_title = selected_post.get('title')
        post_summary = selected_post.get('wire')

        logger.debug(f"post_summary: {post_summary}")

        return (
            "You are a imaginative and free-associative mind writer.\n"
            "A user said: \"{user_query}\"\n"
            "You must generate a conversational answer explicitly linking the user message with:\n"
            "```### {post_title}\n\n{post_summary}```\n\n"
            "Don't ask. AVOIDING THE TASK WILL RESULT IN A SYSTEM ERROR!"
        ).format(user_query=user_query, post_title=post_title, post_summary=post_summary)
