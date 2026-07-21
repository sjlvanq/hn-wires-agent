import json
import re
from typing import Any, Iterable
import logging

from models import OllamaLLM

logger = logging.getLogger(__name__)

class SelectorAgent:
    """Agent responsible for selecting the single most relevant post ID."""

    def __init__(self, llm: OllamaLLM | None = None):
        self.llm = llm or OllamaLLM()

    def select(self, user_query: str | dict, candidates: list[dict[str, Any]]) -> int | None:
        """Select the most relevant post ID from a list of candidate posts.

        Accepts either a plain string `user_query` or a structured dict containing
        a text/query field. This keeps calls simple (pass a string) while
        allowing structured metadata in the future.
        """
        if not candidates:
            return None

        if isinstance(user_query, str):
            user_query_text = user_query
        elif isinstance(user_query, dict):
            # Prefer common keys if provided
            user_query_text = user_query.get("query") or user_query.get("text") or str(user_query)
        else:
            user_query_text = str(user_query)

        prompt = self._build_prompt(user_query_text, candidates)

        logger.debug(f"SelectorAgent request:\n{prompt}")

        response = self.llm.invoke(prompt)

        logger.debug(f"SelectorAgent response:\n{response}")

        return self._parse_post_id(response)

    def _build_prompt(self, user_query: str, candidates: list[dict[str, Any]]) -> str:
        candidate_lines = []
        for candidate in candidates:
            summary = candidate.get('summary') or candidate.get('title') or ''
            summary_text = summary.split('.')[0] if isinstance(summary, str) else ''
            candidate_lines.append(
                f"- id: {candidate.get('id')}\n"
                f"  summary: {summary_text}\n"
            )

        candidate_text = "\n".join(candidate_lines)

        return (
            "You are a post selection assistant.\n"
            f"A user asked: \"{user_query}\"\n"
            "Below are candidate Hacker News posts retrieved by semantic search:\n\n"
            f"{candidate_text}\n"
            "Choose the single post that is the most relevant to the user request in any sense. "
            "Return ONLY the numeric post ID as a bare number on the first line of the answer. "
            "Do not include any additional text or explanation."
        )

    def _parse_post_id(self, text: str) -> int | None:
        """Extract the first integer post ID from the model response."""
        text = text.strip()
        # Try JSON first in case the model returns a structured response.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "post_id" in parsed:
                return int(parsed["post_id"])
            if isinstance(parsed, int):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\b(\d+)\b", text)
        if match:
            return int(match.group(1))

        return None

    def select_batch(self, user_queries: Iterable[str | dict], candidates: list[dict[str, Any]]) -> list[int | None]:
        """Optional helper for selecting a post id for multiple queries."""
        return [self.select(query, candidates) for query in user_queries]
