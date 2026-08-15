import json
import re
from typing import Any, Iterable
import logging

from models import OllamaLLM
from .utils import persist_llm_interaction

logger = logging.getLogger(__name__)

class RelationSketcherAgent:
    """Agent responsible for generating a minimal conceptual bridge between a user query and a selected post.
    
    Always returns a string - uses "[Direct Connection]" as fallback if LLM fails or returns empty response.
    """

    def __init__(self, llm: OllamaLLM | None = None):
        """Create a :class:`SketcherAgent` instance.

        Parameters
        ----------
        llm : OllamaLLM, optional
            Optional language‑model wrapper.  If *None*, a new
            :class:`OllamaLLM` is instantiated.
        """
        self.llm = llm or OllamaLLM()

    def sketch(self, user_query: str, selected_post: dict[str, Any]) -> str:
        """Generate a minimal conceptual bridge between the user query and the selected post.

        Parameters
        ----------
        user_query : str
            The original user request.
        selected_post : dict[str, Any]
            Dictionary containing at least ``title`` and ``wire`` fields for
            the chosen post.

        Returns
        -------
        str
            The raw text produced by the language model, or a fallback
            "[Direct Connection]" if no post was supplied or response is empty.
        """
        if not selected_post:
            logger.warning("No selected post provided to SketcherAgent, using fallback")
            return "[Direct Connection]"

        prompt = self._build_prompt(user_query, selected_post)

        logger.debug(f"SketcherAgent request:\n{prompt}")

        response = self.llm.invoke(prompt)

        logger.debug(f"SketcherAgent response:\n{response}")

        # persist the LLM interaction for later inspection/training
        try:
            persist_llm_interaction(
                agent_name="SketcherAgent",
                llm_model=self.llm.model,
                user_query=user_query,
                prompt=prompt,
                response=response
            )
        except Exception:
            # swallow any persistence errors to avoid affecting main flow
            logger.exception("Failed to persist sketcher interaction")

        # Handle empty or invalid responses with fallback
        if not response or not response.strip():
            logger.warning("SketcherAgent returned empty response, using fallback")
            return "[Direct Connection]"

        return response or "[Direct Connection]"

    def _build_prompt(self, user_query: str, selected_post: list[dict[str, Any]]) -> str:
        """Construct the prompt sent to the LLM.

        Parameters
        ----------
        user_query : str
            The raw text from the user.
        selected_post : dict[str, Any]
            The selected post information.  Expected keys: ``title`` and
            ``wire``.

        Returns
        -------
        str
            A formatted prompt string.
        """
        post_title = selected_post.get('title')
        post_summary = selected_post.get('wire')

        logger.debug(f"post_summary: {post_summary}")

        return f"""
You are a structural reasoning engine and conceptual visualizer specializing in non-linear abstraction mapping.

Your sole task is to generate a minimal conceptual bridge between a User Query and a Post.

INPUT DATA:
- User Query: "{user_query}"
- Post Title: "{post_title}"
- Post Summary: "{post_summary}"

OPERATIONAL DIRECTIVES:
1. ABSTRACT ASSOCIATIONS: Link the Query to the Post via maximum 2-3 intermediate abstraction nodes.
2. GRAMMAR & SYNTAX RULES:
   - Output MUST be strictly a single line of text.
   - Every node concept MUST be enclosed in square brackets: [Node]
   - Every directed relationship MUST be uppercase, concise, and enclosed in double-equals arrow syntax: ==(RELATION)==>
3. NO EXTRA TEXT: Do not output introductions, summaries, markdown code blocks, or conclusions.
"""