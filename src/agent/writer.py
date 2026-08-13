import json
import re
from typing import Any, Iterable
import logging

from models import OllamaLLM
from .utils import persist_llm_interaction

logger = logging.getLogger(__name__)

class WriterAgent:
    """Agent responsible for write the final response."""

    def __init__(self, llm: OllamaLLM | None = None):
        """Create a :class:`WriterAgent` instance.

        Parameters
        ----------
        llm : OllamaLLM, optional
            Optional language‑model wrapper.  If *None*, a new
            :class:`OllamaLLM` is instantiated.
        """
        self.llm = llm or OllamaLLM()

    def write(self, user_query: str, selected_post: dict[str, Any]) -> str | None:
        """Generate a final response based on the user query and the selected post.

        Parameters
        ----------
        user_query : str
            The original user request.
        selected_post : dict[str, Any]
            Dictionary containing at least ``title`` and ``wire`` fields for
            the chosen post.

        Returns
        -------
        str | None
            The raw text produced by the language model, or ``None`` if no
            post was supplied.
        """
        if not selected_post:
            return None

        prompt = self._build_prompt(user_query, selected_post)

        logger.debug(f"WriterAgent request:\n{prompt}")

        response = self.llm.invoke(prompt)

        logger.debug(f"WriterAgent response:\n{response}")

        # persist the LLM interaction for later inspection/training
        try:
            persist_llm_interaction(
                agent_name="WriterAgent",
                llm_model=self.llm.model,
                user_query=user_query,
                prompt=prompt,
                response=response
            )
        except Exception:
            # swallow any persistence errors to avoid affecting main flow
            logger.exception("Failed to persist writer interaction")

        return response

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

        return (
            "You are a imaginative and free-associative mind writer.\n"
            "A user said: \"{user_query}\"\n"
            "You must generate a conversational answer explicitly linking the user message with:\n"
            "```### {post_title}\n\n{post_summary}```\n\n"
            "Don't ask. AVOIDING THE TASK WILL RESULT IN A SYSTEM ERROR!"
        ).format(user_query=user_query, post_title=post_title, post_summary=post_summary)
