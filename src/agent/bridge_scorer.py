import json
import re
from typing import Any, Iterable
import datetime
import os
import logging

from models import OllamaLLM

logger = logging.getLogger(__name__)

class BridgeScorerAgent:
    """Agent responsible for scoring candidate posts against a user query and returning a JSON evaluation."""

    EVALUATION_SCHEMA = {
        "evaluation": [
            {
                "post_id": "string",
                "emotional_connection_score": "float",
                "conceptual_connection_score": "float",
                "metaphorical_connection_score": "float"
            }
        ]
    }

    def __init__(self, llm: OllamaLLM | None = None):
        """Create a :class:`BridgeScorerAgent`.

        Parameters
        ----------
        llm : OllamaLLM, optional
            Optional language‑model wrapper.  If *None*, a new
            :class:`OllamaLLM` is instantiated.
        """
        self.llm = llm or OllamaLLM()

    def evaluate(self, user_query: str | dict, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Evaluate candidate posts against the user query.

        Parameters
        ----------
        user_query : str | dict
            Either the raw query string or a dictionary containing query
            metadata. For dicts, the keys ``query`` or ``text`` are
            searched first.
        candidates : list[dict[str, Any]]
            List of candidate post dictionaries. Each must contain at
            least an ``id`` field and optionally ``summary`` or ``title``.

        Returns
        -------
        dict[str, Any] | None
            A dictionary with an ``evaluation`` array containing per-post
            scores, or ``None`` if no candidates were supplied.
        """
        if not candidates:
            logger.warning("No candidates provided for evaluation.")
            return None

        self._validate_candidates(candidates)

        if isinstance(user_query, str):
            user_query_text = user_query
        elif isinstance(user_query, dict):
            # Prefer common keys if provided
            user_query_text = user_query.get("query") or user_query.get("text") or str(user_query)
        else:
            user_query_text = str(user_query)

        prompt = self._build_prompt(user_query_text, candidates)

        logger.debug(f"BridgeScorerAgent request:\n{prompt}")

        response = self.llm.invoke(prompt)

        logger.debug(f"BridgeScorerAgent response:\n{response}")
        # Persist request/response pair
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "user_query": user_query_text,
            "candidates": candidates,
            "prompt": prompt,
            "response": response,
        }
        self._persist_log(log_entry)

        return self._parse_evaluation(response)

    def _build_prompt(self, user_query: str, candidates: list[dict[str, Any]]) -> str:
        """Construct an LLM prompt requesting a JSON evaluation.

        The prompt builds the candidate list in a readable form, then asks
        the model to return a strict JSON object containing an
        ``evaluation`` array with per-post scores.
        """
        candidate_lines = []
        for candidate in candidates:
            title = candidate.get("title") or ""
            summary = candidate.get("summary") or ""
            candidate_lines.append(
                f"- post_id: {candidate.get('id')}\n"
                f"  title: {title}\n"
                f"  summary: {summary}\n"
            )

        candidate_text = "\n".join(candidate_lines)

        # In _build_prompt method
        schema_json = json.dumps(EVALUATION_SCHEMA, indent=2)

        return (
            "You are a post scoring assistant.\n"
            f"User query: \"{user_query}\"\n"
            "Below are candidate Hacker News posts retrieved by semantic search:\n\n"
            f"{candidate_text}\n"
            f"Return a JSON object matching the following structure:\n{schema_json}\n"
            "The response must contain only this JSON and nothing else."
        )

    def _parse_evaluation(self, text: str) -> dict | None:
        """Parse the LLM response to extract evaluation scores.

        Parameters
        ----------
        text : str
            Raw string returned by the LLM.

        Returns
        -------
        dict | None
            A parsed dictionary containing an ``evaluation`` array, or
            ``None`` if the response could not be decoded.
        """
        text = text.strip()
        try:
            parsed = json.loads(text)
            # Expect a top‑level dict with an ``evaluation`` key.
            if isinstance(parsed, dict) and "evaluation" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: try to load a JSON object directly from any match.
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _persist_log(self, entry: dict) -> None:
        """Append a log entry to ``bridge_scorer.log.jsonl``.

        The log file is created in the same directory as the agent
        implementation for easy reference when training a dedicated
        model.
        """
        log_path = os.path.join(os.path.dirname(__file__), "bridge_scorer.log.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write bridge scorer log: %s", exc)

    def select_batch(self, user_queries: Iterable[str | dict], candidates: list[dict[str, Any]]) -> list[dict | None]:
        """Score each query against the same candidate list.

        Parameters
        ----------
        user_queries : Iterable[str | dict]
            Sequence of user queries to process.
        candidates : list[dict[str, Any]]
            The same candidate list used for each query.

        Returns
        -------
        list[dict | None]
            List of evaluation dictionaries (one per query) or ``None`` if
            no candidates were supplied.
        """
        return [self.evaluate(query, candidates) for query in user_queries]

    def _validate_candidates(self, candidates: list[dict[str, Any]]) -> None:
        """Validate candidate structure.

        Parameters
        ----------
        candidates : list[dict[str, Any]]
            Candidate posts to validate.

        Raises
        ------
        ValueError
            If candidates are missing required fields ('id', 'title', 'summary').
        """
        for i, candidate in enumerate(candidates):
            if "id" not in candidate:
                raise ValueError(f"Candidate at index {i} missing required field 'id'")
            if not any(key in candidate for key in ["title", "summary"]):
                raise ValueError(f"Candidate at index {i} missing 'title' or 'summary'")

