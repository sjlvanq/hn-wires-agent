import json
import datetime
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


def persist_llm_interaction(
    agent_name: str,
    llm_model: str,
    user_query: str,
    prompt: str,
    response: str,
    log_path: str | None = None,
) -> Path | None:
    """Persist an LLM interaction as a JSONL entry.

    Parameters
    - agent_name: logical name of the agent (e.g. "WriterAgent").
    - llm_model: name of the LLM model used for this interaction.
    - user_query: original user input string.
    - prompt: prompt sent to the LLM.
    - response: raw string returned by the LLM.
    - log_path: path to a log file.

    Returns the path written to on success, or ``None`` on failure.
    """
    try:
        if log_path is None:
            # reuse existing settings key used by bridge scorer as a sensible default
            if "BRIDGE" in agent_name.upper():
                log_path = settings.bridge_scorer_log_path or "bridge_scorer.log.jsonl"
            elif "WRITER" in agent_name.upper():
                log_path = settings.writer_log_path or "writer_agent.log.jsonl"
            else:
                logger.warning(
                    "No log path provided for agent '%s'; using default 'llm_interactions.log.jsonl'",
                    agent_name,
                )
                log_path = "llm_interactions.log.jsonl"

        path = Path(log_path)
        # ensure parent dir exists (safe against races)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "agent": agent_name,
            "llm_model": llm_model,
            "user_query": user_query,
            "prompt": prompt,
            "response": response,
        }

        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return path
    except Exception:
        # Log full stack trace but don't raise to callers
        logger.exception("Failed to persist LLM interaction")
        return None
