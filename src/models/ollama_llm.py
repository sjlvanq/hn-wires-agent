from typing import Optional

# https://reference.langchain.com/python/langchain-ollama
from langchain_ollama import ChatOllama
from config.settings import settings

import logging
logger = logging.getLogger(__name__)

class OllamaLLM:
    """Thin wrapper around :class:`langchain_ollama.ChatOllama`.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        num_ctx: Optional[int] = None,
        reasoning: Optional[bool] = None,
    ):
        """Create a new :class:`OllamaLLM` instance.

        Parameters
        ----------
        model : str | None
            Name of the Ollama model to use.  Defaults to
            :data:`settings.orchestrator_llm_model`.
        base_url : str | None
            Base URL for the Ollama server.  Defaults to
            :data:`settings.ollama_base_url`.
        temperature : float | None
            Sampling temperature.  Defaults to
            :data:`settings.orchestrator_temperature`.
        max_tokens : int | None
            Maximum number of tokens to return.  Defaults to
            :data:`settings.orchestrator_max_tokens`.
        num_ctx : int | None
            Number of context tokens to keep.  Defaults to
            :data:`settings.orchestrator_num_ctx`.
        reasoning : bool | None
            Pass-through flag enabling reasoning.  Defaults to
            :data:`settings.orchestrator_reasoning`.
        """
        self.model = model or settings.orchestrator_llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature or settings.orchestrator_temperature
        self.max_tokens = max_tokens or settings.orchestrator_max_tokens
        self.num_ctx = num_ctx or settings.orchestrator_num_ctx
        self.reasoning = reasoning if reasoning is not None else settings.orchestrator_reasoning

        self._llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            num_ctx=self.num_ctx,
            reasoning=self.reasoning
        )

    @property
    def llm(self) -> ChatOllama:
        """Get the underlying LangChain ChatOllama instance."""
        return self._llm

    def invoke(self, prompt: str, **kwargs) -> str:
        """Send a prompt to the underlying model and return the text.

        Parameters
        ----------
        prompt : str
            Prompt string to send to the LLM.
        **kwargs : Any
            Additional arguments forwarded to ``ChatOllama.invoke``.

        Returns
        -------
        str
            The content of the model's response.
        """
        response = self._llm.invoke(prompt, **kwargs)
        logger.debug(f"LLM response:\n{response}")
        return response.content

