from typing import Optional

# https://reference.langchain.com/python/langchain-ollama
from langchain_ollama import ChatOllama
from config.settings import settings

import logging
logger = logging.getLogger(__name__)

class OllamaLLM:
    """Wrapper for Ollama LLM."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        num_ctx: Optional[int] = None,
        reasoning: Optional[bool] = None
    ):
        """
        Initialize Ollama LLM.

        Args:
            model: Ollama model name. Defaults to settings.orchestrator_llm_model.
            base_url: Ollama base URL. Defaults to settings.ollama_base_url.
            temperature: Generation temperature. Defaults to settings.orchestrator_temperature.
            max_tokens: Maximum tokens to generate. Defaults to settings.orchestrator_max_tokens.
            num_ctx: Number of context tokens. Defaults to settings.orchestrator_num_ctx.
            reasoning: Enable reasoning capabilities. Defaults to settings.orchestrator_reasoning.
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
        """
        Invoke the LLM with a prompt.

        Args:
            prompt: Input prompt.
            **kwargs: Additional arguments for the LLM.

        Returns:
            Generated response text.
        """
        response = self._llm.invoke(prompt, **kwargs)
        logger.debug(f"LLM response:\n{response}")
        return response.content

