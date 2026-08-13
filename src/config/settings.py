from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    ollama_embedding_model: str = Field(
        default="all-minilm:33m",
        description="Ollama model for embeddings"
    )

    # Database Configuration
    database_path: str = Field(
        default="db.sqlite",
        description="Path to SQLite database"
    )

    # BridgeScorer Configuration
    bridge_scorer_log_path: str = Field(
        default="bridge_scorer.log.jsonl",
        description="Path to log file for BridgeScorer evaluations"
    )

    # Agent Configuration
    orchestrator_llm_model: str = Field(
        default="functiongemma:270m",
        description="Ollama model for orchestrator"
    )
    orchestrator_max_context_length: int = Field(
        default=2048,
        description="Maximum context length for the agent"
    )
    orchestrator_temperature: float = Field(
        default=0.7,
        description="Temperature for LLM generation"
    )
    orchestrator_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for LLM generation (small model constraint)"
    )
    orchestrator_num_ctx: int = Field(
        default=1024,
        description="Number of context tokens for the agent"
    )
    orchestrator_reasoning: bool | None = Field(
        default=None,
        description="Enable reasoning capabilities in the agent"
    )

    # Bridge Scorer Agent Configuration
    bridge_scorer_llm_model: str = Field(
        default="nemotron-3-nano:30b-cloud",
        description="Ollama model used by the Bridge Scorer Agent"
    )

    # Selector Agent Configuration
    selector_llm_model: str = Field(
        default="alibayram/hunyuan:0.5b",
        description="Ollama model used only by the selector"
    )
    selector_temperature: float = Field(
        default=0.2,
        description="Temperature for selector LLM generation"
    )
    selector_max_tokens: int = Field(
        default=127,
        description="Max tokens for selector LLM generation"
    )
    selector_num_ctx: int = Field(
        default=1024,
        description="Number of context tokens for the agent"
    )
    selector_reasoning: bool | None = Field(
        default=None,
        description="Enable reasoning capabilities for selector LLM"
    )

    # Writer Agent Configuration
    writer_llm_model: str = Field(
        default="alibayram/hunyuan:0.5b",
        description="Ollama model used only by the selector"
    )
    writer_temperature: float = Field(
        default=0.7,
        description="Temperature for writer LLM generation"
    )
    writer_max_tokens: int = Field(
        default=512,
        description="Max tokens for writer LLM generation"
    )
    writer_num_ctx: int = Field(
        default=512,
        description="Number of context tokens for the agent"
    )
    writer_reasoning: bool | None = Field(
        default=None,
        description="Enable reasoning capabilities for writer LLM"
    )

    # Vector Search Configuration
    wires_vector_search_top_k: int = Field(
        default=5,
        description="Number of top results to return from vector search"
    )
    wires_vector_search_threshold: float = Field(
        default=0.7,
        description="Similarity threshold for vector search"
    )
    vector_search_vector_k: int = Field(
        default=384,
        description="Dimensionality of the embedding vectors used in vector search"
    )

    @property
    def database_path_absolute(self) -> Path:
        """Get absolute path to database file."""
        return Path(self.database_path).absolute()

    @property
    def database_dir(self) -> Path:
        """Get directory containing the database file."""
        return self.database_path_absolute.parent


settings = Settings()
