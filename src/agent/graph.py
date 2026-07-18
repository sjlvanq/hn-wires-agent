from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

import logging

from config.settings import settings
from database import NewsRepository
from embeddings import OllamaEmbeddings
from models import OllamaLLM
from .selector import SelectorAgent
from .writer import WriterAgent
from tools import (
    SearchSimilarByEmbeddingTool,
)

logger = logging.getLogger(__name__)

class AgentsState(TypedDict):
    """State of the associative conversational agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    candidates: list[dict]
    selected_post_id: int | None
    keywords: list[str]
    post_details: dict | None
    response: str | None


class NewsAgent:
    """
    LangGraph-based agent designed to enrich the user's conversation
    by systematically associating their inputs with Hacker News articles.
    """

    def __init__(
        self,
        llm: OllamaLLM | None = None,
        selector_llm: OllamaLLM | None = None,
        writer_llm: OllamaLLM | None = None,
        repository: NewsRepository | None = None,
        embeddings: OllamaEmbeddings | None = None,
    ):
        """
        Initialize the news agent.

        Args:
            llm: Ollama LLM instance. If None, creates new instance.
            repository: News repository instance. If None, creates new instance.
            embeddings: Embeddings instance. If None, creates new instance.
        """
        self.llm = llm or OllamaLLM()
        self.selector = SelectorAgent(
            selector_llm
            or OllamaLLM(
                model=settings.selector_llm_model,
                temperature=settings.selector_temperature,
                max_tokens=settings.selector_max_tokens,
                num_ctx=settings.selector_num_ctx,
                reasoning=settings.selector_reasoning
            )
        )
        self.writer = WriterAgent(
            writer_llm
            or OllamaLLM(
                model=settings.writer_llm_model,
                temperature=settings.writer_temperature,
                max_tokens=settings.writer_max_tokens,
                num_ctx=settings.writer_num_ctx,
                reasoning=settings.writer_reasoning
            )
        )

        self.repository = repository or NewsRepository()
        self.embeddings = embeddings or OllamaEmbeddings()

        self.search_tool = SearchSimilarByEmbeddingTool(self.repository, self.embeddings)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentsState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("select", self._select_node)
        workflow.add_node("keywords", self._keywords_node)
        workflow.add_node("fetch", self._fetch_node)
        workflow.add_node("respond", self._respond_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "select")
        workflow.add_edge("select", "keywords")
        workflow.add_edge("keywords", "fetch")
        workflow.add_edge("fetch", "respond")
        workflow.add_edge("respond", END)

        return workflow.compile()

    def _retrieve_node(self, state: AgentsState) -> AgentsState:
        """Retrieve candidate posts using semantic search."""
        query = state["messages"][-1].content
        candidates = self.search_tool.search(query, top_k=settings.wires_vector_search_top_k)
        
        return {
            **state,
            "candidates": candidates,
        }

    def _select_node(self, state: AgentsState) -> AgentsState:
        """Select the single most relevant post ID from the retrieved candidates."""
        selected_post_id = self.selector.select(state["messages"][-1].content, state["candidates"])
        return {
            **state,
            "selected_post_id": selected_post_id,
        }

    def _keywords_node(self, state: AgentsState) -> AgentsState:
        """Extract keywords from the selected post."""
        keywords = []

        if state["selected_post_id"] is not None:
            keywords = self.repository.get_post_keywords(state["selected_post_id"])

        logger.debug(f"""keywords: {keywords}""")

        return {
            **state,
            "keywords": keywords,
        }

    def _fetch_node(self, state: AgentsState) -> AgentsState:
        """Fetch the full post details for the selected post ID."""

        post_details = None
        if state["selected_post_id"] is not None:
            post_details = self.repository.get_post_with_wire(state["selected_post_id"])

        logger.debug(f"""post_details: {post_details}""")

        return {
            **state,
            "post_details": post_details,
        }

    def _respond_node(self, state: AgentsState) -> AgentsState:
        """Build the final response based on the selected post."""
        
        user_query = state["messages"][-1].content

        if not state["post_details"]:
            response_text = (
                "No relevant Hacker News post could be selected for that request. "
                "Try asking a different question or use a more specific topic."
            )
        else:
            post = state["post_details"]
            response_text = self.writer.write(user_query, post)

        messages = [*state["messages"], SystemMessage(content=response_text)]
        return {
            **state,
            "post_details": state["post_details"],
            "response": response_text,
            "messages": messages,
        }

    def invoke(self, message: str, chat_history: list | None = None) -> dict:
        """
        Synchronous invocation of the conversational agent.

        Args:
            message: User message to process.
            chat_history: Optional list of previous messages for context.

        Returns:
            Dictionary with response and updated message history.
        """
        messages = chat_history or []
        messages.append(HumanMessage(content=message))

        initial_state = {
            "messages": messages,
            "candidates": [],
            "selected_post_id": None,
            "keywords": [],
            "post_details": None,
            "response": None,
        }

        result = self.graph.invoke(initial_state)

        # DEBUG
        logger.debug(result)

        return {
            "retrieved": result["candidates"],
            "selected_id": result["selected_post_id"],
            "keywords": result.get("keywords", []),
            "selected_post": result["post_details"],
            "response": result["response"],
            "messages": result["messages"],
        }

    async def ainvoke(self, message: str, chat_history: list | None = None) -> dict:
        """
        Async invoke the agent with a user message.

        Args:
            message: User message to process.
            chat_history: Optional list of previous messages for context.

        Returns:
            Dictionary with response and updated message history.
        """
        messages = chat_history or []
        messages.append(HumanMessage(content=message))

        initial_state = {
            "messages": messages,
            "candidates": [],
            "selected_post_id": None,
            "keywords": [],
            "post_details": None,
            "response": None,
        }

        result = await self.graph.ainvoke(initial_state)

        return {
            "retrieved": result["candidates"],
            "selected_id": result["selected_post_id"],
            "keywords": result.get("keywords", []),
            "selected_post_id": result["post_details"],
            "response": result["response"],
            "messages": result["messages"],
        }

        
