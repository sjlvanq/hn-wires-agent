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
    SearchSimilarByKeywordTool,
)

logger = logging.getLogger(__name__)

class AgentsState(TypedDict):
    """State of the associative conversational agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    candidates: list[dict]
    selected_post_id: int | None
    keywords: list[dict]
    post_details: dict | None
    response: str | None
    skip_agent_response: bool
    last_keywords_ids: list[int] = []


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
        self.search_tool_structured = self.search_tool.as_tool()
        self.keyword_search_tool = SearchSimilarByKeywordTool(self.repository)
        self.keyword_search_tool_structured = self.keyword_search_tool.as_tool()
        self.last_keyword_ids: list[int] = []

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
        messages = state.get("messages") or []
        if not messages:
            logger.error("No messages in state for retrieve node; aborting retrieval.")
            return {**state, "skip_agent_response": True}

        query = messages[-1].content
        try:
            candidates = self.search_tool_structured.run(
                {"query": query, "top_k": settings.wires_vector_search_top_k}
            )
        except Exception:
            logger.exception("Search tool failed in retrieve node")
            candidates = []
        
        return {
            **state,
            "candidates": candidates,
        }

    def _select_node(self, state: AgentsState) -> AgentsState:
        """Select the single most relevant post ID from the retrieved candidates."""
        messages = state.get("messages") or []
        if not messages:
            logger.error("No messages in state for select node; cannot select post.")
            return {**state, "selected_post_id": None, "skip_agent_response": True}
        try:
            selected_post_id = self.selector.select(messages[-1].content, state.get("candidates", []))
        except Exception as e:
            return self._handle_internal_error(
                {**state, "selected_post_id": None},
                e,
                "An error occurred while selecting the post. Please try again later.",
            )
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
        if state.get("skip_agent_response"):
            return {
                **state,
                "response": "",
                "messages": state["messages"],
            }

        messages = state.get("messages") or []
        if not messages:
            logger.error("No messages in state for respond node; nothing to respond to.")
            return {**state, "response": "", "messages": messages, "skip_agent_response": True}

        user_query = messages[-1].content

        if not state["post_details"]:
            response_text = (
                "No relevant Hacker News post could be selected for that request. "
                "Try asking a different question or use a more specific topic."
            )
        else:
            post = state["post_details"]
            try:
                response_text = self.writer.write(user_query, post)
            except Exception as e:
                return self._handle_internal_error(
                    state,
                    e,
                    "An error occurred while generating the response. Please try again later.",
                )

        keywords = state.get("keywords", [])
        if keywords:
            self.last_keyword_ids = [k["id"] for k in keywords]

        messages = [*state["messages"], SystemMessage(content=response_text)]
        return {
            **state,
            "post_details": state["post_details"],
            "response": response_text,
            "messages": messages,
        }

    def _handle_internal_error(self, state: AgentsState, exc: Exception, user_facing_msg: str) -> AgentsState:
        """Uniform internal error handler: log, append a system message and mark skip flag.

        Returns an updated AgentsState suitable to be returned by graph nodes.
        """
        logger.exception("Internal error in NewsAgent: %s", exc)
        messages = list(state.get("messages") or [])
        messages.append(SystemMessage(content=user_facing_msg))
        return {**state, "messages": messages, "response": "", "skip_agent_response": True}

    def invoke(self, message: str, chat_history: list | None = None) -> dict:
        """
        Synchronous invocation of the conversational agent.

        Args:
            message: User message to process.
            chat_history: Optional list of previous messages for context.

        Returns:
            Dictionary with response and updated message history.
        """
        messages = list(chat_history) if chat_history else []
        messages.append(HumanMessage(content=message))

        if message.startswith("/keyword"):
            return self._invoke_explore_keyword_flow(messages)

        return self._invoke_default_flow(messages)

    def _invoke_default_flow(self, messages):
        state = self._build_initial_state(messages)
        result = self.graph.invoke(state)
        return self._format_result(result)

    def _build_initial_state(self, messages) -> AgentsState:
        return {
            "messages": messages,
            "candidates": [],
            "selected_post_id": None,
            "keywords": [],
            "post_details": None,
            "response": None,
            "skip_agent_response": False,
            "last_keywords_ids": [],
        }

    def _format_result(self, result) -> dict:
        return {
            "retrieved": result["candidates"],
            "selected_id": result["selected_post_id"],
            "keywords": result.get("keywords", []),
            "selected_post": result["post_details"],
            "response": result["response"],
            "messages": result["messages"],
        }

    def _invoke_explore_keyword_flow(self, messages: list[BaseMessage]):
        messages = messages or []
        if not messages:
            logger.error("No messages provided to _invoke_explore_keyword_flow")
            return {
                "retrieved": [],
                "selected_id": None,
                "keywords": [],
                "selected_post": None,
                "response": "No message context provided",
                "messages": messages,
            }

        try:
            keyword_id, selection_criteria = self._parse_explore_keyword_command(messages[-1].content)
        except Exception as e:
            logger.exception("Failed to parse /keyword command")
            messages = [*messages, SystemMessage(content="Invalid usage of /keyword. Usage: /keyword <id> [criteria]")]
            return {
                "retrieved": [],
                "selected_id": None,
                "keywords": [],
                "selected_post": None,
                "response": "Invalid command",
                "messages": messages,
            }

        try:
            candidates = self.keyword_search_tool_structured.run({"keyword_id": keyword_id})
        except Exception as e:
            logger.exception("Keyword search failed in explore flow")
            err_state = self._build_explore_state(messages, [], None)
            err_state = self._handle_internal_error(err_state, e, "Error searching by keyword; try again later.")
            return self._format_result(err_state)

        if not candidates:
            state = self._build_explore_state(messages, [], None)
            state = self._handle_internal_error(
                state, 
                ValueError(f"No candidates found for keyword id {keyword_id}."), 
                f"No posts related to keyword {keyword_id} were found."
            )
            return self._format_result(state)

        if selection_criteria is None:
            return self._format_explore_candidates(candidates, messages)

        try:
            selected_post_id = self.selector.select(selection_criteria, candidates)
        except Exception as e:
            err_state = self._build_explore_state(messages, candidates, None)
            err_state = self._handle_internal_error(err_state, e, "An error occurred while selecting the post. Try again later.")
            return self._format_result(err_state)

        state = self._build_explore_state(messages, candidates, selected_post_id)
        state = self._keywords_node(state)
        state = self._fetch_node(state)
        
        self._preserve_keyword_ids(state)
        return self._format_result(state)

    def _parse_explore_keyword_command(self, message: str) -> tuple[int | None, str | None]:
        try:
            keyword_text = message[len("/keyword"):].strip()
            logger.debug(f"Parsing /keyword command: '{keyword_text}'")
            keyword_id = int(keyword_text.split()[0])
            logger.debug(f"Extracted keyword_id: {keyword_id}")
            selection_criteria = " ".join(keyword_text.split()[1:]) or None
            logger.debug(f"Extracted selection_criteria: {selection_criteria}")
            return keyword_id, selection_criteria
        except (ValueError, IndexError):
            raise ValueError("Use /keyword <keyword_id> <optional-criteria>")
        
    def _format_explore_candidates(self, candidates: list, messages: list[BaseMessage]) -> dict:
        """
        Format the candidates for the /keyword command.

        Args:
            candidates: List of candidate posts.
            messages: The current message history.
        """
        return {
            "retrieved": candidates,
            "selected_id": None,
            "keywords": [],
            "selected_post": None,
            "response": None,
            "messages": messages,
        }

    def _build_explore_state(self, messages: list[BaseMessage], candidates: list[dict], selected_post_id: int | None) -> AgentsState:
        """
        Build the state for the /keyword command.

        Args:
            messages: The current message history.
            candidates: List of candidate posts.
            selected_post_id: The ID of the selected post.
        """
        return {
            "messages": messages,
            "candidates": candidates,
            "selected_post_id": selected_post_id,
            "keywords": [],
            "post_details": None,
            "response": None,
        }

    def _preserve_keyword_ids(self, state: AgentsState):
        """Preserve the last keyword IDs for future reference."""
        self.last_keyword_ids = [k["id"] for k in state.get("keywords", [])]

    async def ainvoke(self, message: str, chat_history: list | None = None) -> dict:
        messages = list(chat_history) if chat_history else []
        messages.append(HumanMessage(content=message))

        state = self._build_initial_state(messages)
        result = await self.graph.ainvoke(state)
        return self._format_result(result)

        
