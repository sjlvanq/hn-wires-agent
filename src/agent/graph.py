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
    last_retrieved_posts_ids: list[int] = []


class NewsAgent:
    """NewsAgent
    ---------
    A :class:`LangGraph`‑based state machine that turns a natural‑language
    query into an article selection and a conversational response.
    """

    def __init__(
        self,
        llm: OllamaLLM | None = None,
        selector_llm: OllamaLLM | None = None,
        writer_llm: OllamaLLM | None = None,
        repository: NewsRepository | None = None,
        embeddings: OllamaEmbeddings | None = None,
    ):
        """Create a new :class:`NewsAgent` instance.

        Parameters
        ----------
        llm : OllamaLLM, optional
            Primary language model used for the main conversational flow.
        selector_llm : OllamaLLM, optional
            Language model for the selector sub‑agent.
        writer_llm : OllamaLLM, optional
            Language model for the writer sub‑agent.
        repository : NewsRepository, optional
            Data access layer.  If *None*, a default :class:`NewsRepository`
            is instantiated.
        embeddings : OllamaEmbeddings, optional
            Embedding function for semantic queries.  Defaults to a new
            :class:`OllamaEmbeddings` instance.
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
        self.last_retrieved_posts_ids: list[int] = []

        # Build the graph
        self.graph = self._build_graph()

    # --- Graph Configuration ---

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

    # --- Public Interface ---

    def invoke(self, message: str, chat_history: list | None = None) -> dict:
        """Synchronously invoke the conversational agent.

        Parameters
        ----------
        message : str
            The user query.
        chat_history : list | None, optional
            Past conversation history to provide context to the model.

        Returns
        -------
        dict
            Result dictionary containing ``response`` and updated message
            history.
        """
        messages = list(chat_history) if chat_history else []
        messages.append(HumanMessage(content=message))

        if message.startswith("/keyword"):
            return self._invoke_explore_keyword_flow(messages)
        elif message.startswith("/expand"):
            return self._invoke_expand_post_flow(messages)
        elif message.startswith("/exclude"):
            return self._invoke_exclude_post_flow(messages)
        elif message.startswith("/"):
            messages = [*messages, SystemMessage(content="Unknown command. Use /keyword <id> [criteria] to explore related posts.")]
            return {
                "retrieved": [],
                "selected_id": None,
                "keywords": [],
                "selected_post": None,
                "response": "Unknown command",
                "messages": messages,
                "skip_retrieved": True,
            }

        return self._invoke_default_flow(messages)

    async def ainvoke(self, message: str, chat_history: list | None = None) -> dict:
        messages = list(chat_history) if chat_history else []
        messages.append(HumanMessage(content=message))

        state = self._build_initial_state(messages)
        result = await self.graph.ainvoke(state)
        return self._format_result(result)

    # --- Execution Flows ---

    def _invoke_default_flow(self, messages):
        """Driver for the standard conversational flow.

        Parameters
        ----------
        messages : list
            List of :class:`BaseMessage` objects representing the chat
            history.

        Returns
        -------
        dict
            Formatted result dictionary.
        """
        state = self._build_initial_state(messages)
        result = self.graph.invoke(state)
        return self._format_result(result)

    def _invoke_explore_keyword_flow(self, messages: list[BaseMessage]):
        """Process the special ``/keyword`` command.

        Parameters
        ----------
        messages : list[BaseMessage]
            List of chat messages; the last message is expected to contain
            the ``/keyword`` command.

        Returns
        -------
        dict
            Normalized command output.
        """
        messages, err = self._ensure_messages(messages)
        if err:
            return err

        try:
            command, keyword_id, selection_criteria = self._parse_command_with_id_and_criteria(messages[-1].content)
        except Exception as e:
            #logger.exception("Failed to parse /keyword command")
            return self._command_error_response(messages, str(e))

        if keyword_id not in self.last_keyword_ids:
            invalid_keyword = f"Keyword ID {keyword_id} is not in the last retrieved keywords. Use /keyword with a valid ID from the last response."
            return self._command_error_response(messages, invalid_keyword)

        # Paranoic check: the user might have provided an ID that was in the last response but has since been deleted.
        if not self.repository.keyword_exists(keyword_id):
            invalid_keyword = f"Keyword ID {keyword_id} does not exist."
            return self._command_error_response(messages, invalid_keyword)

        try:
            candidates = self.keyword_search_tool_structured.run({
                "keyword_id": keyword_id,
                "top_k": settings.wires_vector_search_top_k
            })
        except Exception as e:
            logger.exception("Keyword search failed in explore flow")
            return self._handle_internal_error(self._build_command_state(messages, [], None), e, "Error searching by keyword; try again later.")

        if not candidates:
            state = self._build_command_state(messages, [], None)
            state = self._handle_internal_error(
                state, 
                ValueError(f"No candidates found for keyword id {keyword_id}."), 
                f"No posts related to keyword {keyword_id} were found."
            )
            return self._format_result(state)

        state = self._build_command_state(messages, candidates, None)
        self._preserve_posts_ids(state)

        if selection_criteria is None:
            return self._format_explore_candidates(candidates, messages)

        try:
            selected_post_id = self.selector.select(selection_criteria, candidates)
        except Exception as e:
            err_state = self._build_command_state(messages, candidates, None)
            err_state = self._handle_internal_error(err_state, e, "An error occurred while selecting the post. Try again later.")
            return self._format_result(err_state)

        state = self._build_command_state(messages, candidates, selected_post_id)
        state = self._keywords_node(state)
        state = self._fetch_node(state)
        
        self._preserve_keyword_ids(state)

        return self._format_result(state)

    def _invoke_expand_post_flow(self, messages: list[BaseMessage]):
        """Process the special ``/expand`` command.

        Parameters
        ----------
        messages : list[BaseMessage]
            List of chat messages; the last message is expected to contain
            the ``/expand`` command.

        Returns
        -------
        dict
            Normalized command output.
        """
        messages, err = self._ensure_messages(messages)
        if err:
            return err

        try:
            command, post_id = self._parse_command_with_id(messages[-1].content)
        except Exception as e:
            #invalid_usage = f"Invalid usage of {command}. Usage: {command} <post_id>"
            return self._command_error_response(messages, str(e))

        if post_id not in self.last_retrieved_posts_ids:
            invalid_post_id = f"Post ID {post_id} is not in the last retrieved posts. Use /expand with a valid ID from the last response."
            return self._command_error_response(messages, invalid_post_id)

        # Paranoic check: the user might have provided an ID that was in the last response but has since been deleted.
        if not self.repository.post_exists(post_id):
            invalid_post = f"Post ID {post_id} does not exist."
            return self._command_error_response(messages, invalid_post)

        state = self._build_command_state(messages, [], post_id)
        state = self._fetch_node(state)
        state = self._keywords_node(state)

        self._preserve_keyword_ids(state)

        return self._format_result({**state, "skip_retrieved": True})

    def _invoke_exclude_post_flow(self, messages: list[BaseMessage]):
        """Process the special ``/exclude`` command.

        Parameters
        ----------
        messages : list[BaseMessage]
            List of chat messages; the last message is expected to contain
            the ``/exclude`` command.

        Returns
        -------
        dict
            Normalized command output.
        """
        messages, err = self._ensure_messages(messages)
        if err:
            return err

        try:
            command, post_id = self._parse_command_with_id(messages[-1].content)
        except Exception as e:
            return self._command_error_response(messages, str(e))

        if post_id not in self.last_retrieved_posts_ids:
            invalid_post_id = f"Post ID {post_id} is not in the last retrieved posts. Use /exclude with a valid ID from the last response."
            return self._command_error_response(messages, invalid_post_id)

        # Paranoic check: the user might have provided an ID that was in the last response but has since been deleted.
        if not self.repository.post_exists(post_id):
            invalid_post = f"Post ID {post_id} does not exist."
            return self._command_error_response(messages, invalid_post)

        try:
            self.repository.exclude_post(post_id)
        except Exception as e:
            logger.exception("Failed to exclude post")
            return self._handle_internal_error(
                self._build_command_state(messages, [], None),
                e,
                "An error occurred while excluding the post. Please try again later.",
            )

        exclude_message = f"Post ID {post_id} has been excluded from future searches."
        messages.append(SystemMessage(content=exclude_message))
        return {
            "retrieved": [],
            "selected_id": None,
            "keywords": [],
            "selected_post": None,
            "response": exclude_message,
            "messages": messages,
            "skip_retrieved": True,
        }

    # --- Graph Nodes ---

    def _retrieve_node(self, state: AgentsState) -> AgentsState:
        """Retrieve candidate posts using semantic search.

        Parameters
        ----------
        state : AgentsState
            Current graph state.

        Returns
        -------
        AgentsState
            Updated state with a ``candidates`` key containing search
            results.
        """
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
        """Select the single most relevant post ID from the retrieved candidates.

        Parameters
        ----------
        state : AgentsState
            Current graph state.

        Returns
        -------
        AgentsState
            Updated state containing ``selected_post_id``.
        """
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
        """Extract keywords from the selected post.

        Parameters
        ----------
        state : AgentsState
            Current graph state.

        Returns
        -------
        AgentsState
            Updated state with a ``keywords`` list.
        """
        keywords = []

        if state["selected_post_id"] is not None:
            keywords = self.repository.get_post_keywords(state["selected_post_id"])

        logger.debug(f"""keywords: {keywords}""")

        return {
            **state,
            "keywords": keywords,
        }

    def _fetch_node(self, state: AgentsState) -> AgentsState:
        """Fetch the full post details for the selected post ID.

        Parameters
        ----------
        state : AgentsState
            Current graph state.

        Returns
        -------
        AgentsState
            Updated state with ``post_details``.
        """

        post_details = None
        if state["selected_post_id"] is not None:
            post_details = self.repository.get_post_with_wire(state["selected_post_id"])

        logger.debug(f"""post_details: {post_details}""")

        return {
            **state,
            "post_details": post_details,
        }

    def _respond_node(self, state: AgentsState) -> AgentsState:
        """Build the final response based on the selected post.

        Parameters
        ----------
        state : AgentsState
            Current graph state.

        Returns
        -------
        AgentsState
            Updated state with the generated ``response`` and updated message
            history.
        """
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
            try:
                response_text = self._generate_writer_response(user_query, state["post_details"])
            except Exception as e:
                return self._handle_internal_error(state, e,
                    "An error occurred while generating the response. Please try again later.",
                )

        keywords = state.get("keywords", [])
        if keywords:
            self.last_keyword_ids = [k["id"] for k in keywords]

        candidates = state.get("candidates", [])
        if candidates:
            self.last_retrieved_posts_ids = [k["id"] for k in candidates]

        messages = [*state["messages"], SystemMessage(content=response_text)]
        return {
            **state,
            "post_details": state["post_details"],
            "response": response_text,
            "messages": messages,
        }

    # --- State & Formatting ---

    def _build_initial_state(self, messages) -> AgentsState:
        """Create the initial graph state for a new chat turn.

        Parameters
        ----------
        messages : list
            Current message history.

        Returns
        -------
        AgentsState
            Dictionary with all state keys initialized.
        """
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

    def _build_command_state(self, messages: list[BaseMessage], candidates: list[dict], selected_post_id: int | None) -> AgentsState:
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
            "skip_agent_response": False,
            "last_keywords_ids": self.last_keyword_ids,
        }

    def _format_result(self, result) -> dict:
        """Normalize the graph output to the API contract.

        Parameters
        ----------
        result : dict
            Raw output from the :class:`StateGraph`.

        Returns
        -------
        dict
            Normalized dictionary with keys ``retrieved``, ``selected_id``,
            ``keywords``, ``selected_post``, ``response``, ``messages`` and 
            ``skip_retrieved``.
        """
        return {
            "retrieved": result["candidates"],
            "selected_id": result["selected_post_id"],
            "keywords": result.get("keywords", []),
            "selected_post": result["post_details"],
            "response": result["response"],
            "messages": result["messages"],
            "skip_retrieved": result.get("skip_retrieved", False),
        }

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

    # --- Helpers & Utilities ---

    def _parse_command_with_id(self, message: str) -> tuple[str, int]:
        """Parse a command with an ID.

        Parameters
        ----------
        message : str
            Raw user message containing the command.

        Returns
        -------
        tuple[str, int]
            The command and the ID.
        """
        try:
            parts = message.strip().split(maxsplit=1)
            command = parts[0]
            id_part = int(parts[1])
            return command, id_part
        except IndexError:
            raise ValueError(f"Invalid command format. Use {command} <id>")
        except ValueError:
            raise ValueError(f"Invalid ID format. Use {command} <id>")

    def _parse_command_with_id_and_criteria(self, message: str) -> tuple[str, int, str | None]:
        """Parse a command with an ID and optional criteria.

        Parameters
        ----------
        message : str
            Raw user message containing the command.

        Returns
        -------
        tuple[str, int, str | None]
            The command, the ID, and the optional selection criteria.
        """
        try:
            parts = message.strip().split(maxsplit=2)
            command = parts[0]
            id_part = int(parts[1])
            criteria = parts[2] if len(parts) > 2 else None
            return command, id_part, criteria
        except IndexError:
            raise ValueError(f"Invalid command format. Use {command} <id> <optional-criteria>")
        except ValueError:
            raise ValueError(f"Invalid ID format. Use {command} <id> <optional-criteria>")

    def _ensure_messages(self, messages):
        if not messages:
            return None, self._command_error_response(messages, "No message context provided")
        return messages, None

    def _preserve_keyword_ids(self, state: AgentsState):
        """Preserve the last keyword IDs for future reference."""
        self.last_keyword_ids = [k["id"] for k in state.get("keywords", [])]

    def _preserve_posts_ids(self, state: AgentsState):
        """Preserve the last post IDs for future reference."""
        if state.get("candidates"):
            self.last_retrieved_posts_ids = [post["id"] for post in state["candidates"]]

    def _generate_writer_response(self, query: str, post: dict) -> str:
        """Generate a response using the writer agent."""
        try:
            return self.writer.write(query, post)
        except Exception as e:
            raise e

    def _command_error_response(self, messages, response_text):
        messages = [*messages, SystemMessage(content=response_text)]
        return {
            "retrieved": [],
            "selected_id": None,
            "keywords": [],
            "selected_post": None,
            "response": response_text,
            "messages": messages,
            "skip_retrieved": True,
        }

    def _handle_internal_error(self, state: AgentsState, exc: Exception, user_facing_msg: str) -> AgentsState:
        """Handle an unexpected exception in a node.

        Parameters
        ----------
        state : AgentsState
            The state before the error.
        exc : Exception
            The exception that was raised.
        user_facing_msg : str
            User‑friendly message to include as a system prompt.

        Returns
        -------
        AgentsState
            Updated state containing the system message, an empty ``response``
            field, and the ``skip_agent_response`` flag set to ``True``.
        """
        logger.exception("Internal error in NewsAgent: %s", exc)
        messages = list(state.get("messages") or [])
        messages.append(SystemMessage(content=user_facing_msg))
        return {**state, "messages": messages, "response": "", "skip_agent_response": True}
