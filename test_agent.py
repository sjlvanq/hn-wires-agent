#!/usr/bin/env python3
"""Test script for the Hacker News Wires Agent."""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import NewsAgent
from config.settings import settings
from database import NewsRepository
from embeddings import OllamaEmbeddings
from models import OllamaLLM
from tools import SearchSimilarTool

# Load environment variables
load_dotenv()


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    print(f"Ollama Base URL: {settings.ollama_base_url}")
    print(f"LLM Model: {settings.orchestrator_llm_model}")
    print(f"Embedding Model: {settings.ollama_embedding_model}")
    print(f"Database Path: {settings.database_path_absolute}")
    print("✓ Configuration loaded successfully\n")


def test_database():
    """Test database connection and basic queries."""
    print("Testing database connection...")
    try:
        repository = NewsRepository()
        # Test vector search
        test_embedding = [0.0] * 384  # Dummy embedding for testing
        vector_results = repository.wires_vector_search(test_embedding, top_k=3)
        print(f"✓ Vector search found {len(vector_results)} results")

        if vector_results:
            print(f"Sample post: {vector_results[0]['title'][:50]}...")

        print("✓ Database connection successful\n")
    except Exception as e:
        print(f"✗ Database error: {e}\n")


def test_embeddings():
    """Test embedding generation."""
    print("Testing embeddings...")
    try:
        embeddings = OllamaEmbeddings()
        test_text = "This is a test for embedding generation."
        embedding = embeddings.embed_query(test_text)
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
        print("✓ Embeddings working\n")
    except Exception as e:
        print(f"✗ Embeddings error: {e}\n")


def test_llm():
    """Test LLM connection."""
    print("Testing LLM connection...")
    try:
        llm = OllamaLLM()
        response = llm.invoke("Say hello in one word.")
        print(f"✓ LLM response: {response}")
        print("✓ LLM working\n")
    except Exception as e:
        print(f"✗ LLM error: {e}\n")


def test_tools():
    """Test individual tools."""
    print("Testing tools...")
    try:
        repository = NewsRepository()
        embeddings = OllamaEmbeddings()

        # Test similarity search
        similar_tool = SearchSimilarTool(repository, embeddings)
        result = similar_tool.search("machine learning", top_k=2)
        print(f"✓ Similarity search tool working")

        print("✓ All tools working\n")
    except Exception as e:
        print(f"✗ Tools error: {e}\n")


def test_agent():
    """Test the full agent."""
    print("Testing agent...")
    try:
        agent = NewsAgent()
        result = agent.invoke("Show me recent posts about programming")
        print(f"✓ Agent response received")
        print(f"Response: {result['response'][:100]}...")
        print("✓ Agent working\n")
    except Exception as e:
        print(f"✗ Agent error: {e}\n")

def test_tool_calling():
    """Test if the model supports tool calling."""
    print("Testing tool calling support...")
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.tools import StructuredTool

        llm = ChatOllama(model=settings.ollama_llm_model, base_url=settings.ollama_base_url)

        # Create a simple test tool
        def test_func(query: str) -> str:
            return f"Search results for: {query}"

        test_tool = StructuredTool.from_function(
            func=test_func,
            name="search",
            description="Search for content"
        )

        llm_with_tools = llm.bind_tools([test_tool])
        response = llm_with_tools.invoke("Search for python news")

        print(f"Response: {response.content}")
        print(f"Tool calls: {getattr(response, 'tool_calls', None)}")

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("✓ Model supports tool calling")
        else:
            print("✗ Model does not support tool calling")

    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Hacker News Wires Agent - Test Suite")
    print("=" * 50)
    print()

    test_configuration()
    test_database()
    test_embeddings()
    test_llm()
    test_tools()
    test_agent()
    test_tool_calling()

    print("=" * 50)
    print("Test suite completed")
    print("=" * 50)


if __name__ == "__main__":
    main()
