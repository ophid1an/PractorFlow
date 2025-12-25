"""
Usage examples for Pydantic AI integration with local LLM runners.

This file demonstrates the correct way to use LocalLLMModel with
Pydantic AI agents, following the official Pydantic AI patterns.

Note: These examples are for reference and documentation purposes.
      Run them in an async context (e.g., with asyncio.run()).
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Set

from pydantic_ai import Agent, RunContext

# Local LLM imports
from practorflow.llm import ModelPool, create_runner
from practorflow.llm.knowledge.chroma_knowledge_store import ChromaKnowledgeStore
from practorflow.llm.pyai import LocalLLMModel, KnowledgeDeps, search_knowledge
from practorflow.llm.tools.base_web_search import DuckDuckGoSearchTool
from practorflow.llm.tools.serpapi_web_search import SerpAPISearchTool
from practorflow.settings.app_settings import appConfiguration

# Use the centralized configuration
config = appConfiguration.ModelConfiguration


# =============================================================================
# Example 1: Basic Usage - Simple Agent (No Tools)
# =============================================================================


async def example_basic_usage():
    """
    Basic example: Create a Pydantic AI agent with local LLM.

    This demonstrates the simplest integration without any tools.
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    pool = ModelPool.get_instance(max_models=1)

    async with pool.acquire_context(config) as handle:
        # Create runner and model
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        # Create Pydantic AI agent - no tools, no deps
        agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant. Be concise.",
        )

        # Run the agent
        result = await agent.run("What is the capital of France?")
        print(f"Response: {result.output}")
        print(f"Usage: {result.usage()}")


# =============================================================================
# Example 2: RAG with Knowledge Search Tool
# =============================================================================


async def example_rag_with_tool():
    """
    RAG example: Agent with knowledge search tool.

    This demonstrates the correct Pydantic AI pattern for RAG:
    - Define a Deps dataclass with knowledge_store
    - Register the search tool on the agent using @agent.tool
    - Pass dependencies when running the agent
    """
    print("=" * 60)
    print("Example 2: RAG with Knowledge Search Tool")
    print("=" * 60)

    # Setup knowledge store using centralized configuration
    knowledge_store = ChromaKnowledgeStore(appConfiguration.KnowledgeChromaConfiguration)

    pool = ModelPool.get_instance(max_models=1)

    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        # Create agent with deps_type
        agent = Agent(
            model=model,
            deps_type=KnowledgeDeps,
            system_prompt=(
                "You are a helpful assistant with access to a knowledge base. "
                "Use the search_knowledge tool to find relevant information "
                "before answering questions about documents."
            ),
        )

        # Register the search tool on the agent
        agent.tool(search_knowledge)

        # Create dependencies
        deps = KnowledgeDeps(knowledge_store=knowledge_store)

        # Run with dependencies
        result = await agent.run(
            "What information is available in the knowledge base?", deps=deps
        )
        print(f"Response: {result.output}")


# =============================================================================
# Example 3: RAG with Scoped Search (Session Documents)
# =============================================================================


@dataclass
class SessionDeps:
    """Dependencies for session-scoped search."""

    knowledge_store: ChromaKnowledgeStore
    document_scope: Optional[Set[str]] = None
    session_id: Optional[str] = None


async def example_scoped_search():
    """
    Scoped search: Limit search to specific documents.

    Useful when working with session-specific documents,
    like files uploaded in a chat session.
    """
    print("=" * 60)
    print("Example 3: Scoped Search")
    print("=" * 60)

    # Setup knowledge store using centralized configuration
    knowledge_store = ChromaKnowledgeStore(appConfiguration.KnowledgeChromaConfiguration)

    # Get actual document IDs from the knowledge store
    documents = knowledge_store.list_documents()
    if not documents:
        print("No documents in knowledge store. Please add documents first.")
        return

    # Use real document IDs from the store (take first 2)
    session_document_ids = {doc["id"] for doc in documents[:2]}
    print(f"Scoping search to documents: {session_document_ids}")

    pool = ModelPool.get_instance()

    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        # Create agent with custom deps
        agent = Agent(
            model=model,
            deps_type=SessionDeps,
            system_prompt="Answer questions using only the uploaded session documents.",
        )

        # Define a scoped search tool that uses SessionDeps
        @agent.tool
        async def search_session_docs(
            ctx: RunContext[SessionDeps], query: str, top_k: int = 5
        ) -> str:
            """Search documents uploaded in this session.

            Args:
                ctx: The run context with session dependencies.
                query: The search query to find relevant information.
                top_k: Maximum number of results to return.
            """
            from practorflow.llm.pyai.tools import format_search_results

            top_k = max(1, min(20, top_k))

            results = ctx.deps.knowledge_store.search_scoped(
                query=query, top_k=top_k, document_ids=ctx.deps.document_scope
            )

            if not results:
                return "No relevant information found in the session documents."

            return format_search_results(results, query)

        # Create deps with document scope
        deps = SessionDeps(
            knowledge_store=knowledge_store,
            document_scope=session_document_ids,
            session_id="session_123",
        )

        result = await agent.run(
            "What content is available in these documents?", deps=deps
        )
        print(f"Response: {result.output}")


# =============================================================================
# Example 4: Streaming Response
# =============================================================================


async def example_streaming():
    """
    Streaming example: Get response chunks as they're generated.
    """
    print("=" * 60)
    print("Example 4: Streaming Response")
    print("=" * 60)

    pool = ModelPool.get_instance()

    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            system_prompt="You are a storyteller. Write engaging content.",
        )

        prompt = "Tell me a short story about a robot learning to paint."
        print(f"Question: {prompt}")
        print("Streaming response:")

        try:
            async with agent.run_stream(prompt) as response:
                async for chunk in response.stream_text():
                    print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\nStreaming error: {e}")
            import traceback

            traceback.print_exc()


# =============================================================================
# Example 5: Conversation with History
# =============================================================================


async def example_conversation():
    """
    Conversation: Multi-turn dialogue with context.

    Shows how to maintain conversation history across turns.
    """
    print("=" * 60)
    print("Example 5: Conversation with History")
    print("=" * 60)

    pool = ModelPool.get_instance()

    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            system_prompt="You are a helpful coding assistant.",
        )

        result1 = await agent.run("How do I read a file in Python?")
        print(f"User: How do I read a file in Python?")
        print(f"Assistant: {result1.output}\n")

        result2 = await agent.run(
            "How do I handle errors in that code?",
            message_history=result1.all_messages(),
        )
        print(f"User: How do I handle errors in that code?")
        print(f"Assistant: {result2.output}\n")


# =============================================================================
# Example 6: Multiple Tools
# =============================================================================


async def example_multiple_tools():
    """
    Multiple tools: Combine knowledge search with custom tools.
    """
    print("=" * 60)
    print("Example 6: Multiple Tools")
    print("=" * 60)

    @dataclass
    class MultiToolDeps:
        knowledge_store: ChromaKnowledgeStore
        document_scope: Optional[Set[str]] = None

    # Setup knowledge store using centralized configuration
    knowledge_store = ChromaKnowledgeStore(appConfiguration.KnowledgeChromaConfiguration)

    pool = ModelPool.get_instance()

    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=MultiToolDeps,
            system_prompt=(
                "You are a helpful assistant with access to a knowledge base "
                "and a calculator. Use the appropriate tool for each task."
            ),
        )

        # Register knowledge search tool
        @agent.tool
        async def search_docs(
            ctx: RunContext[MultiToolDeps], query: str, top_k: int = 5
        ) -> str:
            """Search the document knowledge base.

            Args:
                ctx: The run context.
                query: The search query.
                top_k: Maximum results to return.
            """
            from practorflow.llm.pyai.tools import format_search_results

            results = ctx.deps.knowledge_store.search_scoped(
                query=query, top_k=top_k, document_ids=ctx.deps.document_scope
            )
            if not results:
                return "No relevant documents found."
            return format_search_results(results, query)

        # Register calculator tool
        @agent.tool_plain
        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression safely.

            Args:
                expression: A mathematical expression to evaluate.
            """
            try:
                allowed_names = {
                    "abs": abs,
                    "round": round,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "pow": pow,
                    "int": int,
                    "float": float,
                }
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"

        deps = MultiToolDeps(knowledge_store=knowledge_store)

        result = await agent.run(
            "Calculate the sum of 10, 20, 30, 40, and 50.", deps=deps
        )
        print(f"Response: {result.output}")


# =============================================================================
# Example 7: Web Search with DuckDuckGo (Free)
# =============================================================================


async def example_web_search_duckduckgo():
    """
    Web search example using DuckDuckGo (free, no API key).

    Demonstrates integrating the DuckDuckGo web search tool with a Pydantic AI agent.
    """
    print("=" * 60)
    print("Example 7: Web Search with DuckDuckGo")
    print("=" * 60)

    @dataclass
    class WebSearchDeps:
        web_search_tool: DuckDuckGoSearchTool

    pool = ModelPool.get_instance(max_models=1)

    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=WebSearchDeps,
            system_prompt=(
                "You are a helpful assistant with access to web search. "
                "Use the search_web tool to find current information "
                "when answering questions about recent events or facts."
            ),
        )

        @agent.tool
        async def search_web(
            ctx: RunContext[WebSearchDeps], query: str, max_results: int = 5
        ) -> str:
            """Search the web for current information.

            Args:
                ctx: The run context with dependencies.
                query: The search query to find information on the web.
                max_results: Maximum number of results to return.
            """
            result = ctx.deps.web_search_tool.execute(
                query=query, max_results=max_results
            )

            if result.success and result.data:
                return result.data
            elif result.success:
                return "No results found for the query."
            else:
                return f"Search error: {result.error}"
            
        # Create dependencies
        deps = WebSearchDeps(web_search_tool=DuckDuckGoSearchTool())

        result = await agent.run(
            "What are the latest developments in artificial intelligence?",
            deps=deps,
        )
        print(f"Response: {result.output}")


# =============================================================================
# Example 8: Web Search with SerpAPI (Premium)
# =============================================================================


async def example_web_search_serpapi():
    """
    Web search example using SerpAPI (premium, requires API key).

    Demonstrates integrating the SerpAPI web search tool with a Pydantic AI agent.
    Supports Google, Bing, Yahoo, and other search engines.
    """
    print("=" * 60)
    print("Example 8: Web Search with SerpAPI")
    print("=" * 60)

    import os

    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        print("Skipping: SERPAPI_API_KEY environment variable not set")
        print("Set it with: export SERPAPI_API_KEY='your-api-key'")
        return

    @dataclass
    class SerpAPISearchDeps:
        web_search_tool: SerpAPISearchTool

    pool = ModelPool.get_instance(max_models=1)

    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=SerpAPISearchDeps,
            system_prompt=(
                "You are a helpful assistant with access to Google search. "
                "Use the available search tools to find current information, "
                "news, or images when answering questions."
            ),
        )

        @agent.tool
        async def search_web(
            ctx: RunContext[SerpAPISearchDeps], query: str, max_results: int = 5
        ) -> str:
            """Search Google for current information.

            Args:
                ctx: The run context with dependencies.
                query: The search query to find information on the web.
                max_results: Maximum number of results to return.
            """
            result = ctx.deps.web_search_tool.execute(
                query=query, max_results=max_results
            )

            if result.success and result.data:
                return result.data
            elif result.success:
                return "No results found for the query."
            else:
                return f"Search error: {result.error}"

        @agent.tool
        async def search_news(
            ctx: RunContext[SerpAPISearchDeps], query: str, max_results: int = 5
        ) -> str:
            """Search Google News for recent news articles.

            Args:
                ctx: The run context with dependencies.
                query: The search query to find news articles.
                max_results: Maximum number of results to return.
            """
            result = ctx.deps.web_search_tool.search_news(
                query=query, max_results=max_results
            )

            if result.success and result.data:
                return result.data
            elif result.success:
                return "No news results found for the query."
            else:
                return f"News search error: {result.error}"

        # Create dependencies with SerpAPI tool
        serp_tool = SerpAPISearchTool(
            api_key=api_key,
            engine="google",
            country="us",
            language="en",
        )

        deps = SerpAPISearchDeps(web_search_tool=serp_tool)

        result = await agent.run(
            "What are the top tech news stories today?",
            deps=deps,
        )
        print(f"Response: {result.output}")


# =============================================================================
# Run Examples
# =============================================================================


async def run_all_examples():
    """Run all examples sequentially."""
    examples = [
        ("Basic Usage", example_basic_usage),
        ("RAG with Tool", example_rag_with_tool),
        # ("Scoped Search", example_scoped_search),
        # ("Streaming", example_streaming),
        # ("Conversation", example_conversation),
        # ("Multiple Tools", example_multiple_tools),
        # ("Web Search DuckDuckGo", example_web_search_duckduckgo),
        # ("Web Search SerpAPI", example_web_search_serpapi),
    ]

    for name, example_fn in examples:
        print(f"\n{'#' * 70}")
        print(f"# Running: {name}")
        print(f"{'#' * 70}\n")
        try:
            await example_fn()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback

            traceback.print_exc()
        print()


if __name__ == "__main__":
    # Run a specific example
    # asyncio.run(example_basic_usage())
    # asyncio.run(example_rag_with_tool())

    # Or run all examples
    asyncio.run(run_all_examples())