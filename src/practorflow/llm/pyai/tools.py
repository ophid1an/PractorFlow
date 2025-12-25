"""
Pydantic AI tool functions for knowledge search integration.

This module provides tool functions designed to be registered on Pydantic AI Agents
using the @agent.tool decorator pattern. Tools receive dependencies via RunContext.

Usage:
    from dataclasses import dataclass
    from pydantic_ai import Agent, RunContext
    from practorflow.llm.pyai import LocalLLMModel
    from practorflow.llm.pyai.tools import search_knowledge
    from practorflow.llm.knowledge import KnowledgeStore

    @dataclass
    class Deps:
        knowledge_store: KnowledgeStore
        document_scope: set[str] | None = None

    model = LocalLLMModel(runner)
    agent = Agent(model, deps_type=Deps)

    # Register the tool on the agent
    agent.tool(search_knowledge)

    # Run with dependencies
    result = await agent.run("Question?", deps=Deps(knowledge_store=ks))
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from pydantic_ai import RunContext

from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("agent", level=appConfiguration.LoggerConfiguration.AgentLevel)


@dataclass
class KnowledgeDeps:
    """
    Dependencies for knowledge-based tools.

    This dataclass can be used directly as deps_type for simple agents,
    or its fields can be included in a larger Deps dataclass.

    Attributes:
        knowledge_store: The knowledge store instance for document search.
        document_scope: Optional set of document IDs to limit search scope.
    """

    knowledge_store: KnowledgeStore
    document_scope: Optional[Set[str]] = None


def format_search_results(results: List[Dict[str, Any]], query: str) -> str:
    """
    Format search results as a string for agent consumption.

    Args:
        results: List of search result dicts from knowledge store.
        query: The original search query.

    Returns:
        Formatted string with search results.
    """
    parts = []
    parts.append(f'Search results for: "{query}"')
    parts.append(f"Found {len(results)} relevant section(s):\n")

    for idx, result in enumerate(results, 1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        filename = result.get("filename") or metadata.get("filename", "unknown")
        similarity = result.get("similarity", 0.0)

        header = f"--- Section {idx} (Source: {filename}, Relevance: {similarity:.2f}) ---"
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


async def search_knowledge(
    ctx: RunContext[KnowledgeDeps], query: str, top_k: int = 5
) -> str:
    """Search the knowledge base for relevant information.

    Use this tool when you need to find specific facts, details,
    or context from stored documents to answer a question accurately.

    Args:
        ctx: The run context containing dependencies.
        query: The search query to find relevant information in the knowledge base.
        top_k: Maximum number of results to return (default: 5, range: 1-20).

    Returns:
        Formatted search results as a string, or a message if no results found.
    """
    top_k = max(1, min(20, top_k))

    logger.debug(f"[search_knowledge] Searching: '{query}' (top_k={top_k})")

    knowledge_store = ctx.deps.knowledge_store
    document_scope = ctx.deps.document_scope

    if document_scope:
        logger.debug(f"[search_knowledge] Scope: {len(document_scope)} documents")

    try:
        results = knowledge_store.search_scoped(
            query=query, top_k=top_k, document_ids=document_scope
        )

        if not results:
            return "No relevant documents found for the query."

        return format_search_results(results, query)

    except Exception as e:
        logger.error(f"[search_knowledge] Error: {e}")
        return f"Search failed: {str(e)}"


async def search_knowledge_generic(
    ctx: RunContext[Any], query: str, top_k: int = 5
) -> str:
    """Search the knowledge base for relevant information.

    Generic version that works with any Deps that has 'knowledge_store' attribute.

    Use this tool when you need to find specific facts, details,
    or context from stored documents to answer a question accurately.

    Args:
        ctx: The run context containing dependencies with knowledge_store attribute.
        query: The search query to find relevant information in the knowledge base.
        top_k: Maximum number of results to return (default: 5, range: 1-20).

    Returns:
        Formatted search results as a string, or a message if no results found.
    """
    top_k = max(1, min(20, top_k))

    logger.debug(f"[search_knowledge_generic] Searching: '{query}' (top_k={top_k})")

    deps = ctx.deps

    if not hasattr(deps, "knowledge_store"):
        return "Error: Dependencies do not include knowledge_store."

    knowledge_store: KnowledgeStore = deps.knowledge_store
    document_scope = getattr(deps, "document_scope", None)

    if document_scope:
        logger.debug(
            f"[search_knowledge_generic] Scope: {len(document_scope)} documents"
        )

    try:
        results = knowledge_store.search_scoped(
            query=query, top_k=top_k, document_ids=document_scope
        )

        if not results:
            return "No relevant documents found for the query."

        return format_search_results(results, query)

    except Exception as e:
        logger.error(f"[search_knowledge_generic] Error: {e}")
        return f"Search failed: {str(e)}"


def register_knowledge_tools(agent, use_generic: bool = False):
    """
    Register knowledge search tools on an agent.

    This is a convenience function for registering the search tool.

    Args:
        agent: The Pydantic AI Agent instance.
        use_generic: If True, use the generic version that works with any Deps.

    Returns:
        The agent (for chaining).

    Example:
        agent = Agent(model, deps_type=KnowledgeDeps)
        register_knowledge_tools(agent)
    """
    if use_generic:
        agent.tool(search_knowledge_generic)
    else:
        agent.tool(search_knowledge)
    return agent