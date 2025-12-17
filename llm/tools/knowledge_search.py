"""
Knowledge search tool for RAG retrieval.

Implements Small-to-Big retrieval over scoped documents.
"""

from typing import Any, Dict, List, Optional

from llm.tools.base import BaseTool, ToolParameter, ToolResult
from llm.knowledge.knowledge_store import KnowledgeStore
from logger.logger import get_logger
from settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class KnowledgeSearchTool(BaseTool):
    """
    Tool for searching the knowledge base.

    Performs semantic search over documents using Small-to-Big retrieval:
    - Searches small chunks for better similarity matching
    - Returns parent (context) chunks for richer LLM context
    """

    def __init__(self, knowledge_store: KnowledgeStore, default_top_k: int = 5):
        """
        Initialize knowledge search tool.

        Args:
            knowledge_store: KnowledgeStore instance to search
            default_top_k: Default number of results to return
        """
        self._knowledge_store = knowledge_store
        self._default_top_k = default_top_k
        self._document_scope: Optional[set] = None

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return (
            "Search the knowledge base for relevant information. "
            "Use this tool when you need to find specific facts, details, "
            "or context from stored documents to answer a question."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query to find relevant information",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Maximum number of results to return (default: 5)",
                required=False,
                default=self._default_top_k,
            ),
        ]

    def set_document_scope(self, document_ids: Optional[set]) -> None:
        """
        Set document scope for search.

        Args:
            document_ids: Set of document IDs to limit search to,
                         or None to search all documents
        """
        self._document_scope = document_ids

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute knowledge search.

        Args:
            query: Search query string
            top_k: Maximum number of results (optional)

        Returns:
            ToolResult with search results or error
        """
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", self._default_top_k)

        if not query:
            return ToolResult(success=False, error="Query parameter is required")

        try:
            logger.debug(f"[KnowledgeSearch] Searching: '{query}' (top_k={top_k})")

            if self._document_scope:
                logger.debug(
                    f"[KnowledgeSearch] Scope: {len(self._document_scope)} documents"
                )

            # Perform scoped search
            results = self._knowledge_store.search_scoped(
                query=query, top_k=top_k, document_ids=self._document_scope
            )

            if not results:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={
                        "query": query,
                        "results_count": 0,
                        "message": "No relevant documents found",
                    },
                )

            # Format results for LLM context
            formatted_results = self._format_results(results)

            return ToolResult(
                success=True,
                data=formatted_results,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "document_ids": list(
                        {r.get("document_id") for r in results if r.get("document_id")}
                    ),
                },
            )

        except Exception as e:
            logger.error(f"[KnowledgeSearch] Error: {e}")
            return ToolResult(success=False, error=f"Search failed: {str(e)}")

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as context string for LLM.

        Args:
            results: List of search result dicts from knowledge store

        Returns:
            Formatted string for LLM context
        """
        context_parts = []

        for idx, result in enumerate(results, 1):
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            similarity = result.get("similarity", 0.0)

            header = f"--- Section {idx} (Source: {filename}, Relevance: {similarity:.2f}) ---"
            context_parts.append(f"{header}\n{text}")

        return "\n\n".join(context_parts)
