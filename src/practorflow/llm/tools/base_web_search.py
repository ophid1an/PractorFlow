"""
Web search tool using DuckDuckGo.

Free, no API key required, no rate limits for reasonable usage.
Uses the ddgs library.

Install: pip install ddgs
"""

from typing import Any, Dict, List

from practorflow.llm.tools.base import BaseTool, ToolParameter, ToolResult
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class DuckDuckGoSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo.

    Free, no API key required, no rate limits for reasonable usage.
    """

    def __init__(
        self,
        default_max_results: int = 5,
        region: str = "wt-wt",
        safesearch: str = "moderate",
    ):
        """
        Initialize DuckDuckGo search tool.

        Args:
            default_max_results: Default number of results to return
            region: Region for search results (default: "wt-wt" for worldwide)
                   Examples: "us-en", "uk-en", "de-de"
            safesearch: Safe search setting ("on", "moderate", "off")
        """
        self._default_max_results = default_max_results
        self._region = region
        self._safesearch = safesearch
        self._ddgs = None

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information. "
            "Use this tool when you need to find up-to-date facts, news, "
            "or information that may not be in your training data."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query to find information on the web",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description=f"Maximum number of results to return (default: {self._default_max_results})",
                required=False,
                default=self._default_max_results,
            ),
        ]

    def _get_client(self):
        """Lazy initialization of DuckDuckGo client."""
        if self._ddgs is None:
            try:
                from ddgs import DDGS
                self._ddgs = DDGS()
            except ImportError:
                raise ImportError(
                    "ddgs is required for DuckDuckGoSearchTool. "
                    "Install with: pip install ddgs"
                )
        return self._ddgs

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute web search.

        Args:
            query: Search query string
            max_results: Maximum number of results (optional)

        Returns:
            ToolResult with search results or error
        """
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", self._default_max_results)

        if not query:
            return ToolResult(success=False, error="Query parameter is required")

        try:
            logger.debug(f"[DuckDuckGo] Searching: '{query}' (max_results={max_results})")

            client = self._get_client()

            raw_results = client.text(
                query,
                region=self._region,
                safesearch=self._safesearch,
                max_results=max_results,
            )

            results = []
            for item in raw_results:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                })

            logger.debug(f"[DuckDuckGo] Found {len(results)} results for '{query}'")

            if not results:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={
                        "query": query,
                        "results_count": 0,
                        "message": "No results found",
                        "provider": "duckduckgo",
                    },
                )

            formatted_results = self._format_results(results)

            return ToolResult(
                success=True,
                data=formatted_results,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "provider": "duckduckgo",
                },
            )

        except Exception as e:
            logger.error(f"[DuckDuckGo] Error: {e}")
            return ToolResult(success=False, error=f"Web search failed: {str(e)}")

    def search_news(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Search for news articles.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            ToolResult with news results
        """
        try:
            logger.debug(f"[DuckDuckGo] News search: '{query}' (max_results={max_results})")

            client = self._get_client()

            raw_results = client.news(
                query,
                region=self._region,
                safesearch=self._safesearch,
                max_results=max_results,
            )

            results = []
            for item in raw_results:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("body", ""),
                    "date": item.get("date", ""),
                    "source": item.get("source", ""),
                })

            if not results:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={
                        "query": query,
                        "results_count": 0,
                        "message": "No news results found",
                        "provider": "duckduckgo",
                    },
                )

            formatted = self._format_news_results(results)

            return ToolResult(
                success=True,
                data=formatted,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "provider": "duckduckgo",
                    "type": "news",
                },
            )

        except Exception as e:
            logger.error(f"[DuckDuckGo] News search error: {e}")
            return ToolResult(success=False, error=f"News search failed: {str(e)}")

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as context string for LLM.

        Args:
            results: List of search result dicts

        Returns:
            Formatted string for LLM context
        """
        context_parts = []

        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("snippet", "")

            header = f"--- Result {idx}: {title} ---"
            body = f"URL: {url}\n{snippet}"
            context_parts.append(f"{header}\n{body}")

        return "\n\n".join(context_parts)

    def _format_news_results(self, results: List[Dict[str, Any]]) -> str:
        """Format news results for LLM context."""
        context_parts = []

        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            date = result.get("date", "")
            source = result.get("source", "")

            header = f"--- News {idx}: {title} ---"
            meta = f"Source: {source} | Date: {date}" if source or date else ""
            body = f"URL: {url}\n{meta}\n{snippet}" if meta else f"URL: {url}\n{snippet}"
            context_parts.append(f"{header}\n{body}")

        return "\n\n".join(context_parts)