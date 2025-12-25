"""
Web search tool using SerpAPI.

Premium option with access to Google, Bing, Yahoo, and other search engines.
Provides structured, reliable results with extensive customization options.

Free tier: 100 searches/month
Install: pip install google-search-results
"""

from typing import Any, Dict, List, Optional

from practorflow.llm.tools.base import BaseTool, ToolParameter, ToolResult
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class SerpAPISearchTool(BaseTool):
    """
    Web search tool using SerpAPI.

    Premium option providing access to multiple search engines (Google, Bing, etc.)
    with structured, reliable results.

    Free tier: 100 searches/month
    """

    def __init__(
        self,
        api_key: str,
        default_max_results: int = 5,
        engine: str = "google",
        country: str = "us",
        language: str = "en",
        safe_search: bool = True,
    ):
        """
        Initialize SerpAPI search tool.

        Args:
            api_key: SerpAPI API key
            default_max_results: Default number of results to return
            engine: Search engine to use ("google", "bing", "yahoo", "duckduckgo", "baidu", "yandex")
            country: Country code for localized results (e.g., "us", "uk", "de")
            language: Language code (e.g., "en", "es", "de")
            safe_search: Enable safe search filtering
        """
        self._api_key = api_key
        self._default_max_results = default_max_results
        self._engine = engine
        self._country = country
        self._language = language
        self._safe_search = safe_search
        self._client = None

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information using Google and other search engines. "
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
        """Lazy initialization of SerpAPI client."""
        if self._client is None:
            try:
                import serpapi
                self._client = serpapi.Client
            except ImportError:
                raise ImportError(
                    "serpapi is required for SerpAPISearchTool. "
                    "Install with: pip install serpapi"
                )
        return self._client

    def _build_params(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Build search parameters based on engine type.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            Dictionary of search parameters
        """
        base_params = {
            "q": query,
            "num": max_results,
        }

        if self._engine == "google":
            base_params.update({
                "engine": "google",
                "gl": self._country,
                "hl": self._language,
                "safe": "active" if self._safe_search else "off",
            })
        elif self._engine == "bing":
            base_params.update({
                "engine": "bing",
                "cc": self._country.upper(),
                "setLang": self._language,
                "safeSearch": "Strict" if self._safe_search else "Off",
            })
        elif self._engine == "yahoo":
            base_params.update({
                "engine": "yahoo",
                "vl": f"lang_{self._language}",
            })
        elif self._engine == "duckduckgo":
            base_params.update({
                "engine": "duckduckgo",
                "kl": f"{self._country}-{self._language}",
            })
        elif self._engine == "baidu":
            base_params.update({
                "engine": "baidu",
            })
        elif self._engine == "yandex":
            base_params.update({
                "engine": "yandex",
                "lang": self._language,
            })
        else:
            base_params["engine"] = self._engine

        return base_params

    def _extract_results(self, response: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """
        Extract results from SerpAPI response based on engine.

        Args:
            response: Raw SerpAPI response
            max_results: Maximum results to extract

        Returns:
            List of normalized result dicts
        """
        results = []

        # Google organic results
        organic_results = response.get("organic_results", [])
        for item in organic_results[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", 0),
                "displayed_link": item.get("displayed_link", ""),
            })

        # If no organic results, try other result types
        if not results:
            # Bing web results
            web_results = response.get("organic_results", []) or response.get("web_results", [])
            for item in web_results[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", item.get("url", "")),
                    "snippet": item.get("snippet", item.get("description", "")),
                    "position": item.get("position", 0),
                })

        return results

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
            logger.debug(f"[SerpAPI] Searching ({self._engine}): '{query}' (max_results={max_results})")

            client_class = self._get_client()
            params = self._build_params(query, max_results)

            client = client_class(api_key=self._api_key)
            response = client.search(params)

            # Check for errors in response
            if "error" in response:
                return ToolResult(
                    success=False,
                    error=f"SerpAPI error: {response['error']}",
                )

            results = self._extract_results(response, max_results)

            logger.debug(f"[SerpAPI] Found {len(results)} results for '{query}'")

            if not results:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={
                        "query": query,
                        "results_count": 0,
                        "message": "No results found",
                        "provider": "serpapi",
                        "engine": self._engine,
                    },
                )

            formatted_results = self._format_results(results)

            # Extract additional metadata from response
            search_metadata = response.get("search_metadata", {})
            search_info = response.get("search_information", {})

            return ToolResult(
                success=True,
                data=formatted_results,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "provider": "serpapi",
                    "engine": self._engine,
                    "total_results": search_info.get("total_results"),
                    "search_time": search_info.get("time_taken_displayed"),
                    "search_id": search_metadata.get("id"),
                },
            )

        except Exception as e:
            logger.error(f"[SerpAPI] Error: {e}")
            return ToolResult(success=False, error=f"Web search failed: {str(e)}")

    def search_news(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Search for news articles using Google News.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            ToolResult with news results
        """
        try:
            logger.debug(f"[SerpAPI] News search: '{query}' (max_results={max_results})")

            client_class = self._get_client()

            params = {
                "q": query,
                "engine": "google_news",
                "gl": self._country,
                "hl": self._language,
            }

            client = client_class(api_key=self._api_key)
            response = client.search(params)

            if "error" in response:
                return ToolResult(
                    success=False,
                    error=f"SerpAPI error: {response['error']}",
                )

            news_results = response.get("news_results", [])

            results = []
            for item in news_results[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "thumbnail": item.get("thumbnail", ""),
                })

            if not results:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={
                        "query": query,
                        "results_count": 0,
                        "message": "No news results found",
                        "provider": "serpapi",
                        "type": "news",
                    },
                )

            formatted = self._format_news_results(results)

            return ToolResult(
                success=True,
                data=formatted,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "provider": "serpapi",
                    "type": "news",
                },
            )

        except Exception as e:
            logger.error(f"[SerpAPI] News search error: {e}")
            return ToolResult(success=False, error=f"News search failed: {str(e)}")

    def search_images(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Search for images using Google Images.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            ToolResult with image results
        """
        try:
            logger.debug(f"[SerpAPI] Image search: '{query}' (max_results={max_results})")

            client_class = self._get_client()

            params = {
                "q": query,
                "engine": "google_images",
                "gl": self._country,
                "hl": self._language,
                "safe": "active" if self._safe_search else "off",
                "num": max_results,
            }

            client = client_class(api_key=self._api_key)
            response = client.search(params)

            if "error" in response:
                return ToolResult(
                    success=False,
                    error=f"SerpAPI error: {response['error']}",
                )

            image_results = response.get("images_results", [])

            results = []
            for item in image_results[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "image_url": item.get("original", ""),
                    "thumbnail": item.get("thumbnail", ""),
                    "source": item.get("source", ""),
                    "width": item.get("original_width"),
                    "height": item.get("original_height"),
                })

            if not results:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={
                        "query": query,
                        "results_count": 0,
                        "message": "No image results found",
                        "provider": "serpapi",
                        "type": "images",
                    },
                )

            formatted = self._format_image_results(results)

            return ToolResult(
                success=True,
                data=formatted,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "provider": "serpapi",
                    "type": "images",
                },
            )

        except Exception as e:
            logger.error(f"[SerpAPI] Image search error: {e}")
            return ToolResult(success=False, error=f"Image search failed: {str(e)}")

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
            position = result.get("position", idx)

            header = f"--- Result {position}: {title} ---"
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

    def _format_image_results(self, results: List[Dict[str, Any]]) -> str:
        """Format image results for LLM context."""
        context_parts = []

        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            image_url = result.get("image_url", "")
            source = result.get("source", "")
            width = result.get("width", "")
            height = result.get("height", "")

            header = f"--- Image {idx}: {title} ---"
            meta_parts = []
            if source:
                meta_parts.append(f"Source: {source}")
            if width and height:
                meta_parts.append(f"Size: {width}x{height}")
            meta = " | ".join(meta_parts) if meta_parts else ""

            body_parts = [f"Page URL: {url}", f"Image URL: {image_url}"]
            if meta:
                body_parts.insert(0, meta)
            body = "\n".join(body_parts)

            context_parts.append(f"{header}\n{body}")

        return "\n\n".join(context_parts)

    def set_engine(self, engine: str) -> None:
        """
        Change the search engine.

        Args:
            engine: Search engine ("google", "bing", "yahoo", "duckduckgo", "baidu", "yandex")
        """
        self._engine = engine
        logger.debug(f"[SerpAPI] Engine changed to: {engine}")