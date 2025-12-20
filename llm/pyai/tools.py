"""
Pydantic AI tool wrappers for knowledge search integration.

Provides two approaches for RAG integration:
1. Native tool: Expose KnowledgeSearchTool as a Pydantic AI function tool
2. Context injection: Helper for manual search before agent run

Usage (Native tool):
    from llm.pyai.tools import create_knowledge_search_tool
    
    tool = create_knowledge_search_tool(runner)
    agent = Agent(model=model, tools=[tool])
    result = await agent.run("What does the document say about X?")

Usage (Context injection):
    from llm.pyai.tools import inject_search_context
    
    await inject_search_context(runner, "relevant query")
    result = await agent.run("Question about the documents")
"""

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass

from pydantic import BaseModel, Field

from llm.base.llm_runner import LLMRunner
from llm.tools.base import ToolResult


class KnowledgeSearchParams(BaseModel):
    """Parameters for knowledge search tool."""
    query: str = Field(
        description="The search query to find relevant information in the knowledge base"
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=20
    )


@dataclass
class SearchResult:
    """Formatted search result for agent consumption."""
    text: str
    source: str
    relevance: float
    
    def format(self) -> str:
        """Format result for display."""
        return f"[Source: {self.source}, Relevance: {self.relevance:.2f}]\n{self.text}"


def create_knowledge_search_tool(
    runner: LLMRunner,
    name: str = "search_knowledge",
    description: Optional[str] = None,
):
    """
    Create a Pydantic AI compatible tool function for knowledge search.
    
    This creates a function that can be passed to Agent's tools parameter,
    allowing the agent to search the knowledge base during reasoning.
    
    Args:
        runner: LLMRunner with knowledge store configured
        name: Tool name (default: "search_knowledge")
        description: Tool description (uses default if not provided)
        
    Returns:
        Async function compatible with Pydantic AI tools
        
    Example:
        runner = create_runner(handle, knowledge_store)
        search_tool = create_knowledge_search_tool(runner)
        
        agent = Agent(
            model=LocalLLMModel(runner),
            tools=[search_tool],
        )
        
        result = await agent.run("What information do we have about topic X?")
    """
    tool_description = description or (
        "Search the knowledge base for relevant information. "
        "Use this tool when you need to find specific facts, details, "
        "or context from stored documents to answer a question accurately."
    )
    
    async def search_knowledge(query: str, top_k: int = 5) -> str:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: The search query to find relevant information
            top_k: Maximum number of results to return (1-20)
            
        Returns:
            Formatted search results as a string
        """
        # Validate top_k
        top_k = max(1, min(20, top_k))
        
        # Execute search through runner's tool registry
        result = runner.search(query=query, top_k=top_k)
        
        if not result.success:
            return f"Search failed: {result.error}"
        
        if not result.data:
            return "No relevant documents found for the query."
        
        # Format results for the agent
        return _format_search_results(result)
    
    # Set function metadata for Pydantic AI
    search_knowledge.__name__ = name
    search_knowledge.__doc__ = tool_description
    
    return search_knowledge


def create_scoped_knowledge_search_tool(
    runner: LLMRunner,
    document_ids: Set[str],
    name: str = "search_documents",
    description: Optional[str] = None,
):
    """
    Create a knowledge search tool scoped to specific documents.
    
    Useful when you want to limit search to a subset of documents,
    such as those in the current session.
    
    Args:
        runner: LLMRunner with knowledge store configured
        document_ids: Set of document IDs to scope search to
        name: Tool name
        description: Tool description
        
    Returns:
        Async function compatible with Pydantic AI tools
    """
    tool_description = description or (
        "Search the current documents for relevant information. "
        "Use this to find specific details from the loaded documents."
    )
    
    async def search_documents(query: str, top_k: int = 5) -> str:
        """
        Search loaded documents for relevant information.
        
        Args:
            query: The search query
            top_k: Maximum number of results
            
        Returns:
            Formatted search results
        """
        top_k = max(1, min(20, top_k))
        
        # Set document scope before search
        runner.set_document_scope(document_ids)
        
        try:
            result = runner.search(query=query, top_k=top_k)
            
            if not result.success:
                return f"Search failed: {result.error}"
            
            if not result.data:
                return "No relevant information found in the documents."
            
            return _format_search_results(result)
        finally:
            # Clear scope after search
            runner.clear_document_scope()
    
    search_documents.__name__ = name
    search_documents.__doc__ = tool_description
    
    return search_documents


async def inject_search_context(
    runner: LLMRunner,
    query: str,
    top_k: int = 5,
    document_ids: Optional[Set[str]] = None,
) -> ToolResult:
    """
    Inject search context for the next generation call.
    
    This is the manual/context injection approach to RAG.
    Call this before running the agent to pre-populate context.
    
    Args:
        runner: LLMRunner with knowledge store
        query: Search query
        top_k: Number of results
        document_ids: Optional document scope
        
    Returns:
        ToolResult from the search
        
    Example:
        # Inject context before agent run
        result = await inject_search_context(runner, "product specifications")
        
        if result.success:
            # Context is now pending; next generate() will include it
            response = await agent.run("Summarize the product specs")
    """
    # Set scope if provided
    if document_ids:
        runner.set_document_scope(document_ids)
    
    try:
        # Execute search - results stored as pending context
        result = runner.search(query=query, top_k=top_k)
        return result
    finally:
        if document_ids:
            runner.clear_document_scope()


def _format_search_results(result: ToolResult) -> str:
    """
    Format ToolResult data as a string for agent consumption.
    
    Args:
        result: ToolResult from knowledge search
        
    Returns:
        Formatted string with search results
    """
    if isinstance(result.data, str):
        return result.data
    
    # Build formatted output
    parts = []
    metadata = result.metadata or {}
    
    query = metadata.get("query", "")
    count = metadata.get("results_count", 0)
    
    if query:
        parts.append(f"Search results for: \"{query}\"")
    parts.append(f"Found {count} relevant section(s):\n")
    
    # The data should already be formatted by KnowledgeSearchTool
    if isinstance(result.data, str):
        parts.append(result.data)
    else:
        parts.append(str(result.data))
    
    return "\n".join(parts)


class MultiToolRegistry:
    """
    Helper to manage multiple Pydantic AI tools with a runner.
    
    Provides convenient registration and retrieval of tools
    that integrate with the LLM runner's capabilities.
    """
    
    def __init__(self, runner: LLMRunner):
        """
        Initialize tool registry.
        
        Args:
            runner: LLMRunner instance
        """
        self.runner = runner
        self._tools: Dict[str, Any] = {}
    
    def add_knowledge_search(
        self,
        name: str = "search_knowledge",
        description: Optional[str] = None,
        document_ids: Optional[Set[str]] = None,
    ) -> "MultiToolRegistry":
        """
        Add knowledge search tool.
        
        Args:
            name: Tool name
            description: Tool description
            document_ids: Optional document scope
            
        Returns:
            Self for chaining
        """
        if document_ids:
            tool = create_scoped_knowledge_search_tool(
                self.runner, document_ids, name, description
            )
        else:
            tool = create_knowledge_search_tool(
                self.runner, name, description
            )
        
        self._tools[name] = tool
        return self
    
    def add_custom_tool(self, name: str, tool_func) -> "MultiToolRegistry":
        """
        Add a custom tool function.
        
        Args:
            name: Tool name
            tool_func: Async function to use as tool
            
        Returns:
            Self for chaining
        """
        self._tools[name] = tool_func
        return self
    
    def get_tools(self) -> List[Any]:
        """Get all registered tools as a list."""
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def clear(self) -> None:
        """Remove all tools."""
        self._tools.clear()
