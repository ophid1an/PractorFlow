from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, AsyncIterator

from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.pool.model_handle import ModelHandle
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.tools.tool_registry import ToolRegistry
from practorflow.llm.tools.base import ToolResult


@dataclass
class StreamChunk:
    """Represents a chunk of streamed response."""
    text: str
    finished: bool = False
    finish_reason: Optional[str] = None
    latency_seconds: Optional[float] = None
    usage: Optional[Dict[str, int]] = None
    context_used: Optional[str] = None
    search_metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMRunner(ABC):
    """Abstract base class for LLM runners with async support."""

    def __init__(
        self,
        handle: ModelHandle,
        knowledge_store: Optional[KnowledgeStore] = None
    ):
        """
        Initialize LLM runner.
        
        Args:
            handle: ModelHandle from ModelPool with loaded model
            knowledge_store: Optional KnowledgeStore for document search
        """
        self.handle = handle
        self.config = handle.config
        self.model_name = handle.config.model_name
        self.device = handle.config.device
        self.dtype = handle.config.dtype
        self.max_new_tokens = handle.config.max_new_tokens
        self.max_context_length = handle.max_context_length
        
        # Direct model access
        self.model = handle.model
        self.tokenizer = handle.tokenizer  # None for llama.cpp
        
        # Knowledge store
        self.knowledge_store = knowledge_store
        
        # Tool registry
        self.tool_registry = ToolRegistry()
        
        # Pending context from tool execution
        self._pending_context: Optional[str] = None
        
        # Register knowledge search tool if knowledge store is available
        if self.knowledge_store:
            from practorflow.llm.tools.knowledge_search import KnowledgeSearchTool
            knowledge_tool = KnowledgeSearchTool(
                knowledge_store=self.knowledge_store,
                default_top_k=self.config.max_search_results
            )
            self.tool_registry.register(knowledge_tool)

    def set_document_scope(self, document_ids: Optional[Set[str]]) -> None:
        """
        Set document scope for knowledge search.
        
        Args:
            document_ids: Set of document IDs to scope searches to,
                         or None to search all documents
        """
        self.tool_registry.set_document_scope(document_ids)
    
    def clear_document_scope(self) -> None:
        """Clear document scope to search all documents."""
        self.tool_registry.clear_document_scope()
    
    def get_document_scope(self) -> Optional[Set[str]]:
        """Get current document scope."""
        return self.tool_registry.get_document_scope()

    def search(self, query: str, top_k: Optional[int] = None) -> ToolResult:
        """
        Search knowledge base within current document scope.
        
        Results are stored as pending context for the next generate() call.
        
        Args:
            query: Search query string
            top_k: Maximum number of results (uses config default if not provided)
            
        Returns:
            ToolResult with search results
        """
        if "knowledge_search" not in self.tool_registry:
            return ToolResult(
                success=False,
                error="Knowledge search tool not available. Ensure knowledge_store is configured."
            )
        
        search_top_k = top_k if top_k is not None else self.config.max_search_results
        
        result = self.tool_registry.execute(
            "knowledge_search",
            query=query,
            top_k=search_top_k
        )
        
        # Store successful results as pending context
        if result.success and result.data:
            self._pending_context = result.to_context_string()
        
        return result
    
    def clear_pending_context(self) -> None:
        """Clear any pending context from tool execution."""
        self._pending_context = None
    
    def has_pending_context(self) -> bool:
        """Check if there's pending context from a tool execution."""
        return self._pending_context is not None

    @abstractmethod
    def supports_function_calling(self) -> bool:
        """
        Check if the model supports native function calling.
        
        This method should inspect the model's metadata, chat template,
        or configuration to determine if it has built-in support for
        function/tool calling.
        
        Returns:
            True if model supports native function calling, False otherwise
        """
        pass

    @abstractmethod
    async def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from messages or prompt with optional context.
        
        This is an async method that runs inference in a thread pool
        to avoid blocking the event loop.
        
        If search() was called before generate(), the search results
        are automatically included as context.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions for native function calling

        Returns:
            Dictionary with keys:
                - reply: Response from the model
                - latency_seconds: Generation time in seconds
                - tool_calls: (optional) List of tool calls if native FC used
                - context_used: (optional) Context string if search was used
                - search_metadata: (optional) Metadata about search results
            
        Note: Either messages or prompt must be provided, not both.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate text with streaming output.
        
        This is an async generator that yields chunks as they are generated.
        The underlying inference runs in a thread pool to avoid blocking.
        
        The final chunk will have finished=True and include timing/usage metadata.
        
        If search() was called before generate_stream(), the search results
        are automatically included as context.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions for native function calling

        Yields:
            StreamChunk objects containing:
                - text: New text fragment (delta)
                - finished: Whether generation is complete
                - finish_reason: Why generation stopped (on final chunk)
                - latency_seconds: Total time (on final chunk)
                - usage: Token counts (on final chunk, if available)
                - context_used: Context string (on final chunk, if search was used)
                - search_metadata: Search metadata (on final chunk, if available)
                - tool_calls: Native tool calls (on final chunk, if applicable)
            
        Note: Either messages or prompt must be provided, not both.
        """
        pass
        # This is needed to make it a valid async generator signature
        # Actual implementations will use 'yield'
        if False:
            yield StreamChunk(text="")

    def get_chat_reply_structure(self) -> Optional[str]:
        """
        Get the chat template that describes the structure of the reply field.
        
        Returns:
            The chat template string from the model, or None if not available.
        """
        return self.handle.get_chat_template()

    def _get_temperature(self, temperature: float = None) -> float:
        """Get temperature value, falling back to config default."""
        return temperature if temperature is not None else self.config.temperature

    def _get_top_p(self, top_p: float = None) -> float:
        """Get top_p value, falling back to config default."""
        return top_p if top_p is not None else self.config.top_p
    
    def _consume_pending_context(self) -> Optional[str]:
        """
        Get and clear pending context.
        
        Returns:
            Context string if available, None otherwise
        """
        context = self._pending_context
        self._pending_context = None
        return context