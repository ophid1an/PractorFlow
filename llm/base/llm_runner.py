from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set

from llm.llm_config import LLMConfig
from llm.base.session import Session
from llm.knowledge.knowledge_store import KnowledgeStore
from llm.tools.tool_registry import ToolRegistry
from llm.tools.base import ToolResult


class LLMRunner(ABC):
    """Abstract base class for LLM runners with tool support."""

    def __init__(
        self,
        config: LLMConfig,
        session: Optional[Session] = None,
        knowledge_store: Optional[KnowledgeStore] = None
    ):
        """
        Initialize LLM runner.
        
        Args:
            config: LLM configuration
            session: Optional Session object for conversation management
            knowledge_store: Optional KnowledgeStore for document search
        """
        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        self.dtype = config.dtype
        self.max_new_tokens = config.max_new_tokens
        
        # Session management
        self.session = session
        
        # Knowledge store
        self.knowledge_store = knowledge_store
        
        # Tool registry
        self.tool_registry = ToolRegistry()
        
        # Pending context from tool execution
        self._pending_context: Optional[str] = None

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
    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from messages or prompt with optional context.
        
        If search() was called before generate(), the search results
        are automatically included as context.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)

        Returns:
            Dictionary with keys:
                - reply: Response from the model
                - latency_seconds: Generation time in seconds
                - context_used: (optional) Context string if search was used
                - search_metadata: (optional) Metadata about search results
            
        Note: Either messages or prompt must be provided, not both.
        """
        pass

    @abstractmethod
    def get_chat_reply_structure(self) -> Optional[str]:
        """
        Get the chat template that describes the structure of the reply field.
        
        Returns:
            The chat template string from the model, or None if not available.
        """
        pass

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