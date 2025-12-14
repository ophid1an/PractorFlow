from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from llm.llm_config import LLMConfig
from llm.base.session import Session
from llm.document.document_loader import DocumentLoader


class LLMRunner(ABC):
    """Abstract base class for LLM runners with RAG support."""

    def __init__(self, config: LLMConfig, session: Optional[Session] = None):
        """
        Initialize LLM runner.
        
        Args:
            config: LLM configuration
            session: Optional Session object for context and document management
        """
        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        self.dtype = config.dtype
        self.max_new_tokens = config.max_new_tokens
        
        # Session and document management
        self.session = session
        self.documents = []  # Fallback if no session

    @property
    def _document_loader(self):
        """Get document loader with tokenizer."""
        # Note: Subclasses must implement tokenizer property
        if not hasattr(self, '_doc_loader') or self._doc_loader is None:
            tokenizer = getattr(self, 'tokenizer', None)
            self._doc_loader = DocumentLoader(tokenizer=tokenizer)
        return self._doc_loader

    @property
    def _active_documents(self) -> List[Dict[str, Any]]:
        """Get active documents from session or fallback."""
        if self.session:
            return self.session.documents
        return self.documents

    @property
    def context_enabled(self) -> bool:
        """Check if context is enabled (has documents)."""
        return len(self._active_documents) > 0

    def load_document_from_base64(
        self,
        base64_data: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a document from base64 data and add to session.
        
        Args:
            base64_data: Base64 encoded file data
            filename: Optional filename with extension
            mime_type: Optional MIME type
            
        Returns:
            Loaded document dict
        """
        print(f"[Document] Loading from base64: {filename or 'unknown'}")
        document = self._document_loader.load_from_base64(base64_data, filename, mime_type)
        
        if self.session:
            self.session.add_document(document)
        else:
            existing_ids = {doc["id"] for doc in self.documents}
            if document["id"] in existing_ids:
                self.documents = [doc if doc["id"] != document["id"] else document 
                                 for doc in self.documents]
            else:
                self.documents.append(document)
        
        print(f"[Document] Loaded: {document['filename']} ({document['file_type']})")
        return document

    def load_document_from_stream(
        self,
        file_stream: Any,
        filename: str,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a document from a file stream (e.g., FastAPI UploadFile).
        
        Args:
            file_stream: File-like object with read() method
            filename: Filename with extension
            mime_type: Optional MIME type
            
        Returns:
            Loaded document dict
        """
        print(f"[Document] Loading from stream: {filename}")
        document = self._document_loader.load_from_stream(file_stream, filename, mime_type)
        
        if self.session:
            self.session.add_document(document)
        else:
            existing_ids = {doc["id"] for doc in self.documents}
            if document["id"] in existing_ids:
                self.documents = [doc if doc["id"] != document["id"] else document 
                                 for doc in self.documents]
            else:
                self.documents.append(document)
        
        print(f"[Document] Loaded: {document['filename']} ({document['file_type']})")
        return document

    @abstractmethod
    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        use_context: bool = True,
        max_context_docs: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text from messages or prompt with optional document context.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (for backward compatibility)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            use_context: Whether to include document context (default: True)
            max_context_docs: Maximum documents to include (default: None = unlimited)
            max_context_chars: Maximum characters for context (default: None = unlimited)

        Returns:
            Dictionary with keys: text, usage, latency_seconds, context_used (if context added)
            
        Note: Either messages or prompt must be provided, not both.
        """
        pass

    def _get_temperature(self, temperature: float = None) -> float:
        """Get temperature value, falling back to config default."""
        return temperature if temperature is not None else self.config.temperature

    def _get_top_p(self, top_p: float = None) -> float:
        """Get top_p value, falling back to config default."""
        return top_p if top_p is not None else self.config.top_p