from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Union
import uuid


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # "user" or "assistant"
    content: Union[str, List[Dict[str, Any]]]  # Can be string or structured content
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: str = "message"
    status: str = "completed"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with agent SDKs."""
        return {
            "id": self.id,
            "role": self.role,
            "type": self.type,
            "status": self.status,
            "content": self.content
        }
    
    def get_text_content(self) -> str:
        """Extract text content from message."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Extract text from structured content
            texts = []
            for item in self.content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
            return "\n".join(texts)
        return str(self.content)


@dataclass
class Session:
    """Conversation session with message history and document context."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    instructions: str = None  # System-level instructions
    documents: List[Dict[str, Any]] = field(default_factory=list)  # Loaded documents for context
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """
        Add a document to the session.
        
        Args:
            document: Document dict from DocumentLoader
                     Format: {"id": "...", "content": "...", "filename": "...", ...}
        """
        # Check if document with same ID already exists
        existing_ids = {doc["id"] for doc in self.documents}
        
        if document["id"] in existing_ids:
            # Update existing document
            for i, doc in enumerate(self.documents):
                if doc["id"] == document["id"]:
                    self.documents[i] = document
                    break
        else:
            # Add new document
            self.documents.append(document)
        
        # Update timestamp
        self.updated_at = datetime.now()
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the session by ID.
        
        Args:
            document_id: Document ID to remove
            
        Returns:
            True if document was found and removed, False otherwise
        """
        initial_length = len(self.documents)
        self.documents = [doc for doc in self.documents if doc["id"] != document_id]
        
        if len(self.documents) < initial_length:
            self.updated_at = datetime.now()
            return True
        return False
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document dict or None if not found
        """
        for doc in self.documents:
            if doc["id"] == document_id:
                return doc
        return None
    
    def clear_documents(self) -> None:
        """Remove all documents from the session."""
        if self.documents:
            self.documents = []
            self.updated_at = datetime.now()
    
    def get_document_count(self) -> int:
        """Get the number of documents in the session."""
        return len(self.documents)
    
    def list_documents(self) -> List[Dict[str, str]]:
        """
        Get a list of document summaries.
        
        Returns:
            List of dicts with id, filename, file_type
        """
        return [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "file_type": doc.get("file_type", "unknown")
            }
            for doc in self.documents
        ]