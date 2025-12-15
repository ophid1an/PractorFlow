"""
Abstract base class for knowledge storage.

Defines the interface for persistent document storage implementations.
Allows adaptation to different vector store backends (ChromaDB, Pinecone, Weaviate, Qdrant, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, BinaryIO
import numpy as np


class KnowledgeStore(ABC):
    """
    Abstract base class for persistent knowledge storage.
    
    Implementations should provide durable document storage with:
    - Document ingestion (file, bytes, base64, stream)
    - Vector similarity search
    - CRUD operations on documents and chunks
    - No TTL/expiration (permanent storage)
    """
    
    @abstractmethod
    def add_document_from_file(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load and store a document from a file path.
        
        Args:
            filepath: Path to the document file
            metadata: Optional additional metadata to store with the document
            
        Returns:
            Document info dict with id, filename, chunk counts, etc.
        """
        pass
    
    @abstractmethod
    def add_document_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load and store a document from raw bytes.
        
        Args:
            file_bytes: Raw file bytes
            filename: Filename with extension
            mime_type: Optional MIME type
            metadata: Optional additional metadata
            
        Returns:
            Document info dict
        """
        pass
    
    @abstractmethod
    def add_document_from_base64(
        self,
        base64_data: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load and store a document from base64 encoded data.
        
        Args:
            base64_data: Base64 encoded file data (may include data URI prefix)
            filename: Optional filename with extension
            mime_type: Optional MIME type
            metadata: Optional additional metadata
            
        Returns:
            Document info dict
        """
        pass
    
    @abstractmethod
    def add_document_from_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load and store a document from a file stream.
        
        Args:
            file_stream: File-like object with read() method
            filename: Filename with extension
            mime_type: Optional MIME type
            metadata: Optional additional metadata
            
        Returns:
            Document info dict
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents/chunks using text query.
        
        Args:
            query: Text query to search for
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dicts with id, text, similarity, metadata
        """
        pass
    
    @abstractmethod
    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents/chunks using embedding vector.
        
        Args:
            query_vector: Query embedding vector
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dicts with id, text, similarity, metadata
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document dict or None if not found
        """
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk dict or None if not found
        """
        pass
    
    @abstractmethod
    def get_context_chunk(self, parent_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a context (parent) chunk by key.
        
        Args:
            parent_key: Parent chunk key
            
        Returns:
            Context chunk dict or None if not found
        """
        pass
    
    @abstractmethod
    def get_retrieval_chunks_by_document(
        self,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all retrieval chunks for a specific document.
        
        Args:
            document_id: Document ID to get chunks for
            
        Returns:
            List of retrieval chunk dicts with id, text, metadata, vector
        """
        pass
    
    @abstractmethod
    def get_context_chunks_by_document(
        self,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all context (parent) chunks for a specific document.
        
        Args:
            document_id: Document ID to get context chunks for
            
        Returns:
            List of context chunk dicts with id, text, metadata
        """
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the store.
        
        Returns:
            List of document summary dicts with id, filename, metadata
        """
        pass
    
    @abstractmethod
    def count_documents(self) -> int:
        """
        Get the total number of documents.
        
        Returns:
            Document count
        """
        pass
    
    @abstractmethod
    def count_chunks(self) -> int:
        """
        Get the total number of retrieval chunks.
        
        Returns:
            Chunk count
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge store.
        
        Returns:
            Dict with counts, dimensions, storage info, etc.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all documents and chunks from the store.
        """
        pass