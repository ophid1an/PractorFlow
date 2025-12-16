"""
ChromaDB-based persistent knowledge store.

Implements KnowledgeStore using ChromaDB for durable document storage.
Uses Small-to-Big chunking strategy via DocumentLoader.

Features:
- Persistent storage via ChromaDB PersistentClient
- Small-to-Big chunking (small chunks for retrieval, large for context)
- Scoped search by document IDs
- Cosine similarity search
- Full CRUD operations
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, BinaryIO
import numpy as np
import chromadb
from chromadb.config import Settings

from llm.knowledge.knowledge_store import KnowledgeStore
from llm.document.document_loader import DocumentLoader
from llm.document.embeddings import SentenceTransformerEmbeddingModel


@dataclass
class ChromaKnowledgeStoreConfig:
    """Configuration for ChromaDB knowledge store."""
    
    # ChromaDB persistence settings
    persist_directory: str = "./knowledge_db"
    retrieval_collection_name: str = "knowledge_retrieval"
    context_collection_name: str = "knowledge_context"
    documents_collection_name: str = "knowledge_documents"
    
    # ChromaDB client settings
    anonymized_telemetry: bool = False
    allow_reset: bool = True
    
    # Search settings
    distance_metric: str = "cosine"
    
    # Batch operation settings
    batch_size: int = 100
    
    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_cache_dir: Optional[str] = "./models"
    
    # Chunking settings (Small-to-Big)
    retrieval_chunk_size: int = 128
    retrieval_chunk_overlap: int = 20
    context_chunk_size: int = 1024
    context_chunk_overlap: int = 100


class ChromaKnowledgeStore(KnowledgeStore):
    """
    ChromaDB-based persistent knowledge store.
    
    Uses three collections:
    - retrieval: Small chunks with embeddings for similarity search
    - context: Large parent chunks for LLM context
    - documents: Document metadata and full content
    """
    
    def __init__(self, config: Optional[ChromaKnowledgeStoreConfig] = None):
        """
        Initialize ChromaDB knowledge store.
        
        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or ChromaKnowledgeStoreConfig()
        
        # Ensure persist directory exists
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        print(f"[ChromaKnowledgeStore] Initializing at: {self.config.persist_directory}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=self.config.anonymized_telemetry,
                allow_reset=self.config.allow_reset
            )
        )
        
        # Initialize collections
        self.retrieval_collection = self.client.get_or_create_collection(
            name=self.config.retrieval_collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
        
        self.context_collection = self.client.get_or_create_collection(
            name=self.config.context_collection_name
        )
        
        self.documents_collection = self.client.get_or_create_collection(
            name=self.config.documents_collection_name
        )
        
        print(f"[ChromaKnowledgeStore] Collections initialized")
        print(f"[ChromaKnowledgeStore] Existing documents: {self.documents_collection.count()}")
        print(f"[ChromaKnowledgeStore] Existing retrieval chunks: {self.retrieval_collection.count()}")
        print(f"[ChromaKnowledgeStore] Existing context chunks: {self.context_collection.count()}")
        
        # Initialize embedding model
        print(f"[ChromaKnowledgeStore] Loading embedding model: {self.config.embedding_model_name}")
        self.embedding_model = SentenceTransformerEmbeddingModel(
            model_name=self.config.embedding_model_name,
            cache_dir=self.config.embedding_cache_dir
        )
        self.dimension = self.embedding_model.embedding_dimension
        
        # Initialize document loader
        self.document_loader = DocumentLoader(
            retrieval_chunk_size=self.config.retrieval_chunk_size,
            retrieval_chunk_overlap=self.config.retrieval_chunk_overlap,
            context_chunk_size=self.config.context_chunk_size,
            context_chunk_overlap=self.config.context_chunk_overlap
        )
        
        print(f"[ChromaKnowledgeStore] Ready. Embedding dimension: {self.dimension}")
    
    def _get_current_timestamp(self) -> str:
        """Get current UTC timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat()
    
    def _store_document(
        self,
        document: Dict[str, Any],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a parsed document with all its chunks.
        
        Args:
            document: Parsed document from DocumentLoader
            additional_metadata: Optional extra metadata
            
        Returns:
            Document info dict
        """
        doc_id = document["id"]
        filename = document["filename"]
        file_type = document["file_type"]
        content = document["content"]
        retrieval_chunks = document.get("retrieval_chunks", [])
        context_chunks = document.get("context_chunks", [])
        
        created_at = self._get_current_timestamp()
        
        # Build document metadata
        doc_metadata = {
            "filename": filename,
            "file_type": file_type,
            "created_at": created_at,
            "retrieval_chunk_count": len(retrieval_chunks),
            "context_chunk_count": len(context_chunks),
            "content_length": len(content),
        }
        
        if additional_metadata:
            doc_metadata.update(additional_metadata)
        
        # Store document record
        self.documents_collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[doc_metadata]
        )
        
        print(f"[ChromaKnowledgeStore] Stored document: {filename} ({doc_id})")
        
        # Store context chunks (large parent chunks)
        if context_chunks:
            context_ids = []
            context_texts = []
            context_metadatas = []
            
            for ctx_chunk in context_chunks:
                ctx_id = f"{doc_id}_{ctx_chunk['id']}"
                context_ids.append(ctx_id)
                context_texts.append(ctx_chunk["text"])
                
                ctx_metadata = ctx_chunk.get("metadata", {}).copy()
                ctx_metadata["document_id"] = doc_id
                ctx_metadata["filename"] = filename
                ctx_metadata["created_at"] = created_at
                context_metadatas.append(ctx_metadata)
            
            # Batch add context chunks
            for i in range(0, len(context_ids), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(context_ids))
                self.context_collection.add(
                    ids=context_ids[i:batch_end],
                    documents=context_texts[i:batch_end],
                    metadatas=context_metadatas[i:batch_end]
                )
            
            print(f"[ChromaKnowledgeStore] Stored {len(context_chunks)} context chunks")
        
        # Embed and store retrieval chunks (small chunks)
        if retrieval_chunks:
            print(f"[ChromaKnowledgeStore] Embedding {len(retrieval_chunks)} retrieval chunks...")
            
            chunk_texts = [chunk["text"] for chunk in retrieval_chunks]
            embeddings = self.embedding_model.embed_batch(chunk_texts, show_progress=True)
            
            chunk_ids = []
            chunk_metadatas = []
            
            for chunk in retrieval_chunks:
                chunk_id = f"{doc_id}_{chunk['id']}"
                parent_key = f"{doc_id}_{chunk['parent_id']}"
                
                chunk_ids.append(chunk_id)
                
                chunk_metadata = chunk.get("metadata", {}).copy()
                chunk_metadata["document_id"] = doc_id
                chunk_metadata["parent_key"] = parent_key
                chunk_metadata["filename"] = filename
                chunk_metadata["created_at"] = created_at
                chunk_metadatas.append(chunk_metadata)
            
            # Batch add retrieval chunks with embeddings
            embeddings_list = embeddings.tolist()
            
            for i in range(0, len(chunk_ids), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(chunk_ids))
                self.retrieval_collection.add(
                    ids=chunk_ids[i:batch_end],
                    embeddings=embeddings_list[i:batch_end],
                    documents=chunk_texts[i:batch_end],
                    metadatas=chunk_metadatas[i:batch_end]
                )
            
            print(f"[ChromaKnowledgeStore] Stored {len(retrieval_chunks)} retrieval chunks")
        
        return {
            "id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "content_length": len(content),
            "retrieval_chunk_count": len(retrieval_chunks),
            "context_chunk_count": len(context_chunks),
            "created_at": created_at,
        }
    
    def add_document_from_file(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load and store a document from a file path."""
        print(f"[ChromaKnowledgeStore] Loading file: {filepath}")
        document = self.document_loader.load_file(filepath)
        return self._store_document(document, metadata)
    
    def add_document_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load and store a document from raw bytes."""
        print(f"[ChromaKnowledgeStore] Loading from bytes: {filename}")
        document = self.document_loader.load_from_bytes(file_bytes, filename, mime_type)
        return self._store_document(document, metadata)
    
    def add_document_from_base64(
        self,
        base64_data: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load and store a document from base64 encoded data."""
        print(f"[ChromaKnowledgeStore] Loading from base64: {filename or 'unknown'}")
        document = self.document_loader.load_from_base64(base64_data, filename, mime_type)
        return self._store_document(document, metadata)
    
    def add_document_from_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load and store a document from a file stream."""
        print(f"[ChromaKnowledgeStore] Loading from stream: {filename}")
        document = self.document_loader.load_from_stream(file_stream, filename, mime_type)
        return self._store_document(document, metadata)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using text query."""
        if self.retrieval_collection.count() == 0:
            return []
        
        # Embed query
        query_vector = self.embedding_model.embed(query)
        return self.search_by_vector(query_vector, top_k, filter_metadata)
    
    def search_scoped(
        self,
        query: str,
        top_k: int = 10,
        document_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with Small-to-Big retrieval, optionally scoped to specific documents.
        
        Searches small retrieval chunks for similarity, then returns the
        corresponding parent (context) chunks for richer LLM context.
        
        Args:
            query: Text query to search for
            top_k: Maximum number of parent chunks to return
            document_ids: Optional set of document IDs to scope search to.
                         If None, searches all documents.
            
        Returns:
            List of parent chunk dicts with id, text, similarity, metadata, document_id
        """
        if self.retrieval_collection.count() == 0:
            return []
        
        # Embed query
        query_vector = self.embedding_model.embed(query)
        query_list = query_vector.tolist()
        
        # Build where clause for document scoping
        where_clause = None
        if document_ids:
            doc_id_list = list(document_ids)
            if len(doc_id_list) == 1:
                where_clause = {"document_id": {"$eq": doc_id_list[0]}}
            else:
                where_clause = {"document_id": {"$in": doc_id_list}}
        
        # Search retrieval chunks (get more than needed to account for parent deduplication)
        search_k = top_k * 3
        
        results = self.retrieval_collection.query(
            query_embeddings=[query_list],
            n_results=search_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results['ids'] or not results['ids'][0]:
            return []
        
        # Extract results
        ids = results['ids'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
        distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
        
        # Deduplicate by parent chunk, keeping best similarity
        seen_parents = {}
        
        for i, chunk_id in enumerate(ids):
            metadata = metadatas[i] if metadatas[i] else {}
            parent_key = metadata.get("parent_key")
            
            if not parent_key:
                continue
            
            similarity = 1.0 - distances[i]
            
            if parent_key not in seen_parents or similarity > seen_parents[parent_key]["similarity"]:
                seen_parents[parent_key] = {
                    "parent_key": parent_key,
                    "similarity": similarity,
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                }
        
        # Sort by similarity and limit to top_k
        sorted_parents = sorted(
            seen_parents.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]
        
        # Fetch parent (context) chunks
        output = []
        for parent_info in sorted_parents:
            parent_key = parent_info["parent_key"]
            
            try:
                parent_result = self.context_collection.get(
                    ids=[parent_key],
                    include=["documents", "metadatas"]
                )
                
                if parent_result and parent_result['ids']:
                    parent_text = parent_result['documents'][0] if parent_result['documents'] else ''
                    parent_metadata = parent_result['metadatas'][0] if parent_result['metadatas'] else {}
                    
                    output.append({
                        "id": parent_key,
                        "text": parent_text,
                        "similarity": parent_info["similarity"],
                        "metadata": parent_metadata,
                        "document_id": parent_info["document_id"],
                        "filename": parent_info["filename"],
                    })
            except Exception as e:
                print(f"[ChromaKnowledgeStore] Error fetching parent chunk {parent_key}: {e}")
                continue
        
        print(f"[ChromaKnowledgeStore] search_scoped: {len(output)} parent chunks from {len(ids)} retrieval hits")
        
        return output
    
    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using embedding vector."""
        if self.retrieval_collection.count() == 0:
            return []
        
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        # Build where clause
        where_clause = None
        if filter_metadata:
            if len(filter_metadata) == 1:
                key, value = next(iter(filter_metadata.items()))
                where_clause = {key: {"$eq": value}}
            else:
                where_clause = {
                    "$and": [{k: {"$eq": v}} for k, v in filter_metadata.items()]
                }
        
        results = self.retrieval_collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            documents = results['documents'][0] if results['documents'] else [''] * len(ids)
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
            distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
            
            for i, chunk_id in enumerate(ids):
                similarity = 1.0 - distances[i]
                output.append({
                    "id": chunk_id,
                    "text": documents[i] if documents[i] else '',
                    "similarity": similarity,
                    "metadata": metadatas[i] if metadatas[i] else {},
                })
        
        return output
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        try:
            results = self.documents_collection.get(
                ids=[document_id],
                include=["documents", "metadatas"]
            )
            
            if results and results['ids']:
                return {
                    "id": results['ids'][0],
                    "content": results['documents'][0] if results['documents'] else '',
                    "metadata": results['metadatas'][0] if results['metadatas'] else {},
                }
        except Exception:
            pass
        
        return None
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific retrieval chunk by ID."""
        try:
            results = self.retrieval_collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results and results['ids']:
                return {
                    "id": results['ids'][0],
                    "text": results['documents'][0] if results['documents'] else '',
                    "metadata": results['metadatas'][0] if results['metadatas'] else {},
                    "vector": np.array(results['embeddings'][0]) if results['embeddings'] else None,
                }
        except Exception:
            pass
        
        return None
    
    def get_context_chunk(self, parent_key: str) -> Optional[Dict[str, Any]]:
        """Get a context (parent) chunk by key."""
        try:
            results = self.context_collection.get(
                ids=[parent_key],
                include=["documents", "metadatas"]
            )
            
            if results and results['ids']:
                return {
                    "id": results['ids'][0],
                    "text": results['documents'][0] if results['documents'] else '',
                    "metadata": results['metadatas'][0] if results['metadatas'] else {},
                }
        except Exception:
            pass
        
        return None
    
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
        try:
            results = self.retrieval_collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["documents", "metadatas", "embeddings"]
            )
            
            chunks = []
            if results and results['ids']:
                ids = results['ids']
                documents = results['documents'] if results['documents'] else [''] * len(ids)
                metadatas = results['metadatas'] if results['metadatas'] else [{}] * len(ids)
                embeddings = results['embeddings'] if results['embeddings'] else [None] * len(ids)
                
                for i, chunk_id in enumerate(ids):
                    chunk = {
                        "id": chunk_id,
                        "text": documents[i] if documents[i] else '',
                        "metadata": metadatas[i] if metadatas[i] else {},
                    }
                    if embeddings[i] is not None:
                        chunk["vector"] = np.array(embeddings[i])
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"[ChromaKnowledgeStore] Error getting retrieval chunks: {e}")
            return []
    
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
        try:
            results = self.context_collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results and results['ids']:
                ids = results['ids']
                documents = results['documents'] if results['documents'] else [''] * len(ids)
                metadatas = results['metadatas'] if results['metadatas'] else [{}] * len(ids)
                
                for i, chunk_id in enumerate(ids):
                    chunks.append({
                        "id": chunk_id,
                        "text": documents[i] if documents[i] else '',
                        "metadata": metadatas[i] if metadatas[i] else {},
                    })
            
            return chunks
            
        except Exception as e:
            print(f"[ChromaKnowledgeStore] Error getting context chunks: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        # Check if document exists
        doc = self.get_document(document_id)
        if not doc:
            return False
        
        try:
            # Delete retrieval chunks
            retrieval_results = self.retrieval_collection.get(
                where={"document_id": {"$eq": document_id}},
                include=[]
            )
            if retrieval_results and retrieval_results['ids']:
                for i in range(0, len(retrieval_results['ids']), self.config.batch_size):
                    batch_ids = retrieval_results['ids'][i:i + self.config.batch_size]
                    self.retrieval_collection.delete(ids=batch_ids)
                print(f"[ChromaKnowledgeStore] Deleted {len(retrieval_results['ids'])} retrieval chunks")
            
            # Delete context chunks
            context_results = self.context_collection.get(
                where={"document_id": {"$eq": document_id}},
                include=[]
            )
            if context_results and context_results['ids']:
                for i in range(0, len(context_results['ids']), self.config.batch_size):
                    batch_ids = context_results['ids'][i:i + self.config.batch_size]
                    self.context_collection.delete(ids=batch_ids)
                print(f"[ChromaKnowledgeStore] Deleted {len(context_results['ids'])} context chunks")
            
            # Delete document record
            self.documents_collection.delete(ids=[document_id])
            print(f"[ChromaKnowledgeStore] Deleted document: {document_id}")
            
            return True
            
        except Exception as e:
            print(f"[ChromaKnowledgeStore] Error deleting document: {e}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the store."""
        try:
            results = self.documents_collection.get(
                include=["metadatas"]
            )
            
            documents = []
            if results and results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    documents.append({
                        "id": doc_id,
                        "filename": metadata.get("filename", "unknown"),
                        "file_type": metadata.get("file_type", "unknown"),
                        "created_at": metadata.get("created_at", ""),
                        "retrieval_chunk_count": metadata.get("retrieval_chunk_count", 0),
                        "context_chunk_count": metadata.get("context_chunk_count", 0),
                        "content_length": metadata.get("content_length", 0),
                    })
            
            return documents
            
        except Exception as e:
            print(f"[ChromaKnowledgeStore] Error listing documents: {e}")
            return []
    
    def count_documents(self) -> int:
        """Get the total number of documents."""
        return self.documents_collection.count()
    
    def count_chunks(self) -> int:
        """Get the total number of retrieval chunks."""
        return self.retrieval_collection.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge store."""
        return {
            "documents": self.documents_collection.count(),
            "retrieval_chunks": self.retrieval_collection.count(),
            "context_chunks": self.context_collection.count(),
            "embedding_dimension": self.dimension,
            "embedding_model": self.config.embedding_model_name,
            "persist_directory": self.config.persist_directory,
            "distance_metric": self.config.distance_metric,
            "retrieval_chunk_size": self.config.retrieval_chunk_size,
            "context_chunk_size": self.config.context_chunk_size,
            "storage_type": "chromadb_persistent",
        }
    
    def clear(self) -> None:
        """Clear all documents and chunks from the store."""
        try:
            # Delete and recreate all collections
            self.client.delete_collection(self.config.retrieval_collection_name)
            self.client.delete_collection(self.config.context_collection_name)
            self.client.delete_collection(self.config.documents_collection_name)
            
            self.retrieval_collection = self.client.get_or_create_collection(
                name=self.config.retrieval_collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            self.context_collection = self.client.get_or_create_collection(
                name=self.config.context_collection_name
            )
            self.documents_collection = self.client.get_or_create_collection(
                name=self.config.documents_collection_name
            )
            
            print("[ChromaKnowledgeStore] Cleared all data")
            
        except Exception as e:
            print(f"[ChromaKnowledgeStore] Error clearing: {e}")
    
    def __repr__(self) -> str:
        return (
            f"ChromaKnowledgeStore(documents={self.count_documents()}, "
            f"chunks={self.count_chunks()}, "
            f"path='{self.config.persist_directory}')"
        )