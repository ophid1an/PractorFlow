"""
Temporary vector store using ChromaDB with time-based expiration.

Replaces MemoryVectorStore with persistent storage and TTL support.

Features:
- ChromaDB PersistentClient for durable storage
- Automatic timestamp metadata (created_at, expires_at) as Unix timestamps
- TTL-based expiration support
- Same API interface as MemoryVectorStore
- Cosine similarity search
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import chromadb
from chromadb.config import Settings

from llm.document.vector_store_config import VectorStoreConfig


class TempVectorStore:
    """
    ChromaDB-based vector store with time-based expiration support.
    
    Features:
    - Persistent storage via ChromaDB
    - Automatic TTL metadata for cleanup (Unix timestamps)
    - Cosine similarity search
    - API compatible with MemoryVectorStore
    """
    
    def __init__(
        self,
        config: VectorStoreConfig,
        dimension: Optional[int] = None
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            config: VectorStoreConfig instance with all settings
            dimension: Expected vector dimension (optional, auto-detected from first insert)
        """
        self.config = config
        self.dimension = dimension
        
        # Initialize ChromaDB PersistentClient
        print(f"[TempVectorStore] Initializing ChromaDB at: {config.persist_directory}")
        
        self.client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(
                anonymized_telemetry=config.anonymized_telemetry,
                allow_reset=config.allow_reset
            )
        )
        
        # Get or create collection with configured distance metric
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": config.distance_metric}
        )
        
        print(f"[TempVectorStore] Collection '{config.collection_name}' ready")
        print(f"[TempVectorStore] Default TTL: {config.default_ttl_hours} hours")
        print(f"[TempVectorStore] Distance metric: {config.distance_metric}")
        print(f"[TempVectorStore] Existing entries: {self.collection.count()}")
    
    def _get_current_timestamp(self) -> float:
        """Get current UTC time as Unix timestamp (float)."""
        return datetime.now(timezone.utc).timestamp()
    
    def _calculate_expiry_timestamp(self, ttl_hours: int) -> float:
        """Calculate expiry Unix timestamp from now."""
        expiry = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        return expiry.timestamp()
    
    def add(
        self,
        text: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        ttl_hours: Optional[int] = None,
        document_id: Optional[str] = None
    ) -> str:
        """
        Add a single entry to the vector store.
        
        Args:
            text: The text content
            vector: The embedding vector
            metadata: Optional metadata dict
            id: Optional custom ID (generated if not provided)
            ttl_hours: Time-to-live in hours (uses config default if not provided)
            document_id: Parent document ID for grouping
            
        Returns:
            The entry ID
        """
        import uuid
        
        # Update dimension if not set
        if self.dimension is None:
            self.dimension = len(vector)
        
        # Generate ID if not provided
        if id is None:
            id = f"vec_{uuid.uuid4().hex[:8]}"
        
        # Calculate timestamps as floats
        ttl = ttl_hours if ttl_hours is not None else self.config.default_ttl_hours
        created_at = self._get_current_timestamp()
        expires_at = self._calculate_expiry_timestamp(ttl)
        
        # Build metadata with timestamps (as floats for ChromaDB comparison)
        entry_metadata = metadata.copy() if metadata else {}
        entry_metadata.update({
            "created_at": created_at,
            "expires_at": expires_at,
            "ttl_hours": ttl,
        })
        
        if document_id:
            entry_metadata["document_id"] = document_id
        
        # Convert vector to list for ChromaDB
        vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
        
        # Add to ChromaDB
        self.collection.add(
            ids=[id],
            embeddings=[vector_list],
            documents=[text],
            metadatas=[entry_metadata]
        )
        
        return id
    
    def add_batch(
        self,
        texts: List[str],
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        ttl_hours: Optional[int] = None,
        document_id: Optional[str] = None
    ) -> List[str]:
        """
        Add multiple entries in batch.
        
        Args:
            texts: List of text contents
            vectors: Array of vectors, shape (n_entries, dimension)
            metadatas: Optional list of metadata dicts
            ids: Optional list of custom IDs
            ttl_hours: Time-to-live in hours (uses config default if not provided)
            document_id: Parent document ID for grouping
            
        Returns:
            List of entry IDs
        """
        import uuid
        
        n_entries = len(texts)
        
        if vectors.shape[0] != n_entries:
            raise ValueError(
                f"Number of vectors {vectors.shape[0]} does not match number of texts {n_entries}"
            )
        
        # Update dimension if not set
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        
        # Prepare IDs
        if ids is None:
            ids = [f"vec_{uuid.uuid4().hex[:8]}" for _ in range(n_entries)]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in range(n_entries)]
        
        # Calculate timestamps as floats
        ttl = ttl_hours if ttl_hours is not None else self.config.default_ttl_hours
        created_at = self._get_current_timestamp()
        expires_at = self._calculate_expiry_timestamp(ttl)
        
        # Add timestamp metadata to all entries
        batch_metadatas = []
        for meta in metadatas:
            entry_metadata = meta.copy() if meta else {}
            entry_metadata.update({
                "created_at": created_at,
                "expires_at": expires_at,
                "ttl_hours": ttl,
            })
            if document_id:
                entry_metadata["document_id"] = document_id
            batch_metadatas.append(entry_metadata)
        
        # Convert vectors to list format
        vectors_list = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors
        
        # Process in batches according to config
        all_ids = []
        batch_size = self.config.batch_size
        
        for i in range(0, n_entries, batch_size):
            batch_end = min(i + batch_size, n_entries)
            
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=vectors_list[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=batch_metadatas[i:batch_end]
            )
            
            all_ids.extend(ids[i:batch_end])
        
        return all_ids
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """
        Search for similar entries using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (exact match)
            
        Returns:
            List of tuples: (id, similarity_score, text, metadata)
            Sorted by similarity (highest first)
        """
        if self.collection.count() == 0:
            return []
        
        # Convert query vector to list
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        
        # Build where clause for filtering
        where_clause = None
        if filter_metadata:
            if len(filter_metadata) == 1:
                key, value = next(iter(filter_metadata.items()))
                where_clause = {key: {"$eq": value}}
            else:
                where_clause = {
                    "$and": [{k: {"$eq": v}} for k, v in filter_metadata.items()]
                }
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert results to expected format
        # For cosine distance: similarity = 1 - distance
        output = []
        
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            documents = results['documents'][0] if results['documents'] else [''] * len(ids)
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
            distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
            
            for i, id in enumerate(ids):
                # Convert cosine distance to similarity
                similarity = 1.0 - distances[i]
                text = documents[i] if documents[i] else ''
                metadata = metadatas[i] if metadatas[i] else {}
                output.append((id, similarity, text, metadata))
        
        return output
    
    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entry by ID.
        
        Args:
            id: Entry ID
            
        Returns:
            Dict with id, text, metadata or None if not found
        """
        try:
            results = self.collection.get(
                ids=[id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results and results['ids']:
                return {
                    "id": results['ids'][0],
                    "text": results['documents'][0] if results['documents'] else '',
                    "metadata": results['metadatas'][0] if results['metadatas'] else {},
                    "vector": np.array(results['embeddings'][0]) if results['embeddings'] else None
                }
        except Exception:
            pass
        
        return None
    
    def delete(self, id: str) -> bool:
        """
        Delete an entry by ID.
        
        Args:
            id: Entry ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            # Check if exists first
            existing = self.collection.get(ids=[id])
            if not existing or not existing['ids']:
                return False
            
            self.collection.delete(ids=[id])
            return True
        except Exception:
            return False
    
    def delete_batch(self, ids: List[str]) -> int:
        """
        Delete multiple entries by IDs.
        
        Args:
            ids: List of entry IDs to delete
            
        Returns:
            Number of entries deleted
        """
        if not ids:
            return 0
        
        try:
            deleted_count = 0
            batch_size = self.config.batch_size
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                
                # Get existing IDs first
                existing = self.collection.get(ids=batch_ids)
                existing_ids = existing['ids'] if existing and existing['ids'] else []
                
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
                    deleted_count += len(existing_ids)
            
            return deleted_count
        except Exception as e:
            print(f"[TempVectorStore] Error in batch delete: {e}")
            return 0
    
    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all entries for a specific document.
        
        Args:
            document_id: Document ID to delete entries for
            
        Returns:
            Number of entries deleted
        """
        try:
            # Query all entries with this document_id
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["metadatas"]
            )
            
            if results and results['ids']:
                ids_to_delete = results['ids']
                
                # Delete in batches
                deleted_count = 0
                batch_size = self.config.batch_size
                
                for i in range(0, len(ids_to_delete), batch_size):
                    batch_ids = ids_to_delete[i:i + batch_size]
                    self.collection.delete(ids=batch_ids)
                    deleted_count += len(batch_ids)
                
                return deleted_count
            
            return 0
        except Exception as e:
            print(f"[TempVectorStore] Error deleting by document_id: {e}")
            return 0
    
    def get_expired_ids(
        self,
        current_time: Optional[datetime] = None
    ) -> List[Tuple[str, str]]:
        """
        Get IDs of expired entries.
        
        Args:
            current_time: Time to check against (defaults to now)
            
        Returns:
            List of tuples: (entry_id, document_id)
        """
        if current_time is None:
            current_timestamp = self._get_current_timestamp()
        else:
            current_timestamp = current_time.timestamp()
        
        try:
            # Query entries where expires_at < current_timestamp (float comparison)
            results = self.collection.get(
                where={"expires_at": {"$lt": current_timestamp}},
                include=["metadatas"]
            )
            
            expired = []
            if results and results['ids']:
                for i, id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    document_id = metadata.get('document_id', '')
                    expired.append((id, document_id))
            
            return expired
        except Exception as e:
            print(f"[TempVectorStore] Error getting expired IDs: {e}")
            return []
    
    def clear(self):
        """Clear all entries from the store."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            print("[TempVectorStore] Cleared all entries")
        except Exception as e:
            print(f"[TempVectorStore] Error clearing: {e}")
    
    def count(self) -> int:
        """Get the number of entries in the store."""
        return self.collection.count()
    
    def list_ids(self) -> List[str]:
        """Get all entry IDs."""
        try:
            results = self.collection.get(include=[])
            return results['ids'] if results and results['ids'] else []
        except Exception:
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        count = self.count()
        
        # Get expired count
        expired = self.get_expired_ids()
        
        return {
            "total_entries": count,
            "dimension": self.dimension,
            "persist_directory": self.config.persist_directory,
            "collection_name": self.config.collection_name,
            "default_ttl_hours": self.config.default_ttl_hours,
            "distance_metric": self.config.distance_metric,
            "expired_entries": len(expired),
            "storage_type": "chromadb_persistent"
        }
    
    def __len__(self) -> int:
        """Get the number of entries."""
        return self.count()
    
    def __repr__(self) -> str:
        return (
            f"TempVectorStore(entries={self.count()}, "
            f"collection='{self.config.collection_name}', "
            f"ttl={self.config.default_ttl_hours}h)"
        )