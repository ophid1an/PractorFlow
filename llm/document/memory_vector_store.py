"""
In-memory vector store for document chunks with semantic search.

This is a pure in-memory implementation - no persistence to disk.

Features:
- Fast cosine similarity search
- Metadata filtering
- Batch operations
- No external dependencies (uses numpy)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import uuid


@dataclass
class VectorEntry:
    """Single entry in the vector store."""
    id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryVectorStore:
    """
    Pure in-memory vector store with cosine similarity search.
    
    Features:
    - Fast similarity search using numpy
    - Metadata support for filtering
    - No persistence - data lost on restart
    """
    
    def __init__(self, dimension: Optional[int] = None):
        """
        Initialize vector store.
        
        Args:
            dimension: Expected vector dimension (optional, auto-detected from first insert)
        """
        self.dimension = dimension
        self.entries: Dict[str, VectorEntry] = {}
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._dirty = True  # Flag to rebuild index
        
        print(f"[MemoryVectorStore] Initialized (dimension: {dimension or 'auto-detect'})")
    
    def add(
        self,
        text: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        Add a single entry to the vector store.
        
        Args:
            text: The text content
            vector: The embedding vector
            metadata: Optional metadata dict
            id: Optional custom ID (generated if not provided)
            
        Returns:
            The entry ID
        """
        # Validate and set dimension
        if self.dimension is None:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match store dimension {self.dimension}"
            )
        
        # Generate ID if not provided
        if id is None:
            id = f"vec_{uuid.uuid4().hex[:8]}"
        
        # Create entry
        entry = VectorEntry(
            id=id,
            vector=vector,
            text=text,
            metadata=metadata or {}
        )
        
        # Store entry
        self.entries[id] = entry
        self._dirty = True
        
        return id
    
    def add_batch(
        self,
        texts: List[str],
        vectors: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple entries in batch.
        
        Args:
            texts: List of text contents
            vectors: Array of vectors, shape (n_entries, dimension)
            metadatas: Optional list of metadata dicts
            ids: Optional list of custom IDs
            
        Returns:
            List of entry IDs
        """
        n_entries = len(texts)
        
        if vectors.shape[0] != n_entries:
            raise ValueError(
                f"Number of vectors {vectors.shape[0]} does not match number of texts {n_entries}"
            )
        
        # Prepare metadata and IDs
        if metadatas is None:
            metadatas = [{}] * n_entries
        if ids is None:
            ids = [f"vec_{uuid.uuid4().hex[:8]}" for _ in range(n_entries)]
        
        # Validate dimension
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        elif vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match store dimension {self.dimension}"
            )
        
        # Add all entries
        entry_ids = []
        for i in range(n_entries):
            id = self.add(
                text=texts[i],
                vector=vectors[i],
                metadata=metadatas[i],
                id=ids[i]
            )
            entry_ids.append(id)
        
        return entry_ids
    
    def _rebuild_index(self):
        """Rebuild the vector index for fast search."""
        if not self.entries:
            self._vectors = None
            self._ids = []
            self._dirty = False
            return
        
        # Stack all vectors
        self._ids = list(self.entries.keys())
        vectors = [self.entries[id].vector for id in self._ids]
        self._vectors = np.vstack(vectors)
        self._dirty = False
    
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
        if not self.entries:
            return []
        
        # Rebuild index if needed
        if self._dirty:
            self._rebuild_index()
        
        # Validate query dimension
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match store dimension {self.dimension}"
            )
        
        # Compute cosine similarities
        # Assuming vectors are already normalized, cosine similarity = dot product
        similarities = np.dot(self._vectors, query_vector)
        
        # Apply metadata filtering if specified
        if filter_metadata:
            valid_indices = []
            for idx, id in enumerate(self._ids):
                entry = self.entries[id]
                # Check if all filter conditions match
                if all(entry.metadata.get(k) == v for k, v in filter_metadata.items()):
                    valid_indices.append(idx)
            
            if not valid_indices:
                return []
            
            # Filter similarities and IDs
            similarities = similarities[valid_indices]
            filtered_ids = [self._ids[i] for i in valid_indices]
        else:
            filtered_ids = self._ids
        
        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            id = filtered_ids[idx]
            entry = self.entries[id]
            similarity = float(similarities[idx])
            results.append((id, similarity, entry.text, entry.metadata))
        
        return results
    
    def get(self, id: str) -> Optional[VectorEntry]:
        """Get an entry by ID."""
        return self.entries.get(id)
    
    def delete(self, id: str) -> bool:
        """
        Delete an entry by ID.
        
        Args:
            id: Entry ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if id in self.entries:
            del self.entries[id]
            self._dirty = True
            return True
        return False
    
    def clear(self):
        """Clear all entries from the store."""
        self.entries.clear()
        self._vectors = None
        self._ids = []
        self._dirty = True
        print("[MemoryVectorStore] Cleared all entries")
    
    def count(self) -> int:
        """Get the number of entries in the store."""
        return len(self.entries)
    
    def list_ids(self) -> List[str]:
        """Get all entry IDs."""
        return list(self.entries.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_entries": len(self.entries),
            "dimension": self.dimension,
            "index_built": not self._dirty,
            "memory_mb": self._estimate_memory_mb()
        }
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        if not self.entries or self.dimension is None:
            return 0.0
        
        # Rough estimate: each vector + metadata
        vector_size = self.dimension * 4  # float32
        avg_text_size = sum(len(e.text) for e in self.entries.values()) / len(self.entries)
        entry_size = vector_size + avg_text_size + 1000  # +1000 for metadata overhead
        
        total_bytes = entry_size * len(self.entries)
        return total_bytes / (1024 * 1024)
    
    def __len__(self) -> int:
        """Get the number of entries."""
        return len(self.entries)
    
    def __repr__(self) -> str:
        return f"MemoryVectorStore(entries={len(self.entries)}, dimension={self.dimension})"
