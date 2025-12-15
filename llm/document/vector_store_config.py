"""
Configuration for ChromaDB-based vector store.

Centralizes all vector store settings in a single dataclass.
"""

from dataclasses import dataclass


@dataclass
class VectorStoreConfig:
    """Configuration for TempVectorStore (ChromaDB-based)."""
    
    # ChromaDB persistence settings
    persist_directory: str = "./chroma_db"
    collection_name: str = "temp_vectors"
    
    # TTL settings
    default_ttl_hours: int = 24
    
    # ChromaDB client settings
    anonymized_telemetry: bool = False
    allow_reset: bool = True
    
    # Search settings
    distance_metric: str = "cosine"  # Options: cosine, l2, ip
    
    # Batch operation settings
    batch_size: int = 100  # Max entries per batch operation
