from typing import Optional
from dataclasses import dataclass


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
