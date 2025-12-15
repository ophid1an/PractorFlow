"""
Vector store cleanup module for expired documents.

Standalone cleanup functionality - NO background threads or schedulers.
External scheduler (cron, celery, etc.) calls cleanup_expired_documents().

Features:
- Query and delete expired chunks from ChromaDB
- Clean up corresponding context chunks from context_store dict
- Group deletions by document_id for efficiency
- Graceful error handling (continues on individual failures)
- Returns detailed cleanup statistics
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

from llm.document.temp_vector_store import TempVectorStore


@dataclass
class CleanupResult:
    """Result statistics from cleanup operation."""
    
    # Counts
    expired_chunks_found: int = 0
    chunks_deleted: int = 0
    context_chunks_deleted: int = 0
    documents_affected: int = 0
    
    # Document details
    deleted_document_ids: List[str] = field(default_factory=list)
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    partial_failure: bool = False
    
    # Timing
    cleanup_time_seconds: float = 0.0
    cleanup_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "expired_chunks_found": self.expired_chunks_found,
            "chunks_deleted": self.chunks_deleted,
            "context_chunks_deleted": self.context_chunks_deleted,
            "documents_affected": self.documents_affected,
            "deleted_document_ids": self.deleted_document_ids,
            "errors": self.errors,
            "partial_failure": self.partial_failure,
            "cleanup_time_seconds": self.cleanup_time_seconds,
            "cleanup_timestamp": self.cleanup_timestamp,
        }


def cleanup_expired_documents(
    vector_store: TempVectorStore,
    context_store: Dict[str, Dict[str, Any]],
    current_time: Optional[datetime] = None
) -> CleanupResult:
    """
    Clean up expired documents from vector store and context store.
    
    This function is passive - it should be called by an external scheduler
    (cron job, Celery task, APScheduler, etc.).
    
    Args:
        vector_store: TempVectorStore instance (ChromaDB-based)
        context_store: Dictionary storing context chunks (parent chunks)
                      Format: {parent_key: {"text": ..., "metadata": ..., "document_id": ...}}
        current_time: Time to check expiration against (defaults to now UTC)
        
    Returns:
        CleanupResult with statistics about the cleanup operation
    """
    import time
    start_time = time.time()
    
    result = CleanupResult()
    
    # Set current time
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    result.cleanup_timestamp = current_time.isoformat()
    
    print(f"[Cleanup] Starting cleanup at {result.cleanup_timestamp}")
    
    # Step 1: Get all expired chunk IDs from vector store
    try:
        expired_entries = vector_store.get_expired_ids(current_time)
        result.expired_chunks_found = len(expired_entries)
        
        if not expired_entries:
            print("[Cleanup] No expired chunks found")
            result.cleanup_time_seconds = time.time() - start_time
            return result
        
        print(f"[Cleanup] Found {len(expired_entries)} expired chunks")
        
    except Exception as e:
        error_info = {
            "stage": "get_expired_ids",
            "error": str(e),
            "error_type": type(e).__name__
        }
        result.errors.append(error_info)
        result.partial_failure = True
        print(f"[Cleanup] Error getting expired IDs: {e}")
        result.cleanup_time_seconds = time.time() - start_time
        return result
    
    # Step 2: Group expired chunks by document_id
    document_chunks: Dict[str, List[str]] = {}  # document_id -> [chunk_ids]
    orphan_chunk_ids: List[str] = []  # chunks without document_id
    
    for chunk_id, document_id in expired_entries:
        if document_id:
            if document_id not in document_chunks:
                document_chunks[document_id] = []
            document_chunks[document_id].append(chunk_id)
        else:
            orphan_chunk_ids.append(chunk_id)
    
    result.documents_affected = len(document_chunks)
    result.deleted_document_ids = list(document_chunks.keys())
    
    print(f"[Cleanup] Affected documents: {len(document_chunks)}")
    if orphan_chunk_ids:
        print(f"[Cleanup] Orphan chunks (no document_id): {len(orphan_chunk_ids)}")
    
    # Step 3: Delete chunks from vector store (grouped by document)
    all_chunk_ids_to_delete: List[str] = []
    
    for document_id, chunk_ids in document_chunks.items():
        all_chunk_ids_to_delete.extend(chunk_ids)
    
    all_chunk_ids_to_delete.extend(orphan_chunk_ids)
    
    if all_chunk_ids_to_delete:
        try:
            deleted_count = vector_store.delete_batch(all_chunk_ids_to_delete)
            result.chunks_deleted = deleted_count
            print(f"[Cleanup] Deleted {deleted_count} chunks from vector store")
            
        except Exception as e:
            error_info = {
                "stage": "delete_vector_chunks",
                "error": str(e),
                "error_type": type(e).__name__,
                "chunk_count": len(all_chunk_ids_to_delete)
            }
            result.errors.append(error_info)
            result.partial_failure = True
            print(f"[Cleanup] Error deleting vector chunks: {e}")
    
    # Step 4: Clean up context store (parent chunks)
    context_keys_to_delete: Set[str] = set()
    
    # Find context keys that belong to expired documents
    for parent_key, parent_data in list(context_store.items()):
        parent_doc_id = parent_data.get("document_id", "")
        
        if parent_doc_id in document_chunks:
            context_keys_to_delete.add(parent_key)
    
    # Delete context chunks
    for parent_key in context_keys_to_delete:
        try:
            del context_store[parent_key]
            result.context_chunks_deleted += 1
        except KeyError:
            pass  # Already deleted
        except Exception as e:
            error_info = {
                "stage": "delete_context_chunk",
                "parent_key": parent_key,
                "error": str(e),
                "error_type": type(e).__name__
            }
            result.errors.append(error_info)
            result.partial_failure = True
    
    if result.context_chunks_deleted > 0:
        print(f"[Cleanup] Deleted {result.context_chunks_deleted} context chunks")
    
    # Finalize
    result.cleanup_time_seconds = time.time() - start_time
    
    print(f"[Cleanup] Completed in {result.cleanup_time_seconds:.3f}s")
    print(f"[Cleanup] Summary: {result.chunks_deleted} vector chunks, "
          f"{result.context_chunks_deleted} context chunks, "
          f"{result.documents_affected} documents")
    
    if result.errors:
        print(f"[Cleanup] Warnings: {len(result.errors)} errors occurred")
    
    return result


def get_cleanup_status(
    vector_store: TempVectorStore,
    context_store: Dict[str, Dict[str, Any]],
    current_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get status of expired documents without performing cleanup.
    
    Useful for monitoring and deciding when to trigger cleanup.
    
    Args:
        vector_store: TempVectorStore instance
        context_store: Dictionary storing context chunks
        current_time: Time to check expiration against (defaults to now UTC)
        
    Returns:
        Dictionary with expiration statistics
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    try:
        expired_entries = vector_store.get_expired_ids(current_time)
        
        # Group by document
        document_ids: Set[str] = set()
        for chunk_id, document_id in expired_entries:
            if document_id:
                document_ids.add(document_id)
        
        # Count context chunks that would be deleted
        context_chunks_affected = 0
        for parent_key, parent_data in context_store.items():
            if parent_data.get("document_id", "") in document_ids:
                context_chunks_affected += 1
        
        return {
            "check_time": current_time.isoformat(),
            "expired_chunks": len(expired_entries),
            "expired_documents": len(document_ids),
            "expired_document_ids": list(document_ids),
            "context_chunks_affected": context_chunks_affected,
            "total_vector_chunks": vector_store.count(),
            "total_context_chunks": len(context_store),
            "cleanup_recommended": len(expired_entries) > 0
        }
        
    except Exception as e:
        return {
            "check_time": current_time.isoformat(),
            "error": str(e),
            "error_type": type(e).__name__,
            "cleanup_recommended": False
        }
