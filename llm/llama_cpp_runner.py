import time
import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from llm.llm_config import LLMConfig
from llm.base.llm_runner import LLMRunner
from llm.base.session import Session
from llm.document.embeddings import SentenceTransformerEmbeddingModel
from llm.document.vector_store_config import VectorStoreConfig
from llm.document.temp_vector_store import TempVectorStore
from llm.document.vector_cleanup import cleanup_expired_documents, get_cleanup_status, CleanupResult
from llm.document.document_loader import DocumentLoader
from llm.knowledge.knowledge_store import KnowledgeStore


class LlamaCppRunner(LLMRunner):
    """Runner for GGUF models using llama-cpp-python with Small-to-Big RAG."""

    def __init__(
        self,
        config: LLMConfig,
        session: Optional[Session] = None,
        knowledge_store: Optional["KnowledgeStore"] = None
    ):
        super().__init__(config, session, knowledge_store)
        
        os.makedirs(config.models_dir, exist_ok=True)
        model_path = self._resolve_model_path()

        print(f"[llama_cpp] Loading model: {model_path}")

        if not os.path.isfile(model_path):
            raise ValueError(f"[llama_cpp] Model path is not a file: {model_path}")

        with open(model_path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("[llama_cpp] File is not a valid GGUF file")

        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "embedding": False,
            "verbose": True,
        }
        
        if config.n_batch is not None:
            llama_kwargs["n_batch"] = config.n_batch

        self.model = Llama(**llama_kwargs)
        # Disable the prompt cache
        self.model.set_cache(None)
        if hasattr(self.model, 'n_ctx'):
            self.max_context_length = self.model.n_ctx()
        else:
            self.max_context_length = config.n_ctx

        print(f"[llama_cpp] Model loaded. Context window: {self.max_context_length}")
        
        # Initialize embedding model
        print("[llama_cpp] Initializing embedding model...")
        self.embedding_model = SentenceTransformerEmbeddingModel(
            model_name=config.embedding_model_name,
            cache_dir=config.models_dir
        )
        
        # Initialize vector store config
        vector_store_config = VectorStoreConfig(
            persist_directory=config.chroma_persist_dir,
            collection_name=config.chroma_collection_name,
            default_ttl_hours=config.default_ttl_hours,
            distance_metric=config.chroma_distance_metric,
            batch_size=config.chroma_batch_size
        )
        
        # Vector store for SMALL retrieval chunks (ChromaDB-based with TTL)
        print("[llama_cpp] Initializing ChromaDB vector store...")
        self.vector_store = TempVectorStore(
            config=vector_store_config,
            dimension=self.embedding_model.embedding_dimension
        )
        
        # Store for LARGE context chunks (parent chunks)
        self.context_store: Dict[str, Dict[str, Any]] = {}
        
        # Initialize document loader with Small-to-Big settings
        self._doc_loader = DocumentLoader(
            retrieval_chunk_size=config.retrieval_chunk_size,
            retrieval_chunk_overlap=config.retrieval_chunk_overlap,
            context_chunk_size=config.context_chunk_size,
            context_chunk_overlap=config.context_chunk_overlap,
        )
        
        print("[llama_cpp] Small-to-Big RAG with ChromaDB initialized")
        
        if self.knowledge_store:
            print("[llama_cpp] Knowledge store configured for on-demand loading")

    def _resolve_model_path(self) -> str:
        if self.config.local_model_path:
            if not os.path.exists(self.config.local_model_path):
                raise ValueError(f"[llama_cpp] Path does not exist: {self.config.local_model_path}")
            return self.config.local_model_path

        filename = os.path.basename(self.model_name)
        found = self._find_file_recursive(self.config.models_dir, filename)
        if found:
            self.config.local_model_path = found
            return found

        if self.model_name.count("/") < 1:
            raise ValueError("[llama_cpp] model_name must be 'repo_id/filename.gguf'")

        repo_id, filename = self.model_name.rsplit("/", 1)
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.config.models_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        self.config.local_model_path = model_path
        return model_path

    def _find_file_recursive(self, root_dir: str, filename: str) -> Optional[str]:
        for dirpath, _, filenames in os.walk(root_dir):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def load_document(
        self,
        filepath: str,
        ttl_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load document with Small-to-Big chunking.
        
        Args:
            filepath: Path to the document file
            ttl_hours: Optional TTL override (uses config default if not provided)
            
        Returns:
            Loaded document dict
        """
        print(f"[Document] Loading: {filepath}")
        
        document = self._doc_loader.load_file(filepath)
        doc_id = document["id"]
        
        # Store document in session
        if self.session:
            self.session.add_document(document)
        else:
            self.documents.append(document)
        
        # Get chunks
        retrieval_chunks = document.get("retrieval_chunks", [])
        context_chunks = document.get("context_chunks", [])
        
        # Store context chunks (parent chunks) in context_store
        for ctx_chunk in context_chunks:
            store_key = f"{doc_id}_{ctx_chunk['id']}"
            self.context_store[store_key] = {
                "text": ctx_chunk["text"],
                "metadata": ctx_chunk["metadata"],
                "document_id": doc_id,
            }
        
        # Embed and store retrieval chunks (small chunks) in vector store
        if retrieval_chunks:
            print(f"[Document] Embedding {len(retrieval_chunks)} retrieval chunks...")
            
            chunk_texts = [chunk["text"] for chunk in retrieval_chunks]
            embeddings = self.embedding_model.embed_batch(chunk_texts, show_progress=True)
            
            # Prepare metadata with document_id and parent references
            chunk_metadatas = []
            chunk_ids = []
            
            for chunk in retrieval_chunks:
                chunk_id = f"{doc_id}_{chunk['id']}"
                parent_key = f"{doc_id}_{chunk['parent_id']}"
                
                metadata = chunk["metadata"].copy()
                metadata["document_id"] = doc_id
                metadata["parent_key"] = parent_key
                
                chunk_metadatas.append(metadata)
                chunk_ids.append(chunk_id)
            
            # Batch add to vector store with TTL
            self.vector_store.add_batch(
                texts=chunk_texts,
                vectors=embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids,
                ttl_hours=ttl_hours,
                document_id=doc_id
            )
        
        print(f"[Document] Loaded: {document['filename']}")
        print(f"[Document] {len(retrieval_chunks)} retrieval chunks → {len(context_chunks)} context chunks")
        
        return document

    def load_document_from_knowledge(
        self,
        document_id: str,
        ttl_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a document from knowledge store into temporary context.
        
        Fetches pre-computed chunks and embeddings from KnowledgeStore
        and loads them into the temporary vector store and context store.
        
        Args:
            document_id: Document ID in the knowledge store
            ttl_hours: Optional TTL override for temporary storage
            
        Returns:
            Document info dict with loading statistics
            
        Raises:
            ValueError: If knowledge store is not configured or document not found
        """
        if not self.knowledge_store:
            raise ValueError("Knowledge store is not configured")
        
        print(f"[Knowledge] Loading document from knowledge store: {document_id}")
        
        # Get document metadata
        document = self.knowledge_store.get_document(document_id)
        if not document:
            raise ValueError(f"Document not found in knowledge store: {document_id}")
        
        doc_metadata = document.get("metadata", {})
        filename = doc_metadata.get("filename", "unknown")
        
        print(f"[Knowledge] Found document: {filename}")
        
        # Get retrieval chunks with embeddings
        retrieval_chunks = self.knowledge_store.get_retrieval_chunks_by_document(document_id)
        
        # Get context chunks
        context_chunks = self.knowledge_store.get_context_chunks_by_document(document_id)
        
        print(f"[Knowledge] Retrieved {len(retrieval_chunks)} retrieval chunks, {len(context_chunks)} context chunks")
        
        # Load context chunks into context_store
        context_loaded = 0
        for ctx_chunk in context_chunks:
            ctx_id = ctx_chunk["id"]  # Already prefixed with doc_id from knowledge store
            self.context_store[ctx_id] = {
                "text": ctx_chunk["text"],
                "metadata": ctx_chunk.get("metadata", {}),
                "document_id": document_id,
            }
            context_loaded += 1
        
        # Load retrieval chunks into vector store
        retrieval_loaded = 0
        if retrieval_chunks:
            chunk_texts = []
            chunk_vectors = []
            chunk_metadatas = []
            chunk_ids = []
            
            for chunk in retrieval_chunks:
                # Skip chunks without embeddings
                if "vector" not in chunk or chunk["vector"] is None:
                    print(f"[Knowledge] Warning: Chunk {chunk['id']} has no embedding, skipping")
                    continue
                
                chunk_id = chunk["id"]  # Already prefixed with doc_id
                chunk_texts.append(chunk["text"])
                chunk_vectors.append(chunk["vector"])
                
                metadata = chunk.get("metadata", {}).copy()
                metadata["document_id"] = document_id
                metadata["source"] = "knowledge_store"
                chunk_metadatas.append(metadata)
                chunk_ids.append(chunk_id)
            
            if chunk_vectors:
                # Stack vectors into numpy array
                vectors_array = np.vstack(chunk_vectors)
                
                # Batch add to vector store with TTL
                self.vector_store.add_batch(
                    texts=chunk_texts,
                    vectors=vectors_array,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids,
                    ttl_hours=ttl_hours,
                    document_id=document_id
                )
                retrieval_loaded = len(chunk_ids)
        
        print(f"[Knowledge] Loaded into temp storage: {retrieval_loaded} retrieval chunks, {context_loaded} context chunks")
        
        return {
            "document_id": document_id,
            "filename": filename,
            "retrieval_chunks_loaded": retrieval_loaded,
            "context_chunks_loaded": context_loaded,
            "ttl_hours": ttl_hours or self.config.default_ttl_hours,
        }

    def cleanup_expired_documents(self) -> CleanupResult:
        """
        Clean up expired documents from vector store and context store.
        
        Should be called by external scheduler (cron, celery, etc.).
        
        Returns:
            CleanupResult with cleanup statistics
        """
        print("[llama_cpp] Running document cleanup...")
        
        result = cleanup_expired_documents(
            vector_store=self.vector_store,
            context_store=self.context_store
        )
        
        return result
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """
        Get status of expired documents without performing cleanup.
        
        Returns:
            Dictionary with expiration statistics
        """
        return get_cleanup_status(
            vector_store=self.vector_store,
            context_store=self.context_store
        )

    def unload_document(self, document_id: str) -> bool:
        """
        Unload a document from temporary context.
        
        Removes the document's chunks from vector store.
        Use for documents loaded via load_document().
        
        Args:
            document_id: Document ID to unload
            
        Returns:
            True if document was found and unloaded, False otherwise
        """
        deleted_chunks = self.vector_store.delete_by_document_id(document_id)
        return deleted_chunks > 0

    def unload_document_from_knowledge(self, document_id: str) -> bool:
        """
        Unload a document from temporary context.
        
        Removes the document's chunks from context store only.
        Vector store chunks have TTL and will expire automatically.
        Use for documents loaded via load_document_from_knowledge().
        
        Args:
            document_id: Document ID to unload
            
        Returns:
            True if document was found and unloaded, False otherwise
        """
        context_keys_to_remove = [
            key for key, data in self.context_store.items()
            if data.get("document_id") == document_id
        ]
        
        for key in context_keys_to_remove:
            del self.context_store[key]
        
        return len(context_keys_to_remove) > 0

    def get_chat_reply_structure(self) -> Optional[str]:
        """Get the chat template that describes the structure of the reply field.
        
        Returns:
            The chat template string from the model's GGUF metadata,
            or None if not available.
        """
        try:
            metadata = self.model.metadata
            if metadata and "tokenizer.chat_template" in metadata:
                return metadata["tokenizer.chat_template"]
        except Exception:
            pass
        return None

    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
    ) -> Dict[str, Any]:
        """Generate with Small-to-Big retrieval using chat completion API."""
        start_time = time.time()

        context = None
        context_sources = []
        
        if self.context_enabled:
            query = self._extract_query(messages, prompt)
            
            if query:
                print(f"[Retrieval] Searching for relevant chunks...")
                context, context_sources = self._small_to_big_retrieve(
                    query=query,
                    max_chunks=self.config.max_retrieval_chunks,
                )
                
                if context:
                    print(f"[Retrieval] Retrieved {len(context_sources)} context chunks ({len(context)} chars)")

        # Build chat messages for the API
        chat_messages = self._build_chat_messages(messages, prompt, instructions, context)

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        print("[llama_cpp] Generating...")
        
        # Build completion kwargs
        completion_kwargs = {
            "messages": chat_messages,
            "temperature": temp,
            "top_p": tp,
            "max_tokens": self.max_new_tokens,
        }
        
        # Add stop tokens if configured
        if self.config.stop_tokens:
            completion_kwargs["stop"] = self.config.stop_tokens
        
        # Use chat completion API
        result = self.model.create_chat_completion(**completion_kwargs)

        latency = time.time() - start_time

        response = {
            "reply": result,
            "latency_seconds": latency,
        }
        
        if context:
            response["context_used"] = context
            response["context_sources"] = context_sources
        
        return response

    def _build_chat_messages(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages array for create_chat_completion API."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")
        
        chat_messages = []
        
        # Build system message from instructions and context
        system_parts = []
        if instructions:
            system_parts.append(instructions)
        if context:
            system_parts.append(f"---CONTEXT---\n{context}\n---END CONTEXT---")
            system_parts.append("Use the above context to answer the user's question when relevant.")
        
        if system_parts:
            chat_messages.append({
                "role": "system",
                "content": "\n\n".join(system_parts)
            })
        
        # Add conversation messages
        if messages is not None:
            for msg in messages:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        else:
            # Single prompt becomes a user message
            chat_messages.append({
                "role": "user",
                "content": prompt
            })
        
        return chat_messages

    def _small_to_big_retrieve(
        self,
        query: str,
        max_chunks: int = 10,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Small-to-Big retrieval:
        1. Search vector store for similar SMALL chunks
        2. Get unique PARENT (context) chunks
        3. Return parent chunks for LLM context
        """
        if self.vector_store.count() == 0:
            return None, []
        
        # Step 1: Embed query and search for small chunks
        query_vector = self.embedding_model.embed(query)
        
        # Retrieve more small chunks than needed (we'll dedupe by parent)
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=max_chunks * 3  # Get more to account for deduplication
        )
        
        if not results:
            return None, []
        
        # Step 2: Get unique parent chunks, preserving order by best similarity
        seen_parents = set()
        parent_chunks = []
        
        for chunk_id, similarity, text, metadata in results:
            parent_key = metadata.get("parent_key")
            
            if not parent_key or parent_key in seen_parents:
                continue
            
            seen_parents.add(parent_key)
            
            # Get parent chunk from context store
            parent_data = self.context_store.get(parent_key)
            if parent_data:
                parent_chunks.append({
                    "parent_key": parent_key,
                    "text": parent_data["text"],
                    "metadata": parent_data["metadata"],
                    "best_similarity": similarity,  # Similarity of best child
                })
            
            # Stop if we have enough parent chunks
            if len(parent_chunks) >= max_chunks:
                break
        
        if not parent_chunks:
            return None, []
        
        # Step 3: Build context from parent chunks
        context_parts = []
        sources = []
        
        for parent in parent_chunks:
            text = parent["text"]
            filename = parent["metadata"].get("filename", "unknown")
            context_parts.append(f"[{filename}]\n{text}")
            
            sources.append({
                "parent_key": parent["parent_key"],
                "filename": filename,
                "similarity": parent["best_similarity"],
                "chars": len(text),
            })
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _extract_query(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        if messages:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    return msg["content"]
        return prompt

    @property
    def context_enabled(self) -> bool:
        return self.vector_store.count() > 0