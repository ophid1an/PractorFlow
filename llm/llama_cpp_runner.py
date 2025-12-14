import time
import os
from typing import Dict, Any, Optional, List

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from llm.llm_config import LLMConfig
from llm.base.llm_runner import LLMRunner
from llm.base.session import Session
from llm.document.embeddings import LLMEmbeddingModel
from llm.document.memory_vector_store import MemoryVectorStore


class LlamaCppRunner(LLMRunner):
    """Runner for GGUF models using llama-cpp-python with RAG support."""

    def __init__(self, config: LLMConfig, session: Optional[Session] = None):
        """
        Initialize LlamaCpp runner with optional session.
        
        Args:
            config: LLM configuration
            session: Optional Session object for context and document management
        """
        super().__init__(config, session)
        
        os.makedirs(config.models_dir, exist_ok=True)

        model_path = self._resolve_model_path()

        print(f"[llama_cpp] FINAL model_path = {model_path}")

        if not os.path.isfile(model_path):
            raise ValueError(f"[llama_cpp] Model path is not a file: {model_path}")

        # Quick GGUF sanity check
        with open(model_path, "rb") as f:
            magic = f.read(4)

        if magic not in (b"GGUF",):
            raise ValueError("[llama_cpp] File is not a valid GGUF file")

        # Build kwargs - always pass n_ctx to override model's default
        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "embedding": True,  # Required for RAG embeddings
            "verbose": True,
        }
        
        print(f"[llama_cpp] Using n_ctx = {config.n_ctx}")
        
        # Only add n_batch if explicitly set by user
        if config.n_batch is not None:
            llama_kwargs["n_batch"] = config.n_batch

        self.model = Llama(**llama_kwargs)

        # Get actual context length
        if hasattr(self.model, 'n_ctx'):
            self.max_context_length = self.model.n_ctx()
            print(f"[llama_cpp] Model context window: {self.max_context_length}")
        else:
            self.max_context_length = None
            print("[llama_cpp] Model context window: unlimited")

        print("[llama_cpp] Model loaded successfully")
        
        # Initialize embedding model and vector store
        print("[llama_cpp] Initializing embedding model...")
        self.embedding_model = LLMEmbeddingModel(self)
        
        print("[llama_cpp] Initializing vector store...")
        self.vector_store = MemoryVectorStore(dimension=self.embedding_model.embedding_dimension)
        
        print("[llama_cpp] RAG components initialized")

    def _resolve_model_path(self) -> str:

        # 1️⃣ Explicit local_model_path
        if self.config.local_model_path:
            if not os.path.exists(self.config.local_model_path):
                raise ValueError(
                    f"[llama_cpp] local_model_path does not exist: "
                    f"{self.config.local_model_path}"
                )
            return self.config.local_model_path

        # 2️⃣ Search models_dir recursively (HF cache layout)
        filename = os.path.basename(self.model_name)
        found = self._find_file_recursive(self.config.models_dir, filename)
        if found:
            self.config.local_model_path = found
            return found

        if self.model_name.count("/") < 1:
            raise ValueError(
                "[llama_cpp] Cannot download model.\n"
                "model_name must be '<repo_id>/<filename>.gguf'\n"
                "Example:\n"
                "mistralai/Ministral-3-3B-Instruct-2512-GGUF/"
                "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
            )

        repo_id, filename = self.model_name.rsplit("/", 1)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.config.models_dir,
            token=os.environ.get("HF_TOKEN"),
        )

        print(f"[llama_cpp] Downloaded to: {model_path}")

        self.config.local_model_path = model_path
        return model_path

    def _find_file_recursive(self, root_dir: str, filename: str) -> Optional[str]:
        for dirpath, _, filenames in os.walk(root_dir):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def load_document(self, filepath: str) -> Dict[str, Any]:
        """
        Load a document, chunk it, embed chunks, and store in vector store.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Loaded document dict with chunks
        """
        print(f"[Document] Loading: {filepath}")
        
        # Load and chunk document
        document = self._document_loader.load_file(filepath)
        
        # Add to session if available
        if self.session:
            self.session.add_document(document)
        else:
            existing_ids = {doc["id"] for doc in self.documents}
            if document["id"] in existing_ids:
                self.documents = [doc if doc["id"] != document["id"] else document 
                                 for doc in self.documents]
            else:
                self.documents.append(document)
        
        # Embed and store chunks
        chunks = document.get("chunks", [])
        if chunks:
            print(f"[Document] Embedding {len(chunks)} chunks...")
            
            # Extract chunk texts
            chunk_texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_batch(chunk_texts, show_progress=True)
            
            # Store in vector store with metadata
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document['id']}_chunk_{i}"
                
                # Add document_id to metadata for filtering
                metadata = chunk["metadata"].copy()
                metadata["document_id"] = document["id"]
                
                self.vector_store.add(
                    text=chunk["text"],
                    vector=embeddings[i],
                    metadata=metadata,
                    id=chunk_id
                )
                chunk_ids.append(chunk_id)
            
            print(f"[Document] Stored {len(chunk_ids)} chunks in vector store")
        
        print(f"[Document] Loaded: {document['filename']} ({document['file_type']})")
        return document

    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        use_context: bool = True,
        max_context_docs: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text from messages or prompt with RAG support.

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
            Dictionary with keys: text, usage, latency_seconds, context_used (if context was added)
        """
        start_time = time.time()

        # Retrieve relevant context if enabled
        context = None
        context_sources = []
        
        if use_context and self.context_enabled:
            query = self._extract_query(messages, prompt)
            
            if query:
                print(f"[Context] Searching vector store...")
                context, context_sources = self._retrieve_context(
                    query, 
                    max_docs=max_context_docs,
                    max_chars=max_context_chars
                )
                
                if context:
                    print(f"[Context] Retrieved {len(context_sources)} chunks ({len(context)} chars)")

        # Build final prompt (with context if retrieved)
        final_prompt = self._build_prompt(messages, prompt, instructions, context)

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        print("[llama_cpp] Generating")
        print(f"[llama_cpp] temperature = {temp}")
        print(f"[llama_cpp] top_p = {tp}")
        
        # Build generation kwargs
        gen_kwargs = {
            "temperature": temp,
            "top_p": tp,
        }
        
        # Only add max_tokens if explicitly set
        if self.max_new_tokens is not None:
            gen_kwargs["max_tokens"] = self.max_new_tokens
            print(f"[llama_cpp] max_new_tokens = {self.max_new_tokens}")
        else:
            print("[llama_cpp] max_new_tokens = unlimited")

        result = self.model(final_prompt, **gen_kwargs)

        text = result["choices"][0]["text"]

        usage = result.get("usage", {})
        latency = time.time() - start_time

        response = {
            "text": text,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "latency_seconds": latency,
        }
        
        # Add context information if used
        if context:
            response["context_used"] = context
            response["context_sources"] = context_sources
        
        return response

    def _extract_query(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Extract the user query for context retrieval."""
        if messages:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    return msg["content"]
        elif prompt:
            return prompt
        return None

    def _retrieve_context(
        self,
        query: str,
        max_docs: Optional[int] = None,
        max_chars: Optional[int] = None
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Retrieve relevant context from vector store using semantic search.
        
        Args:
            query: User query
            max_docs: Maximum number of chunks to retrieve (default: None = unlimited)
            max_chars: Maximum total characters (default: None = unlimited)
            
        Returns:
            Tuple of (context_string, list_of_source_dicts)
        """
        if self.vector_store.count() == 0:
            return None, []
        
        # Embed the query
        query_vector = self.embedding_model.embed(query)
        
        # Search vector store
        # Use large top_k if unlimited, otherwise use max_docs
        top_k = max_docs if max_docs is not None else 20
        
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )
        
        if not results:
            return None, []
        
        # Build context from results
        context_parts = []
        sources = []
        total_chars = 0
        
        for chunk_id, similarity, text, metadata in results:
            # Apply max_chars limit if specified
            if max_chars is not None and total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 0:
                    text = text[:remaining]
                else:
                    break
            
            # Format chunk with metadata
            filename = metadata.get("filename", "unknown")
            page = metadata.get("page", "?")
            context_parts.append(f"[{filename} - Page {page}]\n{text}")
            
            sources.append({
                "chunk_id": chunk_id,
                "filename": filename,
                "similarity": similarity,
                "page": page
            })
            
            total_chars += len(text)
            
            if max_chars is not None and total_chars >= max_chars:
                break
        
        if not context_parts:
            return None, []
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _build_prompt(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Build final prompt from messages/prompt, instructions, and context."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")
        
        # Build from messages
        if messages is not None:
            return self._format_messages(messages, instructions, context)
        
        # Build from single prompt
        parts = []
        
        if instructions:
            parts.append(instructions)
        
        if context:
            parts.append(f"\n---DOCUMENT CONTEXT---\n{context}\n---END CONTEXT---\n")
        
        parts.append(prompt)
        
        return "\n\n".join(parts)

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Format messages into a prompt string with optional context."""
        parts = []
        
        if instructions:
            parts.append(f"Instructions: {instructions}\n")
        
        if context:
            parts.append(f"---DOCUMENT CONTEXT---")
            parts.append(context)
            parts.append(f"---END CONTEXT---\n")
            parts.append("Use the above context to answer the user's question when relevant.\n")
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        parts.append("Assistant:")
        return "\n".join(parts)
    
    @property
    def context_enabled(self) -> bool:
        """Check if context is enabled (has chunks in vector store)."""
        return self.vector_store.count() > 0