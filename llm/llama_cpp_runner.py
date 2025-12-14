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
        super().__init__(config, session)

        os.makedirs(config.models_dir, exist_ok=True)

        model_path = self._resolve_model_path()

        print(f"[llama_cpp] FINAL model_path = {model_path}")

        if not os.path.isfile(model_path):
            raise ValueError(f"[llama_cpp] Model path is not a file: {model_path}")

        with open(model_path, "rb") as f:
            magic = f.read(4)

        if magic not in (b"GGUF",):
            raise ValueError("[llama_cpp] File is not a valid GGUF file")

        # 🔹 Generation model (NO embeddings)
        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "embedding": False,
            "verbose": True,
        }

        print(f"[llama_cpp] Using n_ctx = {config.n_ctx}")

        if config.n_batch is not None:
            llama_kwargs["n_batch"] = config.n_batch

        self.model = Llama(**llama_kwargs)

        if hasattr(self.model, 'n_ctx'):
            self.max_context_length = self.model.n_ctx()
            print(f"[llama_cpp] Model context window: {self.max_context_length}")
        else:
            self.max_context_length = None
            print("[llama_cpp] Model context window: unlimited")

        print("[llama_cpp] Generation model loaded successfully")

        # 🔹 Lazy-init embedding components
        self.embedding_model = None
        self.vector_store = None
        self._embedding_llama = None

    def _create_embedding_components(self):
        if self.embedding_model is not None:
            return

        print("[llama_cpp] Initializing embedding-only model...")

        llama_kwargs = {
            "model_path": self.config.local_model_path,
            "n_ctx": self.config.n_ctx,
            "n_gpu_layers": self.config.n_gpu_layers,
            "embedding": True,
            "verbose": True,
        }

        if self.config.n_batch is not None:
            llama_kwargs["n_batch"] = self.config.n_batch

        self._embedding_llama = Llama(**llama_kwargs)

        embedding_runner = type(self)(
            self.config,
            session=None
        )
        embedding_runner.model = self._embedding_llama

        self.embedding_model = LLMEmbeddingModel(embedding_runner)
        self.vector_store = MemoryVectorStore(
            dimension=self.embedding_model.embedding_dimension
        )

        print("[llama_cpp] Embedding components initialized")

    def _resolve_model_path(self) -> str:
        if self.config.local_model_path:
            if not os.path.exists(self.config.local_model_path):
                raise ValueError(
                    f"[llama_cpp] local_model_path does not exist: "
                    f"{self.config.local_model_path}"
                )
            return self.config.local_model_path

        filename = os.path.basename(self.model_name)
        found = self._find_file_recursive(self.config.models_dir, filename)
        if found:
            self.config.local_model_path = found
            return found

        if self.model_name.count("/") < 1:
            raise ValueError(
                "[llama_cpp] Cannot download model.\n"
                "model_name must be '<repo_id>/<filename>.gguf'"
            )

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

    def load_document(self, filepath: str) -> Dict[str, Any]:
        print(f"[Document] Loading: {filepath}")

        document = self._document_loader.load_file(filepath)

        if self.session:
            self.session.add_document(document)
        else:
            self.documents.append(document)

        chunks = document.get("chunks", [])
        if not chunks:
            return document

        # 🔹 Initialize embeddings ONLY now
        self._create_embedding_components()

        print(f"[Document] Embedding {len(chunks)} chunks...")

        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(chunk_texts)

        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"].copy()
            metadata["document_id"] = document["id"]

            self.vector_store.add(
                text=chunk["text"],
                vector=embeddings[i],
                metadata=metadata,
                id=f"{document['id']}_chunk_{i}",
            )

        print(f"[Document] Stored {len(chunks)} chunks in vector store")
        return document

    @property
    def context_enabled(self) -> bool:
        return self.vector_store is not None and self.vector_store.count() > 0
