from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class LLMConfig:
    """Configuration for LLM runners supporting multiple backends."""

    # Model settings
    model_name: str = "Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_k_m.gguf"
    device: str = "cpu"
    dtype: torch.dtype = torch.bfloat16
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    quantization: Optional[str] = None

    # Paths
    local_model_path: Optional[str] = None
    models_dir: str = "./models"

    # Llama.cpp settings
    n_gpu_layers: int = -1
    n_ctx: int = 32768
    n_batch: int = 2048

    # Backend
    backend: Optional[str] = "llama_cpp"

    # RAG / Retrieval settings
    max_retrieval_chunks: int = 10
    chars_per_token: int = 3
    reserved_tokens: int = 4096

    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Small-to-Big Chunking settings
    # Small chunks for retrieval (embedding/search)
    retrieval_chunk_size: int = 128
    retrieval_chunk_overlap: int = 20
    
    # Large chunks for generation (context to LLM)
    context_chunk_size: int = 1024
    context_chunk_overlap: int = 100