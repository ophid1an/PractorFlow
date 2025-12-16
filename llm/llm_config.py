from dataclasses import dataclass, field
from typing import List, Optional
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

    # Generation settings
    stop_tokens: Optional[List[str]] = None

    # Knowledge search settings
    max_search_results: int = 5