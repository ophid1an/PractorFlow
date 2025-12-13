from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class LLMConfig:
    """Configuration for LLM runners supporting multiple backends."""

    model_name: str = "Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_k_m.gguf"
    device: str = "cpu"
    dtype: torch.dtype = torch.bfloat16
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    quantization: Optional[str] = None

    local_model_path: Optional[str] = None
    models_dir: str = "./models"

    n_gpu_layers: int = 0  # CPU-only
    n_ctx: int = 4096
    n_batch: int = 512

    backend: Optional[str] = "llama_cpp"

    def __post_init__(self):
        if self.backend is None:
            self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        return "llama_cpp"