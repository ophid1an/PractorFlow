from dataclasses import dataclass
from typing import List, Optional, Union
import torch


@dataclass
class LLMConfig:
    """
    Configuration for LLM runners supporting multiple backends.
    
    Supports:
    - llama.cpp backend (GGUF models)
    - transformers backend (HuggingFace models including GPT-OSS)
    """

    # Model settings
    # model_name: str = "bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    model_name: str = "Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_k_m.gguf"
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device
    dtype: Union[torch.dtype, str, None] = "auto"  # torch.dtype, "auto", or None
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    # Quantization (for transformers backend)
    quantization: Optional[str] = None  # "4bit", "8bit", or None

    # Paths
    models_dir: str = "../models"

    # Llama.cpp specific settings
    n_gpu_layers: int = -1
    n_ctx: int = 32768
    n_batch: int = 2048

    # Backend selection
    backend: Optional[str] = "llama_cpp"  # "llama_cpp" or "transformers"

    # Generation settings
    stop_tokens: Optional[List[str]] = None

    # Knowledge search settings
    max_search_results: int = 5

    # Transformers optimization settings (speed over memory)
    use_torch_compile: bool = True  # Enable torch.compile() for faster inference
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    warmup_on_load: bool = True  # Run warmup inference after loading