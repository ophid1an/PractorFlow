"""
LLM module with tool-based RAG support and model pooling.

Provides:
- Model pooling for API server usage (ModelPool, ModelHandle)
- LLM runners (llama.cpp, transformers)
- Knowledge store for document storage
- Tool registry for extensible tool support
- Streaming support
"""

from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.base.llm_runner import LLMRunner, StreamChunk
from practorflow.llm.pool.model_handle import ModelHandle
from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.transformers_runner import TransformersRunner
from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from practorflow.llm.factory import create_runner

__all__ = [
    "LLMConfig",
    "LLMRunner",
    "StreamChunk",
    "ModelHandle",
    "ModelPool",
    "TransformersRunner",
    "LlamaCppRunner",
    "create_runner",
]