from typing import Optional

from practorflow.llm.base.llm_runner import LLMRunner
from practorflow.llm.pool.model_handle import ModelHandle
from practorflow.llm.transformers_runner import TransformersRunner
from practorflow.llm.llama_cpp_runner import LlamaCppRunner
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore


def create_runner(
    handle: ModelHandle,
    knowledge_store: Optional[KnowledgeStore] = None
) -> LLMRunner:
    """
    Factory function to create appropriate model runner from a pooled model handle.

    Args:
        handle: ModelHandle from ModelPool with loaded model
        knowledge_store: Optional KnowledgeStore for document search

    Returns:
        Appropriate LLMRunner subclass instance

    Raises:
        ValueError: If backend is not supported
    """
    backend = handle.backend

    if backend == "llama_cpp":
        return LlamaCppRunner(handle, knowledge_store=knowledge_store)
    elif backend == "transformers":
        return TransformersRunner(handle, knowledge_store=knowledge_store)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: llama_cpp, transformers"
        )