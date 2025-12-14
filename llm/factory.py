from typing import Optional

from llm.base.llm_runner import LLMRunner
from llm.base.session import Session
from llm.llm_config import LLMConfig
from llm.transformers_runner import TransformersRunner
from llm.llama_cpp_runner import LlamaCppRunner


def create_runner(
    config: LLMConfig, 
    session: Optional[Session] = None
) -> LLMRunner:
    """Factory function to create appropriate model runner with optional session.

    Args:
        config: LLMConfig instance with model parameters
        session: Optional Session object for context and document management

    Returns:
        Appropriate LLMRunner subclass instance with integrated session

    Raises:
        ValueError: If backend is not supported
    """
    backend = config.backend

    if backend == "llama_cpp":
        return LlamaCppRunner(config, session=session)
    elif backend == "transformers":
        return TransformersRunner(config, session=session)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: llama_cpp, transformers"
        )