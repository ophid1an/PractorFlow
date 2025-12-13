from llm.base.llm_runner import LLMRunner
from llm.llm_config import LLMConfig
from llm.transformers_runner import TransformersRunner
from llm.llama_cpp_runner import LlamaCppRunner


def create_runner(config: LLMConfig) -> LLMRunner:
    """Factory function to create appropriate model runner.

    Args:
        config: LLMConfig instance with model parameters

    Returns:
        Appropriate LLMRunner subclass instance

    Raises:
        ValueError: If backend is not supported
    """
    backend = config.backend

    if backend == "llama_cpp":
        return LlamaCppRunner(config)
    elif backend == "transformers":
        return TransformersRunner(config)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: llama_cpp, transformers"
        )
