from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from llm.llm_config import LLMConfig


class LLMRunner(ABC):
    """Abstract base class for LLM runners."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self.device = config.device
        self.dtype = config.dtype
        self.max_new_tokens = config.max_new_tokens

    @abstractmethod
    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate text from messages or prompt.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (for backward compatibility)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)

        Returns:
            Dictionary with keys: text, usage, latency_seconds
            
        Note: Either messages or prompt must be provided, not both.
        """
        pass

    def _get_temperature(self, temperature: float = None) -> float:
        """Get temperature value, falling back to config default."""
        return temperature if temperature is not None else self.config.temperature

    def _get_top_p(self, top_p: float = None) -> float:
        """Get top_p value, falling back to config default."""
        return top_p if top_p is not None else self.config.top_p