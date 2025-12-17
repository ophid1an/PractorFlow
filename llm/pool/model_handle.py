"""
ModelHandle - Wrapper for loaded model with metadata.

Provides unified interface for both llama.cpp and transformers models.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

from llm.llm_config import LLMConfig


@dataclass
class ModelHandle:
    """
    Handle to a loaded model.
    
    Wraps the actual model instance with metadata for pool management.
    Provides unified access regardless of backend.
    """
    
    config: LLMConfig
    backend: str
    model: Any
    tokenizer: Optional[Any] = None  # Only for transformers
    max_context_length: int = 4096
    config_hash: str = ""
    
    # Pool management metadata
    ref_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    
    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used_at = datetime.now()
    
    def acquire(self) -> None:
        """Increment reference count."""
        self.ref_count += 1
        self.touch()
    
    def release(self) -> None:
        """Decrement reference count."""
        if self.ref_count > 0:
            self.ref_count -= 1
    
    @property
    def is_in_use(self) -> bool:
        """Check if model is currently in use."""
        return self.ref_count > 0
    
    @property
    def is_llama_cpp(self) -> bool:
        """Check if this is a llama.cpp model."""
        return self.backend == "llama_cpp"
    
    @property
    def is_transformers(self) -> bool:
        """Check if this is a transformers model."""
        return self.backend == "transformers"
    
    def get_chat_template(self) -> Optional[str]:
        """Get chat template from model."""
        if self.is_llama_cpp:
            try:
                metadata = self.model.metadata
                if metadata and "tokenizer.chat_template" in metadata:
                    return metadata["tokenizer.chat_template"]
            except Exception:
                pass
        elif self.is_transformers:
            try:
                if hasattr(self.tokenizer, 'chat_template'):
                    return self.tokenizer.chat_template
            except Exception:
                pass
        return None
    
    def __repr__(self) -> str:
        return (
            f"ModelHandle(backend={self.backend}, "
            f"ref_count={self.ref_count}, "
            f"config_hash={self.config_hash[:8]}...)"
        )
