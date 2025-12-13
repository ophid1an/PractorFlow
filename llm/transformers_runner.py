import time
import os
from typing import Dict, Any, Optional, List

import torch
from transformers import AutoModel, AutoTokenizer

from llm.llm_config import LLMConfig
from llm.base.llm_runner import LLMRunner


class TransformersRunner(LLMRunner):
    """Runner for models using HuggingFace Transformers."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        os.makedirs(config.models_dir, exist_ok=True)

        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=config.models_dir,
        )
        
        print(f"Loading model {self.model_name}...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            cache_dir=config.models_dir,
            trust_remote_code=True,
        )
        self.model.eval()

        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_context_length = self.model.config.max_position_embeddings
        else:
            self.max_context_length = 4096

        print("Model loaded successfully!")

    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
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
        """
        start_time = time.time()

        # Build final prompt
        final_prompt = self._build_prompt(messages, prompt, instructions)

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        input_ids = self.tokenizer.encode(final_prompt, return_tensors="pt")
        input_ids = self._truncate_input(input_ids)
        input_length = input_ids.shape[1]

        input_ids = input_ids.to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=tp,
                use_cache=True,
            )

        generated_ids = output_ids[:, input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        completion_tokens = generated_ids.shape[1]
        total_tokens = input_length + completion_tokens

        latency = time.time() - start_time

        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "latency_seconds": latency,
        }

    def _build_prompt(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> str:
        """Build final prompt from messages/prompt and instructions."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")
        
        # Build from messages
        if messages is not None:
            return self._format_messages(messages, instructions)
        
        # Build from single prompt
        if instructions:
            return f"{instructions}\n\n{prompt}"
        return prompt

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        instructions: Optional[str] = None
    ) -> str:
        """Format messages into a prompt string.
        
        Override this in subclasses for model-specific chat templates.
        """
        parts = []
        
        # Add instructions if provided
        if instructions:
            parts.append(f"Instructions: {instructions}\n")
        
        # Add conversation
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        parts.append("Assistant:")
        return "\n".join(parts)

    def _truncate_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Truncate input from the left if it exceeds context window."""
        input_length = input_ids.shape[1]
        max_input_length = self.max_context_length - self.max_new_tokens

        if input_length > max_input_length:
            input_ids = input_ids[:, -max_input_length:]

        return input_ids

    def _get_temperature(self, temperature: float = None) -> float:
        """Get temperature value, falling back to config default."""
        return temperature if temperature is not None else self.config.temperature

    def _get_top_p(self, top_p: float = None) -> float:
        """Get top_p value, falling back to config default."""
        return top_p if top_p is not None else self.config.top_p