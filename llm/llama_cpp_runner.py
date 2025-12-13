import time
import os
from typing import Dict, Any, Optional, List

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from llm.llm_config import LLMConfig
from llm.base.llm_runner import LLMRunner


class LlamaCppRunner(LLMRunner):
    """Runner for GGUF models using llama-cpp-python with verbose debug."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        os.makedirs(config.models_dir, exist_ok=True)

        model_path = self._resolve_model_path()

        print(f"[llama_cpp] FINAL model_path = {model_path}")

        if not os.path.isfile(model_path):
            raise ValueError(f"[llama_cpp] Model path is not a file: {model_path}")

        file_size = os.path.getsize(model_path)

        # Quick GGUF sanity check
        with open(model_path, "rb") as f:
            magic = f.read(4)

        if magic not in (b"GGUF",):
            raise ValueError("[llama_cpp] File is not a valid GGUF file")

        self.model = Llama(
            model_path=model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=True,
        )

        self.max_context_length = config.n_ctx

        print("[llama_cpp] Model loaded successfully")

    def _resolve_model_path(self) -> str:

        # 1️⃣ Explicit local_model_path
        if self.config.local_model_path:
            if not os.path.exists(self.config.local_model_path):
                raise ValueError(
                    f"[llama_cpp] local_model_path does not exist: "
                    f"{self.config.local_model_path}"
                )
            return self.config.local_model_path

        # 2️⃣ Search models_dir recursively (HF cache layout)
        filename = os.path.basename(self.model_name)
        found = self._find_file_recursive(self.config.models_dir, filename)
        if found:
            self.config.local_model_path = found
            return found

        if self.model_name.count("/") < 1:
            raise ValueError(
                "[llama_cpp] Cannot download model.\n"
                "model_name must be '<repo_id>/<filename>.gguf'\n"
                "Example:\n"
                "mistralai/Ministral-3-3B-Instruct-2512-GGUF/"
                "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
            )

        repo_id, filename = self.model_name.rsplit("/", 1)

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.config.models_dir,
            token=os.environ.get("HF_TOKEN"),
        )

        print(f"[llama_cpp] Downloaded to: {model_path}")

        self.config.local_model_path = model_path
        return model_path

    def _find_file_recursive(self, root_dir: str, filename: str) -> Optional[str]:
        for dirpath, _, filenames in os.walk(root_dir):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

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

        print("[llama_cpp] Generating")
        print(f"[llama_cpp] temperature = {temp}")
        print(f"[llama_cpp] top_p = {tp}")
        print(f"[llama_cpp] max_new_tokens = {self.max_new_tokens}")

        result = self.model(
            final_prompt,
            max_tokens=self.max_new_tokens,
            temperature=temp,
            top_p=tp,
        )

        text = result["choices"][0]["text"]

        usage = result.get("usage", {})
        latency = time.time() - start_time

        return {
            "text": text,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
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

