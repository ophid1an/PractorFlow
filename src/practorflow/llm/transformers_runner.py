import asyncio
import time
from threading import Thread
from typing import Dict, Any, Optional, List, AsyncIterator

import torch
from transformers import TextIteratorStreamer

from practorflow.llm.base.llm_runner import LLMRunner, StreamChunk
from practorflow.llm.pool.model_handle import ModelHandle
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger(
    "transformers_runner", level=appConfiguration.LoggerConfiguration.RunnerLevel
)


class TransformersRunner(LLMRunner):
    """Async runner for HuggingFace Transformers models with optimizations."""

    def __init__(
        self, handle: ModelHandle, knowledge_store: Optional[KnowledgeStore] = None
    ):
        if not handle.is_transformers:
            raise ValueError(f"Expected transformers handle, got {handle.backend}")

        super().__init__(handle, knowledge_store)

        if hasattr(self.model, "device"):
            self._device = self.model.device
        else:
            self._device = next(self.model.parameters()).device

        # Pre-compute generation defaults to avoid repeated attribute access
        self._pad_token_id = self.tokenizer.pad_token_id
        self._eos_token_id = self.tokenizer.eos_token_id

        logger.info(
            f"[TransformersRunner] Initialized with pooled model: {self.model_name}"
        )
        logger.info(f"[TransformersRunner] Device: {self._device}")

    def supports_function_calling(self) -> bool:
        """
        Check if the transformers model supports native function calling.

        Returns:
            True if model supports native function calling, False otherwise
        """
        try:
            if self.tokenizer and hasattr(self.tokenizer, "chat_template"):
                chat_template = self.tokenizer.chat_template
                if chat_template:
                    template_lower = str(chat_template).lower()
                    if any(
                        keyword in template_lower
                        for keyword in [
                            "tool",
                            "function",
                            "<tool_call>",
                            "<function_call>",
                            "tools",
                            "functions",
                            "tool_use",
                            "function_use",
                        ]
                    ):
                        logger.info(
                            f"[TransformersRunner] Model supports function calling (detected in chat template)"
                        )
                        return True

            if hasattr(self.model, "config"):
                config = self.model.config
                if hasattr(config, "to_dict"):
                    config_dict = config.to_dict()
                    if config_dict.get("supports_function_calling", False):
                        logger.info(
                            f"[TransformersRunner] Model supports function calling (config flag)"
                        )
                        return True

            logger.info(
                f"[TransformersRunner] Model does NOT support native function calling"
            )
            return False

        except Exception as e:
            logger.warning(
                f"[TransformersRunner] Error detecting function calling support: {e}"
            )
            return False

    def _build_chat_messages(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages for the model."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")

        chat_messages = []
        system_parts = []

        if instructions:
            system_parts.append(instructions)

        if context:
            system_parts.append(
                "You have access to the following REFERENCE DOCUMENTS. "
                "Use ONLY this information to answer the user's question. "
                "If the answer is in the documents, provide it. "
                "Quote or reference specific parts when relevant."
            )
            system_parts.append(f"REFERENCE DOCUMENTS:\n{context}\nEND OF DOCUMENTS.")

        if system_parts:
            chat_messages.append(
                {"role": "system", "content": "\n\n".join(system_parts)}
            )

        if messages is not None:
            for msg in messages:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            user_content = prompt
            if context:
                user_content = f"{prompt}\n\n(Answer based on the reference documents provided above.)"
            chat_messages.append({"role": "user", "content": user_content})

        return chat_messages

    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback message formatting for tokenizers without chat template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _prepare_inputs(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> tuple:
        """Prepare tokenized inputs for generation."""
        chat_messages = self._build_chat_messages(
            messages, prompt, instructions, context
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = self._format_messages_fallback(chat_messages)

        # Tokenize and move to device in one step
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=(
                self.max_context_length - self.max_new_tokens
                if self.max_context_length
                else None
            ),
        ).to(self._device, non_blocking=True)

        input_length = inputs["input_ids"].shape[1]

        return inputs, input_length

    def _build_generation_kwargs(
        self,
        inputs: Dict[str, torch.Tensor],
        temperature: float,
        top_p: float,
        streamer: Optional[TextIteratorStreamer] = None,
    ) -> Dict[str, Any]:
        """Build generation kwargs dict to avoid repetition."""
        do_sample = temperature > 0

        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._pad_token_id,
            "eos_token_id": self._eos_token_id,
        }

        # Only include sampling params when do_sample=True
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        return gen_kwargs

    def _generate_sync(
        self,
        inputs: Dict[str, torch.Tensor],
        input_length: int,
        temperature: float,
        top_p: float,
    ) -> tuple:
        """Synchronous generation - runs in thread pool."""
        gen_kwargs = self._build_generation_kwargs(inputs, temperature, top_p)

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return response_text, len(generated_tokens)

    async def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Async generate with optional context from prior search() call.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions (not used in transformers backend currently)

        Returns:
            Dictionary with reply, latency_seconds, usage, and optionally context info
        """
        start_time = time.perf_counter()

        if tools:
            logger.warning(
                "[TransformersRunner] Native tool calling not implemented for transformers backend, tools will be ignored"
            )

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(
                f"[TransformersRunner] Using search context ({len(context)} chars)"
            )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[TransformersRunner] Generating (async)...")

        inputs, input_length = self._prepare_inputs(
            messages, prompt, instructions, context
        )

        loop = asyncio.get_running_loop()
        response_text, completion_tokens = await loop.run_in_executor(
            None, self._generate_sync, inputs, input_length, temp, tp
        )

        latency = time.perf_counter() - start_time

        response = {
            "reply": response_text,
            "latency_seconds": latency,
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
            },
        }

        if context:
            response["context_used"] = context
            if context_metadata:
                response["search_metadata"] = context_metadata

        return response

    async def generate_stream(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Async streaming generation using TextIteratorStreamer.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions (not used in transformers backend currently)

        Yields:
            StreamChunk objects with text deltas and final metadata
        """
        start_time = time.perf_counter()

        if tools:
            logger.warning(
                "[TransformersRunner] Native tool calling not implemented for transformers backend, tools will be ignored"
            )

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(
                f"[TransformersRunner] Using search context ({len(context)} chars)"
            )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[TransformersRunner] Generating (async streaming)...")

        inputs, input_length = self._prepare_inputs(
            messages, prompt, instructions, context
        )

        # Create streamer with skip_prompt to avoid re-emitting input
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = self._build_generation_kwargs(inputs, temp, tp, streamer)

        # Track generated tokens via output length after generation
        generation_complete = asyncio.Event()
        generation_error: Optional[Exception] = None
        output_ids: Optional[torch.Tensor] = None

        def generate_in_thread():
            nonlocal generation_error, output_ids
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(**gen_kwargs)
                    output_ids = outputs[0]
            except Exception as e:
                generation_error = e
            finally:
                generation_complete.set()

        # Start generation thread
        thread = Thread(target=generate_in_thread, daemon=True)
        thread.start()

        # Stream tokens from streamer
        loop = asyncio.get_running_loop()

        def get_next_token():
            """Blocking call to get next token from streamer."""
            try:
                return next(iter(streamer))
            except StopIteration:
                return None

        while True:
            # Get next text chunk from streamer (blocking in thread pool)
            text = await loop.run_in_executor(None, get_next_token)

            if text is None:
                break

            if text:
                yield StreamChunk(text=text, finished=False)

        # Wait for generation to complete
        await generation_complete.wait()

        # Calculate final stats
        latency = time.perf_counter() - start_time

        completion_tokens = 0
        if output_ids is not None:
            completion_tokens = len(output_ids) - input_length

        # Yield final chunk with metadata
        final_chunk = StreamChunk(
            text="",
            finished=True,
            finish_reason="error" if generation_error else "stop",
            latency_seconds=latency,
            usage={
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
            },
            context_used=context,
            search_metadata=context_metadata,
        )

        if generation_error:
            logger.error(f"[TransformersRunner] Generation error: {generation_error}")
            final_chunk.finish_reason = f"error: {str(generation_error)}"

        yield final_chunk
