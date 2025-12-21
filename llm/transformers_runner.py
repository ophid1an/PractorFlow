import asyncio
import time
from threading import Thread
from typing import Dict, Any, Optional, List, AsyncIterator

import torch
from transformers import TextIteratorStreamer

from llm.base.llm_runner import LLMRunner, StreamChunk
from llm.pool.model_handle import ModelHandle
from llm.knowledge.knowledge_store import KnowledgeStore
from logger.logger import get_logger
from settings.app_settings import appConfiguration

logger = get_logger(
    "transformers_runner", level=appConfiguration.LoggerConfiguration.RunnerLevel
)


class TransformersRunner(LLMRunner):
    """Async runner for HuggingFace Transformers models."""

    def __init__(
        self, handle: ModelHandle, knowledge_store: Optional[KnowledgeStore] = None
    ):
        if not handle.is_transformers:
            raise ValueError(f"Expected transformers handle, got {handle.backend}")

        super().__init__(handle, knowledge_store)

        # Determine device
        if hasattr(self.model, "device"):
            self._device = self.model.device
        else:
            self._device = next(self.model.parameters()).device

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
            if self.tokenizer and hasattr(self.tokenizer, 'chat_template'):
                chat_template = self.tokenizer.chat_template
                if chat_template:
                    template_lower = str(chat_template).lower()
                    if any(keyword in template_lower for keyword in [
                        'tool', 'function', '<tool_call>', '<function_call>',
                        'tools', 'functions', 'tool_use', 'function_use'
                    ]):
                        logger.info(f"[TransformersRunner] Model supports function calling (detected in chat template)")
                        return True
            
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'to_dict'):
                    config_dict = config.to_dict()
                    if config_dict.get('supports_function_calling', False):
                        logger.info(f"[TransformersRunner] Model supports function calling (config flag)")
                        return True
            
            logger.info(f"[TransformersRunner] Model does NOT support native function calling")
            return False
            
        except Exception as e:
            logger.warning(f"[TransformersRunner] Error detecting function calling support: {e}")
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

        # Add custom instructions first
        if instructions:
            system_parts.append(instructions)

        # Add context with clear instructions for the model to use it
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
            # When context is provided, remind the model to use it
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

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=(
                self.max_context_length - self.max_new_tokens
                if self.max_context_length
                else None
            ),
        ).to(self._device)

        input_length = inputs["input_ids"].shape[1]

        return inputs, input_length

    def _generate_sync(
        self,
        inputs: Dict[str, torch.Tensor],
        input_length: int,
        temperature: float,
        top_p: float,
    ) -> tuple:
        """Synchronous generation - runs in thread pool."""
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

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
    ) -> Dict[str, Any]:
        """Async generate with optional context from prior search() call."""
        start_time = time.time()

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

        # Run blocking inference in thread pool
        loop = asyncio.get_event_loop()
        response_text, completion_tokens = await loop.run_in_executor(
            None, self._generate_sync, inputs, input_length, temp, tp
        )

        latency = time.time() - start_time

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
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming generation using TextIteratorStreamer."""
        start_time = time.time()

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

        # Create async queue to bridge sync streamer to async
        queue: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def stream_in_thread():
            """Run streaming generation in thread, put chunks in queue."""
            try:
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": temp > 0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "streamer": streamer,
                }

                if temp > 0:
                    gen_kwargs["temperature"] = temp
                    gen_kwargs["top_p"] = tp

                generated_token_count = 0

                # Start generation in background thread
                def generate_thread():
                    with torch.inference_mode():
                        self.model.generate(**gen_kwargs)

                gen_thread = Thread(target=generate_thread)
                gen_thread.start()

                # Iterate streamer and put chunks in queue
                for text in streamer:
                    if text:
                        generated_token_count += len(
                            self.tokenizer.encode(text, add_special_tokens=False)
                        )
                        chunk = StreamChunk(text=text, finished=False)
                        asyncio.run_coroutine_threadsafe(
                            queue.put(chunk), loop
                        ).result()

                gen_thread.join()

                # Put final chunk
                latency = time.time() - start_time
                final_chunk = StreamChunk(
                    text="",
                    finished=True,
                    finish_reason="stop",
                    latency_seconds=latency,
                    usage={
                        "prompt_tokens": input_length,
                        "completion_tokens": generated_token_count,
                        "total_tokens": input_length + generated_token_count,
                    },
                    context_used=context,
                    search_metadata=context_metadata,
                )
                asyncio.run_coroutine_threadsafe(queue.put(final_chunk), loop).result()

            except Exception as e:
                error_chunk = StreamChunk(
                    text="", finished=True, finish_reason=f"error: {str(e)}"
                )
                asyncio.run_coroutine_threadsafe(queue.put(error_chunk), loop).result()
            finally:
                # Signal end of stream
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        # Start streaming in thread
        loop.run_in_executor(None, stream_in_thread)

        # Yield chunks from queue
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk