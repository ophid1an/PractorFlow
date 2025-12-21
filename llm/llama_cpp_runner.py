import asyncio
import json
import time
from typing import Dict, Any, Optional, List, AsyncIterator

from llm.base.llm_runner import LLMRunner, StreamChunk
from llm.pool.model_handle import ModelHandle
from llm.knowledge.knowledge_store import KnowledgeStore
from logger.logger import get_logger
from settings.app_settings import appConfiguration

logger = get_logger(
    "llama_cpp_runner", level=appConfiguration.LoggerConfiguration.RunnerLevel
)


class LlamaCppRunner(LLMRunner):
    """Async runner for GGUF models using llama-cpp-python."""

    def __init__(
        self, handle: ModelHandle, knowledge_store: Optional[KnowledgeStore] = None
    ):
        if not handle.is_llama_cpp:
            raise ValueError(f"Expected llama_cpp handle, got {handle.backend}")

        super().__init__(handle, knowledge_store)
        logger.info(
            f"[LlamaCppRunner] Initialized with pooled model: {self.model_name}"
        )

    def supports_function_calling(self) -> bool:
        """
        Check if the llama.cpp model supports native function calling.
        
        Returns:
            True if model supports native function calling, False otherwise
        """
        try:
            metadata = self.model.metadata if hasattr(self.model, 'metadata') else {}
            
            if metadata:
                chat_template = metadata.get("tokenizer.chat_template", "")
                if chat_template:
                    template_lower = str(chat_template).lower()
                    if any(keyword in template_lower for keyword in [
                        'tool', 'function', '<tool_call>', '<function_call>',
                        'tools', 'functions', 'tool_use', 'function_use'
                    ]):
                        logger.info(f"[LlamaCppRunner] Model supports function calling (detected in chat template)")
                        return True
            
            logger.info(f"[LlamaCppRunner] Model does NOT support native function calling")
            return False
            
        except Exception as e:
            logger.warning(f"[LlamaCppRunner] Error detecting function calling support: {e}")
            return False

    def _convert_tools_to_llama_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert tool definitions to llama.cpp native format.
        
        Args:
            tools: List of tool definitions in OpenAI/Pydantic AI format
            
        Returns:
            List of tools in llama.cpp format
        """
        llama_tools = []
        for tool in tools:
            if "function" in tool:
                # Already in OpenAI format
                llama_tools.append(tool)
            elif "name" in tool and "description" in tool:
                # Pydantic AI ToolDefinition format
                llama_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get("parameters", tool.get("parameters_json_schema", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }))
                    }
                }
                llama_tools.append(llama_tool)
            else:
                logger.warning(f"[LlamaCppRunner] Unknown tool format: {tool}")
        return llama_tools

    def _extract_tool_calls_from_response(
        self, 
        response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract tool calls from llama.cpp native response.
        
        Args:
            response: Response from create_chat_completion
            
        Returns:
            List of tool call dicts with tool_name, args, tool_call_id
        """
        tool_calls = []
        choices = response.get("choices", [])
        
        if not choices:
            return tool_calls
        
        message = choices[0].get("message", {})
        native_tool_calls = message.get("tool_calls", [])
        
        for tc in native_tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            
            # Parse arguments
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
                logger.warning(f"[LlamaCppRunner] Failed to parse tool arguments: {args_str}")
            
            tool_calls.append({
                "tool_name": tool_name,
                "args": args,
                "tool_call_id": tc.get("id", f"call_{len(tool_calls)}")
            })
        
        return tool_calls

    def _build_chat_messages(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages array for create_chat_completion API."""
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

    def _generate_sync(
        self,
        chat_messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synchronous generation - runs in thread pool."""
        completion_kwargs = {
            "messages": chat_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.max_new_tokens,
        }

        if self.config.stop_tokens:
            completion_kwargs["stop"] = self.config.stop_tokens

        # Add native tools if provided
        if tools:
            llama_tools = self._convert_tools_to_llama_format(tools)
            if llama_tools:
                completion_kwargs["tools"] = llama_tools
                completion_kwargs["tool_choice"] = "auto"
                logger.debug(f"[LlamaCppRunner] Passing {len(llama_tools)} tools to native API")

        return self.model.create_chat_completion(**completion_kwargs)

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
            tools: Optional list of tool definitions for native function calling
            
        Returns:
            Dictionary with reply, latency_seconds, and optionally tool_calls
        """
        start_time = time.time()

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(f"[LlamaCppRunner] Using search context ({len(context)} chars)")

        chat_messages = self._build_chat_messages(
            messages, prompt, instructions, context
        )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[LlamaCppRunner] Generating (async)...")
        if tools:
            logger.info(f"[LlamaCppRunner] With {len(tools)} native tools")

        # Run blocking inference in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._generate_sync, chat_messages, temp, tp, tools
        )

        latency = time.time() - start_time

        # Extract tool calls if present
        tool_calls = self._extract_tool_calls_from_response(result)
        
        response = {
            "reply": result,
            "latency_seconds": latency,
        }

        if tool_calls:
            response["tool_calls"] = tool_calls
            logger.info(f"[LlamaCppRunner] Native tool calls extracted: {[tc['tool_name'] for tc in tool_calls]}")

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
        Async streaming generation.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (alternative to messages)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            tools: Optional list of tool definitions for native function calling
            
        Yields:
            StreamChunk objects with text deltas and final metadata
        """
        start_time = time.time()

        context = self._consume_pending_context()
        context_metadata = None

        if context:
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            logger.info(f"[LlamaCppRunner] Using search context ({len(context)} chars)")

        chat_messages = self._build_chat_messages(
            messages, prompt, instructions, context
        )

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        logger.info("[LlamaCppRunner] Generating (async streaming)...")
        if tools:
            logger.info(f"[LlamaCppRunner] With {len(tools)} native tools")

        # Create async queue to bridge sync generator to async
        queue: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue()

        def stream_in_thread():
            """Run streaming generation in thread, put chunks in queue."""
            try:
                completion_kwargs = {
                    "messages": chat_messages,
                    "temperature": temp,
                    "top_p": tp,
                    "max_tokens": self.max_new_tokens,
                    "stream": True,
                }

                if self.config.stop_tokens:
                    completion_kwargs["stop"] = self.config.stop_tokens

                # Add native tools if provided
                if tools:
                    llama_tools = self._convert_tools_to_llama_format(tools)
                    if llama_tools:
                        completion_kwargs["tools"] = llama_tools
                        completion_kwargs["tool_choice"] = "auto"

                finish_reason = None
                accumulated_tool_calls = []

                for chunk in self.model.create_chat_completion(**completion_kwargs):
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")

                    chunk_finish_reason = choice.get("finish_reason")
                    if chunk_finish_reason:
                        finish_reason = chunk_finish_reason

                    # Handle streaming tool calls
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(accumulated_tool_calls) <= idx:
                                accumulated_tool_calls.append({
                                    "id": "",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            if "id" in tc:
                                accumulated_tool_calls[idx]["id"] = tc["id"]
                            if "function" in tc:
                                if "name" in tc["function"]:
                                    accumulated_tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                                if "arguments" in tc["function"]:
                                    accumulated_tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

                    if content:
                        # Put chunk in queue (blocking call from thread)
                        asyncio.run_coroutine_threadsafe(
                            queue.put(StreamChunk(text=content, finished=False)), loop
                        ).result()

                # Process accumulated tool calls
                tool_calls_result = []
                for tc in accumulated_tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")
                    
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    
                    if tool_name:
                        tool_calls_result.append({
                            "tool_name": tool_name,
                            "args": args,
                            "tool_call_id": tc.get("id", f"call_{len(tool_calls_result)}")
                        })

                # Put final chunk
                latency = time.time() - start_time
                final_chunk = StreamChunk(
                    text="",
                    finished=True,
                    finish_reason=finish_reason,
                    latency_seconds=latency,
                    context_used=context,
                    search_metadata=context_metadata,
                )
                
                # Attach tool calls to final chunk if present
                if tool_calls_result:
                    final_chunk.tool_calls = tool_calls_result
                    
                asyncio.run_coroutine_threadsafe(queue.put(final_chunk), loop).result()

            except Exception as e:
                # Put error chunk
                error_chunk = StreamChunk(
                    text="", finished=True, finish_reason=f"error: {str(e)}"
                )
                asyncio.run_coroutine_threadsafe(queue.put(error_chunk), loop).result()
            finally:
                # Signal end of stream
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        # Start generation in thread
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, stream_in_thread)

        # Yield chunks from queue
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk