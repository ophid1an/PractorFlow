"""
Pydantic AI Model implementation for local LLM runners.

Provides LocalLLMModel that implements the Pydantic AI Model protocol,
bridging Pydantic AI agents with llama.cpp and transformers backends.

Supports both native function calling (for capable models) and
prompt engineering fallback (for models without native support).
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional, Any

from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ModelRequest,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.settings import ModelSettings

from llm.base.llm_runner import LLMRunner, StreamChunk
from llm.pyai.message_converter import MessageConverter
from llm.pyai.stream_response import LocalStreamedResponse

from logger.logger import get_logger
from settings.app_settings import appConfiguration

logger = get_logger("pyai_model", level=appConfiguration.LoggerConfiguration.RunnerLevel)


class LocalModelSettings(ModelSettings, total=False):
    """Settings specific to local LLM models."""
    temperature: float
    top_p: float
    max_tokens: int


class LocalLLMModel(Model):
    """
    Pydantic AI Model implementation for local LLM runners.
    
    Wraps an LLMRunner (llama.cpp or transformers) to provide
    the Model interface expected by Pydantic AI agents.
    
    Supports hybrid function calling:
    - Native: Uses model's built-in function calling if supported
    - Prompt Engineering: Falls back to instructing via system prompt
    
    Usage:
        runner = create_runner(handle, knowledge_store)
        model = LocalLLMModel(runner)
        agent = Agent(model=model, tools=[search_tool])
        result = await agent.run("Search for information about X")
    """
    
    def __init__(
        self,
        runner: LLMRunner,
        model_name_override: Optional[str] = None,
        system_prompt: Optional[str] = None,
        settings: Optional[ModelSettings] = None,
    ):
        """
        Initialize LocalLLMModel.
        
        Args:
            runner: LLMRunner instance (llama.cpp or transformers)
            model_name_override: Optional model name override
            system_prompt: Optional default system prompt
            settings: Optional model settings
        """
        super().__init__(settings=settings)
        self.runner = runner
        self._model_name = model_name_override if model_name_override is not None else runner.model_name
        self._system_prompt = system_prompt
        self._converter = MessageConverter()
        
        # Detect function calling capability
        self._supports_native_fc = runner.supports_function_calling()
        logger.info(f"[LocalLLMModel] Function calling mode: {'NATIVE' if self._supports_native_fc else 'PROMPT_ENGINEERING'}")
    
    @property
    def model_name(self) -> str:
        """Return the model name for identification."""
        return f"local:{self._model_name}"
    
    @property
    def system(self) -> str:
        """Return the model provider identifier."""
        return "local"
    
    def _build_tool_system_prompt(self, tools: List[ToolDefinition]) -> str:
        """
        Build system prompt with tool descriptions for prompt engineering mode.
        
        Args:
            tools: List of available tools
            
        Returns:
            System prompt instructing model how to use tools
        """
        if not tools:
            return ""
        
        tool_descriptions = []
        for tool in tools:
            # Build parameter descriptions
            params_desc = []
            if hasattr(tool, 'parameters_json_schema'):
                schema = tool.parameters_json_schema
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    is_required = param_name in required
                    req_str = " (required)" if is_required else " (optional)"
                    params_desc.append(f"  - {param_name} ({param_type}){req_str}: {param_desc}")
            
            tool_desc = f"""
Tool: {tool.name}
Description: {tool.description}
Parameters:
{chr(10).join(params_desc) if params_desc else '  (no parameters)'}
"""
            tool_descriptions.append(tool_desc)
        
        system_prompt = f"""You have access to the following tools:

{chr(10).join(tool_descriptions)}

To use a tool, respond with ONLY a JSON object in this exact format:
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}

CRITICAL RULES:
1. Output ONLY the JSON object, nothing else
2. Do not include markdown code blocks or backticks
3. Do not add any explanation before or after the JSON
4. Use valid JSON syntax with double quotes
5. ALL required parameters MUST be included in "args"
6. If you don't need to use a tool, respond normally in plain text

Example of correct tool usage:
User: "Search for information about AI"
Assistant: {{"tool": "search_knowledge", "args": {{"query": "AI", "top_k": 5}}}}

After receiving tool results, use them to answer the user's question in plain text."""
        
        return system_prompt
    
    def _extract_tool_calls_from_text(
        self,
        text: str,
        available_tools: List[ToolDefinition]
    ) -> List[ToolCallPart]:
        """
        Extract tool calls from model's text output (prompt engineering mode).
        
        Args:
            text: Model's text response
            available_tools: List of available tools
            
        Returns:
            List of extracted ToolCallPart objects
        """
        tool_calls = []
        tool_names = {tool.name for tool in available_tools}
        
        # Try to find JSON objects in the text
        # Pattern 1: Clean JSON object
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}'
        
        # Pattern 2: JSON in code blocks
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        
        matches = []
        
        # Check for code blocks first
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                json_str = match.group(1).strip()
                obj = json.loads(json_str)
                matches.append(obj)
            except json.JSONDecodeError:
                continue
        
        # If no code blocks, look for bare JSON
        if not matches:
            # Find all JSON-like structures
            brace_depth = 0
            start_idx = None
            
            for i, char in enumerate(text):
                if char == '{':
                    if brace_depth == 0:
                        start_idx = i
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx is not None:
                        try:
                            json_str = text[start_idx:i+1]
                            obj = json.loads(json_str)
                            if 'tool' in obj or 'name' in obj:
                                matches.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = None
        
        # Convert matches to ToolCallPart objects
        for obj in matches:
            tool_name = obj.get('tool') or obj.get('name')
            tool_args = obj.get('args') or obj.get('arguments', {})
            
            if tool_name and tool_name in tool_names:
                tool_calls.append(ToolCallPart(
                    tool_name=tool_name,
                    args=tool_args,
                    tool_call_id=f"call_{len(tool_calls)}"
                ))
        
        return tool_calls
    
    def _clean_text_from_tool_calls(self, text: str) -> str:
        """
        Remove tool call JSON from text response.
        
        Args:
            text: Original text with potential tool calls
            
        Returns:
            Cleaned text without tool call JSON
        """
        # Remove code blocks with JSON
        text = re.sub(r'```(?:json)?\s*\{[^`]+\}\s*```', '', text, flags=re.DOTALL)
        
        # Remove bare JSON objects that look like tool calls
        text = re.sub(r'\{[^{}]*"tool"\s*:[^{}]*\}', '', text)
        text = re.sub(r'\{[^{}]*"name"\s*:[^{}]*\}', '', text)
        
        return text.strip()
    
    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Make a non-streaming request to the local LLM.
        
        Implements hybrid function calling:
        - Uses native function calling if model supports it
        - Falls back to prompt engineering otherwise
        
        Args:
            messages: Conversation history
            model_settings: Optional settings
            model_request_parameters: Request parameters including tools
            
        Returns:
            ModelResponse with generated content and/or tool calls
        """
        # Prepare request
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        
        # Convert messages to runner format
        conversion = self._converter.convert_messages(messages, self._system_prompt)
        
        # Extract settings
        temperature = model_settings.get('temperature') if model_settings else None
        top_p = model_settings.get('top_p') if model_settings else None
        
        # Check if tools are requested
        has_tools = bool(model_request_parameters.function_tools)
        
        # Build system prompt
        system_parts = []
        if conversion.system_prompt:
            system_parts.append(conversion.system_prompt)
        
        # Add tool instructions for prompt engineering mode
        if has_tools and not self._supports_native_fc:
            tool_prompt = self._build_tool_system_prompt(model_request_parameters.function_tools)
            system_parts.append(tool_prompt)
            logger.debug("[LocalLLMModel] Using PROMPT_ENGINEERING mode for tools")
        elif has_tools and self._supports_native_fc:
            logger.debug("[LocalLLMModel] Using NATIVE mode for tools")
        
        final_system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        # Build generation kwargs
        gen_kwargs = {
            "messages": conversion.messages,
            "instructions": final_system_prompt,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        # TODO: For native function calling mode, pass tools to runner
        # This requires extending the runner interface to accept tools parameter
        # For now, we only implement prompt engineering mode
        
        # Call runner
        result = await self.runner.generate(**gen_kwargs)
        
        # Extract response text
        reply = result.get("reply", "")
        if isinstance(reply, dict):
            # llama.cpp returns dict with choices
            choices = reply.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                reply = message.get("content", "")
        
        # Build response parts
        parts: List[ModelResponsePart] = []
        
        # Extract tool calls if tools were provided
        if has_tools:
            tool_calls = self._extract_tool_calls_from_text(
                reply, 
                model_request_parameters.function_tools
            )
            
            if tool_calls:
                parts.extend(tool_calls)
                # Log extracted tool calls for debugging
                for tc in tool_calls:
                    logger.info(f"[LocalLLMModel] Extracted tool call: {tc.tool_name}({tc.args})")
                # Remove tool call JSON from text
                reply = self._clean_text_from_tool_calls(reply)
                logger.info(f"[LocalLLMModel] Extracted {len(tool_calls)} tool calls")
            else:
                logger.debug(f"[LocalLLMModel] No tool calls found in response: {reply[:200]}")
        
        # Add text part if there's remaining text
        if reply.strip():
            parts.append(TextPart(content=reply))
        
        # If no parts, add empty text part
        if not parts:
            parts.append(TextPart(content=""))
        
        # Build usage
        usage_data = result.get("usage", {})
        usage = RequestUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        )
        
        return ModelResponse(
            parts=parts,
            model_name=self.model_name,
            timestamp=datetime.now(timezone.utc),
            usage=usage,
        )
    
    @asynccontextmanager
    async def request_stream(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        """
        Make a streaming request to the local LLM.
        
        Note: Streaming with tool calling is complex because we need to
        accumulate the full response to parse tool calls. The stream wrapper
        handles this.
        
        Args:
            messages: Conversation history
            model_settings: Optional settings
            model_request_parameters: Request parameters including tools
            run_context: Optional run context
            
        Yields:
            StreamedResponse that can be iterated for events
        """
        # Prepare request
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        
        # Convert messages
        conversion = self._converter.convert_messages(messages, self._system_prompt)
        
        # Extract settings
        temperature = model_settings.get('temperature') if model_settings else None
        top_p = model_settings.get('top_p') if model_settings else None
        
        # Check if tools are requested
        has_tools = bool(model_request_parameters.function_tools)
        
        # Build system prompt
        system_parts = []
        if conversion.system_prompt:
            system_parts.append(conversion.system_prompt)
        
        # Add tool instructions for prompt engineering mode
        if has_tools and not self._supports_native_fc:
            tool_prompt = self._build_tool_system_prompt(model_request_parameters.function_tools)
            system_parts.append(tool_prompt)
        
        final_system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        # Build kwargs
        gen_kwargs = {
            "messages": conversion.messages,
            "instructions": final_system_prompt,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        # Create streamed response wrapper
        stream = LocalStreamedResponse(
            runner=self.runner,
            gen_kwargs=gen_kwargs,
            model_name_str=self.model_name,
            model_request_parameters=model_request_parameters,
            has_tools=has_tools,
            supports_native_fc=self._supports_native_fc,
        )
        
        try:
            yield stream
        finally:
            pass