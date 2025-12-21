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
from typing import AsyncIterator, List, Optional, Any, Dict

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
    ToolReturnPart,
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.settings import ModelSettings

from llm.base.llm_runner import LLMRunner, StreamChunk
from llm.pyai.message_converter import MessageConverter
from llm.pyai.stream_response import LocalStreamedResponse

from logger.logger import get_logger
from settings.app_settings import appConfiguration

logger = get_logger("agent", level=appConfiguration.LoggerConfiguration.AgentLevel)


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
    """
    
    def __init__(
        self,
        runner: LLMRunner,
        system_prompt: Optional[str] = None,
        settings: Optional[ModelSettings] = None,
    ):
        """
        Initialize LocalLLMModel.
        
        Args:
            runner: LLMRunner instance (llama.cpp or transformers)
            system_prompt: Optional default system prompt
            settings: Optional model settings
        """
        super().__init__(settings=settings)
        self.runner = runner
        self._model_name = runner.model_name
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
    
    def _convert_tool_definitions_to_native(
        self, 
        tools: List[ToolDefinition]
    ) -> List[Dict[str, Any]]:
        """
        Convert Pydantic AI ToolDefinition objects to native tool format.
        
        Args:
            tools: List of ToolDefinition from Pydantic AI
            
        Returns:
            List of tool dicts in OpenAI/llama.cpp format
        """
        native_tools = []
        for tool in tools:
            params = {
                "type": "object",
                "properties": {},
                "required": []
            }
            if hasattr(tool, 'parameters_json_schema'):
                params = tool.parameters_json_schema
            
            native_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params
                }
            }
            native_tools.append(native_tool)
        return native_tools
    
    def _has_tool_results(self, messages: List[ModelMessage]) -> bool:
        """
        Check if the messages contain any tool results.
        
        Args:
            messages: List of ModelMessage
            
        Returns:
            True if any message contains ToolReturnPart
        """
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return True
        return False
    
    def _extract_last_user_message(self, messages: List[ModelMessage]) -> Optional[str]:
        """
        Extract the last user message from the conversation.
        
        Args:
            messages: List of ModelMessage
            
        Returns:
            The last user message text, or None if not found
        """
        for msg in reversed(messages):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        content = part.content
                        if isinstance(content, str):
                            return content
                        return str(content)
        return None
    
    def _build_tool_system_prompt(
        self, 
        function_tools: List[ToolDefinition],
        output_tools: List[ToolDefinition] = None,
    ) -> str:
        """
        Build system prompt with tool descriptions for prompt engineering mode.
        
        Args:
            function_tools: List of available function tools
            output_tools: List of output tools (for structured output)
            
        Returns:
            System prompt instructing model how to use tools
        """
        output_tools = output_tools or []
        function_tools = function_tools or []
        
        # No tools at all - return empty string
        if not function_tools and not output_tools:
            return ""
        
        prompt_parts = []
        all_tool_names = []
        
        # Handle function tools
        if function_tools:
            tool_descriptions = []
            
            for tool in function_tools:
                all_tool_names.append(tool.name)
                params_desc = []
                
                if hasattr(tool, 'parameters_json_schema'):
                    schema = tool.parameters_json_schema
                    properties = schema.get('properties', {})
                    required = schema.get('required', [])
                    
                    for param_name, param_info in properties.items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        is_required = param_name in required
                        req_str = " (REQUIRED)" if is_required else " (optional)"
                        params_desc.append(f"    - {param_name} ({param_type}){req_str}: {param_desc}")
                
                tool_desc = f"- {tool.name}: {tool.description}\n  Parameters:\n" + "\n".join(params_desc)
                tool_descriptions.append(tool_desc)
            
            prompt_parts.append("FUNCTION TOOLS:\n" + "\n".join(tool_descriptions))
        
        # Handle output tools (structured output)
        output_tool_name = None
        example_args = {}
        if output_tools:
            output_tool = output_tools[0]
            output_tool_name = output_tool.name
            all_tool_names.append(output_tool.name)
            
            output_params = []
            
            if hasattr(output_tool, 'parameters_json_schema'):
                schema = output_tool.parameters_json_schema
                properties = schema.get('properties', {})
                
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get('type', 'string')
                    prop_desc = prop_info.get('description', prop_name)
                    output_params.append(f"    - {prop_name} ({prop_type}): {prop_desc}")
                    
                    if prop_type == 'string':
                        example_args[prop_name] = f"your {prop_name} here"
                    elif prop_type == 'array':
                        example_args[prop_name] = ["item1", "item2", "item3"]
                    elif prop_type == 'integer':
                        example_args[prop_name] = 0
                    elif prop_type == 'boolean':
                        example_args[prop_name] = True
            
            output_desc = f"- {output_tool.name}: Use this to return your final structured answer\n  Parameters:\n" + "\n".join(output_params)
            prompt_parts.append(f"OUTPUT TOOL (you MUST call this for your final answer):\n{output_desc}")
        
        # Build instruction based on what tools are available
        if function_tools and output_tools:
            example_output = json.dumps({"tool": output_tool_name, "args": example_args})
            instruction = f"""You MUST respond with JSON tool calls only. Never respond with plain text.

{chr(10).join(prompt_parts)}

WORKFLOW:
1. First, call a function tool to gather information
2. After receiving tool results, you MUST call "{output_tool_name}" with your final answer

RESPONSE FORMAT (use this exact JSON structure):
{{"tool": "TOOL_NAME", "args": {{"param": "value"}}}}

IMPORTANT: When you see "[Tool Result:" in the conversation, that means you already called a tool.
After seeing tool results, your next response MUST be:
{example_output}

Never write plain text. Always respond with a JSON tool call."""
        
        elif output_tools:
            example_output = json.dumps({"tool": output_tool_name, "args": example_args})
            instruction = f"""You MUST respond with a JSON tool call only. Never respond with plain text.

{chr(10).join(prompt_parts)}

RESPONSE FORMAT:
{example_output}

Respond with ONLY the JSON object above, filled with your answer. No other text."""
        
        else:
            first_tool = function_tools[0].name
            instruction = f"""You have access to tools. When you need information, respond with a JSON tool call.

{chr(10).join(prompt_parts)}

TO USE A TOOL, respond with:
{{"tool": "{first_tool}", "args": {{"query": "your search terms"}}}}

After receiving tool results, provide your answer based on the information received."""
        
        return instruction
    
    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """
        Extract JSON object from text that may contain surrounding content.
        """
        # Try to find JSON in code blocks first
        code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
        
        # Try to find bare JSON objects using brace matching
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
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        start_idx = None
                        continue
        
        return None
    
    def _extract_tool_calls_from_text(
        self,
        text: str,
        available_tools: List[ToolDefinition],
        user_message_fallback: Optional[str] = None,
    ) -> List[ToolCallPart]:
        """
        Extract tool calls from model's text output (prompt engineering mode).
        """
        tool_calls = []
        tool_names = {tool.name for tool in available_tools}
        
        # Build a map of tool name to required parameters
        tool_required_params = {}
        for tool in available_tools:
            if hasattr(tool, 'parameters_json_schema'):
                schema = tool.parameters_json_schema
                tool_required_params[tool.name] = schema.get('required', [])
        
        matches = []
        
        # Pattern for JSON in code blocks
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                json_str = match.group(1).strip()
                obj = json.loads(json_str)
                if 'tool' in obj or 'name' in obj:
                    matches.append(obj)
            except json.JSONDecodeError:
                continue
        
        # Look for bare JSON objects
        if not matches:
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
            
            # Skip if tool name doesn't match any available tool
            if not tool_name or tool_name not in tool_names:
                logger.warning(f"[LocalLLMModel] Unknown tool name: {tool_name}, available: {tool_names}")
                continue
            
            # Ensure tool_args is a dict
            if not isinstance(tool_args, dict):
                tool_args = {}
            
            # Fix empty or missing query parameter using fallback
            if 'query' in tool_required_params.get(tool_name, []):
                query_value = tool_args.get('query', '')
                if not query_value or (isinstance(query_value, str) and not query_value.strip()):
                    if user_message_fallback:
                        tool_args['query'] = user_message_fallback
                        logger.info(f"[LocalLLMModel] Empty query parameter, using user message fallback: '{user_message_fallback}'")
                    else:
                        logger.warning(f"[LocalLLMModel] Empty query parameter and no fallback available")
            
            tool_calls.append(ToolCallPart(
                tool_name=tool_name,
                args=tool_args,
                tool_call_id=f"call_{len(tool_calls)}"
            ))
        
        return tool_calls
    
    def _clean_text_from_tool_calls(self, text: str) -> str:
        """
        Remove tool call JSON from text response.
        """
        # Remove code blocks with JSON
        text = re.sub(r'```(?:json)?\s*\{[^`]+\}\s*```', '', text, flags=re.DOTALL)
        
        # Remove bare JSON objects that look like tool calls
        text = re.sub(r'\{[^{}]*"tool"\s*:[^{}]*\}', '', text)
        text = re.sub(r'\{[^{}]*"name"\s*:[^{}]*\}', '', text)
        
        return text.strip()
    
    def _convert_native_tool_calls_to_parts(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolCallPart]:
        """
        Convert native tool calls from runner to ToolCallPart objects.
        
        Args:
            tool_calls: List of tool call dicts from runner
            
        Returns:
            List of ToolCallPart objects
        """
        parts = []
        for tc in tool_calls:
            parts.append(ToolCallPart(
                tool_name=tc["tool_name"],
                args=tc.get("args", {}),
                tool_call_id=tc.get("tool_call_id", f"call_{len(parts)}")
            ))
        return parts
    
    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Make a non-streaming request to the local LLM.
        """
        # Prepare request
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        
        # Extract the last user message for fallback
        user_message_fallback = self._extract_last_user_message(messages)
        
        # Convert messages to runner format
        conversion = self._converter.convert_messages(messages, self._system_prompt)
        
        # Extract settings
        temperature = model_settings.get('temperature') if model_settings else None
        top_p = model_settings.get('top_p') if model_settings else None
        
        # Check if tools are requested
        has_function_tools = bool(model_request_parameters.function_tools)
        has_output_tools = bool(model_request_parameters.output_tools)
        has_tools = has_function_tools or has_output_tools
        
        # Check if there are tool results in the messages (means we already called a tool)
        has_tool_results = self._has_tool_results(messages)
        
        # Build system prompt
        system_parts = []
        if conversion.system_prompt:
            system_parts.append(conversion.system_prompt)
        
        # Prepare native tools if in native mode
        native_tools = None
        if has_tools and self._supports_native_fc:
            # Convert all tools to native format
            all_tool_defs = list(model_request_parameters.function_tools or []) + \
                           list(model_request_parameters.output_tools or [])
            native_tools = self._convert_tool_definitions_to_native(all_tool_defs)
            logger.debug(f"[LocalLLMModel] Using NATIVE mode with {len(native_tools)} tools")
        elif has_tools and not self._supports_native_fc:
            # Add tool instructions for prompt engineering mode
            tool_prompt = self._build_tool_system_prompt(
                model_request_parameters.function_tools,
                model_request_parameters.output_tools,
            )
            system_parts.append(tool_prompt)
            logger.debug("[LocalLLMModel] Using PROMPT_ENGINEERING mode for tools")
        
        final_system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        # Get the converted messages
        runner_messages = conversion.messages
        
        # If we have tool results and output tools in prompt engineering mode, append a reminder
        if has_tool_results and has_output_tools and not self._supports_native_fc and runner_messages:
            output_tool_name = model_request_parameters.output_tools[0].name
            reminder = f'\n\nRespond with ONLY a JSON tool call: {{"tool": "{output_tool_name}", "args": {{...}}}}'
            
            # Find the last user message and append reminder
            for i in range(len(runner_messages) - 1, -1, -1):
                if runner_messages[i].get("role") == "user":
                    runner_messages[i]["content"] += reminder
                    break
        
        # Build generation kwargs
        gen_kwargs = {
            "messages": runner_messages,
            "instructions": final_system_prompt,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        # Add native tools if available
        if native_tools:
            gen_kwargs["tools"] = native_tools
        
        # Call runner
        result = await self.runner.generate(**gen_kwargs)
        
        # Build response parts
        parts: List[ModelResponsePart] = []
        
        # Check for native tool calls first
        if native_tools and "tool_calls" in result and result["tool_calls"]:
            native_tool_calls = result["tool_calls"]
            parts.extend(self._convert_native_tool_calls_to_parts(native_tool_calls))
            logger.info(f"[LocalLLMModel] Native tool calls: {[tc['tool_name'] for tc in native_tool_calls]}")
        
        # Extract response text
        reply = result.get("reply", "")
        if isinstance(reply, dict):
            # llama.cpp returns dict with choices
            choices = reply.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                reply = message.get("content", "") or ""
        
        # If no native tool calls but tools were provided (prompt engineering mode)
        if has_tools and not parts:
            # Combine function tools and output tools for extraction
            all_tools = list(model_request_parameters.function_tools or []) + \
                       list(model_request_parameters.output_tools or [])
            
            tool_calls = self._extract_tool_calls_from_text(
                reply, 
                all_tools,
                user_message_fallback=user_message_fallback,
            )
            
            if tool_calls:
                parts.extend(tool_calls)
                for tc in tool_calls:
                    logger.info(f"[LocalLLMModel] Extracted tool call: {tc.tool_name}({tc.args})")
                reply = self._clean_text_from_tool_calls(reply)
                logger.info(f"[LocalLLMModel] Extracted {len(tool_calls)} tool calls from text")
            else:
                logger.debug(f"[LocalLLMModel] No tool calls found in response: {reply[:200]}")
        
        # Add text part if there's remaining text
        if reply and reply.strip():
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
        """
        # Prepare request
        model_settings, model_request_parameters = self.prepare_request(
            model_settings, model_request_parameters
        )
        
        # Extract the last user message for fallback
        user_message_fallback = self._extract_last_user_message(messages)
        
        # Convert messages
        conversion = self._converter.convert_messages(messages, self._system_prompt)
        
        # Extract settings
        temperature = model_settings.get('temperature') if model_settings else None
        top_p = model_settings.get('top_p') if model_settings else None
        
        # Check if tools are requested
        has_function_tools = bool(model_request_parameters.function_tools)
        has_output_tools = bool(model_request_parameters.output_tools)
        has_tools = has_function_tools or has_output_tools
        
        # Check if there are tool results in the messages
        has_tool_results = self._has_tool_results(messages)
        
        # Build system prompt
        system_parts = []
        if conversion.system_prompt:
            system_parts.append(conversion.system_prompt)
        
        # Prepare native tools if in native mode
        native_tools = None
        if has_tools and self._supports_native_fc:
            all_tool_defs = list(model_request_parameters.function_tools or []) + \
                           list(model_request_parameters.output_tools or [])
            native_tools = self._convert_tool_definitions_to_native(all_tool_defs)
            logger.debug(f"[LocalLLMModel] Stream: Using NATIVE mode with {len(native_tools)} tools")
        elif has_tools and not self._supports_native_fc:
            # Add tool instructions for prompt engineering mode
            tool_prompt = self._build_tool_system_prompt(
                model_request_parameters.function_tools,
                model_request_parameters.output_tools,
            )
            system_parts.append(tool_prompt)
        
        final_system_prompt = "\n\n".join(system_parts) if system_parts else None
        
        # Get the converted messages
        runner_messages = conversion.messages
        
        # If we have tool results and output tools in prompt engineering mode, append a reminder
        if has_tool_results and has_output_tools and not self._supports_native_fc and runner_messages:
            output_tool_name = model_request_parameters.output_tools[0].name
            reminder = f'\n\nRespond with ONLY a JSON tool call: {{"tool": "{output_tool_name}", "args": {{...}}}}'
            
            for i in range(len(runner_messages) - 1, -1, -1):
                if runner_messages[i].get("role") == "user":
                    runner_messages[i]["content"] += reminder
                    break
        
        # Build kwargs
        gen_kwargs = {
            "messages": runner_messages,
            "instructions": final_system_prompt,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        # Add native tools if available
        if native_tools:
            gen_kwargs["tools"] = native_tools
        
        # Create streamed response wrapper
        stream = LocalStreamedResponse(
            runner=self.runner,
            gen_kwargs=gen_kwargs,
            model_name_str=self.model_name,
            model_request_parameters=model_request_parameters,
            has_tools=has_tools,
            supports_native_fc=self._supports_native_fc,
            user_message_fallback=user_message_fallback,
        )
        
        try:
            yield stream
        finally:
            pass