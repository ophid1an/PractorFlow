from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPartDelta
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import (
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai.tools import ToolDefinition

from llm.base.llm_runner import LLMRunner

from logger.logger import get_logger
from settings.app_settings import appConfiguration

logger = get_logger("pyai_stream", level=appConfiguration.LoggerConfiguration.RunnerLevel)


@dataclass
class LocalStreamedResponse:
    """
    Streamed response for local LLM runners with tool calling support.
    
    Handles both native function calling and prompt engineering modes.
    """
    
    model_request_parameters: ModelRequestParameters
    _runner: LLMRunner = field(repr=False)
    _gen_kwargs: dict = field(repr=False)
    _model_name_str: str = field(default="")
    _has_tools: bool = field(default=False)
    _supports_native_fc: bool = field(default=False)
    _user_message_fallback: Optional[str] = field(default=None)
    _timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _usage: Optional[RequestUsage] = field(default=None, init=False)
    _parts: List[ModelResponsePart] = field(default_factory=list, init=False)
    _accumulated_text: str = field(default="", init=False)
    _stream_started: bool = field(default=False, init=False)
    
    def __init__(
        self,
        runner: LLMRunner,
        gen_kwargs: dict,
        model_name_str: str,
        model_request_parameters: ModelRequestParameters,
        has_tools: bool = False,
        supports_native_fc: bool = False,
        user_message_fallback: Optional[str] = None,
    ):
        self._runner = runner
        self._gen_kwargs = gen_kwargs
        self._model_name_str = model_name_str
        self.model_request_parameters = model_request_parameters
        self._has_tools = has_tools
        self._supports_native_fc = supports_native_fc
        self._user_message_fallback = user_message_fallback
        self._timestamp = datetime.now(timezone.utc)
        self._usage = None
        self._parts = []
        self._accumulated_text = ""
        self._stream_started = False
    
    def _extract_tool_calls_from_text(
        self,
        text: str,
        available_tools: List[ToolDefinition]
    ) -> List[ToolCallPart]:
        """Extract tool calls from accumulated text."""
        tool_calls = []
        tool_names = {tool.name for tool in available_tools}
        
        # Build a map of tool name to required parameters
        tool_required_params = {}
        for tool in available_tools:
            if hasattr(tool, 'parameters_json_schema'):
                schema = tool.parameters_json_schema
                tool_required_params[tool.name] = schema.get('required', [])
        
        # Try to find JSON objects
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}'
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        
        matches = []
        
        # Check for code blocks
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                json_str = match.group(1).strip()
                obj = json.loads(json_str)
                matches.append(obj)
            except json.JSONDecodeError:
                continue
        
        # If no code blocks, look for bare JSON
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
        
        # Convert to ToolCallPart
        for obj in matches:
            tool_name = obj.get('tool') or obj.get('name')
            tool_args = obj.get('args') or obj.get('arguments', {})
            
            if tool_name and tool_name in tool_names:
                # Ensure tool_args is a dict
                if not isinstance(tool_args, dict):
                    tool_args = {}
                
                # Fix empty or missing query parameter using fallback
                if 'query' in tool_required_params.get(tool_name, []):
                    query_value = tool_args.get('query', '')
                    if not query_value or (isinstance(query_value, str) and not query_value.strip()):
                        if self._user_message_fallback:
                            tool_args['query'] = self._user_message_fallback
                            logger.info(f"[LocalStreamedResponse] Empty query parameter, using user message fallback: '{self._user_message_fallback}'")
                        else:
                            logger.warning(f"[LocalStreamedResponse] Empty query parameter and no fallback available")
                
                tool_calls.append(ToolCallPart(
                    tool_name=tool_name,
                    args=tool_args,
                    tool_call_id=f"call_{len(tool_calls)}"
                ))
        
        return tool_calls
    
    def _clean_text_from_tool_calls(self, text: str) -> str:
        """Remove tool call JSON from text."""
        # Remove code blocks with JSON
        text = re.sub(r'```(?:json)?\s*\{[^`]+\}\s*```', '', text, flags=re.DOTALL)
        
        # Remove bare JSON objects
        text = re.sub(r'\{[^{}]*"tool"\s*:[^{}]*\}', '', text)
        text = re.sub(r'\{[^{}]*"name"\s*:[^{}]*\}', '', text)
        
        return text.strip()
    
    async def __aiter__(self):
        """Stream events with tool call extraction at the end."""
        text_part_index = 0
        first_chunk = True
        
        async for chunk in self._runner.generate_stream(**self._gen_kwargs):
            if chunk.text:
                self._accumulated_text += chunk.text
                
                if first_chunk:
                    part = TextPart(content=chunk.text)
                    self._parts.append(part)
                    yield PartStartEvent(index=text_part_index, part=part)
                    first_chunk = False
                else:
                    self._parts[text_part_index] = TextPart(content=self._accumulated_text)
                    yield PartDeltaEvent(
                        index=text_part_index,
                        delta=TextPartDelta(content_delta=chunk.text)
                    )
            
            if chunk.finished:
                if chunk.usage:
                    self._usage = RequestUsage(
                        input_tokens=chunk.usage.get("prompt_tokens", 0),
                        output_tokens=chunk.usage.get("completion_tokens", 0),
                    )
                
                # Post-process for tool calls if needed
                if self._has_tools and self.model_request_parameters.function_tools:
                    tool_calls = self._extract_tool_calls_from_text(
                        self._accumulated_text,
                        self.model_request_parameters.function_tools
                    )
                    
                    if tool_calls:
                        # Clean the text part
                        cleaned_text = self._clean_text_from_tool_calls(self._accumulated_text)
                        
                        # Rebuild parts list
                        new_parts = []
                        
                        # Add tool calls first
                        new_parts.extend(tool_calls)
                        
                        # Add cleaned text if any
                        if cleaned_text:
                            new_parts.append(TextPart(content=cleaned_text))
                        
                        self._parts = new_parts
                        
                        logger.info(f"[LocalStreamedResponse] Extracted {len(tool_calls)} tool calls from stream")
        
        self._stream_started = True
    
    def get(self) -> ModelResponse:
        """Build the final ModelResponse."""
        return ModelResponse(
            parts=self._parts if self._parts else [TextPart(content=self._accumulated_text)],
            model_name=self._model_name_str,
            timestamp=self._timestamp,
            usage=self._usage or RequestUsage(input_tokens=0, output_tokens=0),
        )
    
    def usage(self) -> RequestUsage:
        """Get current usage stats."""
        return self._usage or RequestUsage(input_tokens=0, output_tokens=0)
    
    @property
    def model_name(self) -> str:
        return self._model_name_str
    
    @property
    def timestamp(self) -> datetime:
        return self._timestamp
    
    @property
    def provider_name(self) -> str | None:
        return "local"
    
    @property
    def provider_url(self) -> str | None:
        return None