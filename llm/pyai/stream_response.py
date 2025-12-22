"""
Streamed response implementation for local LLM runners.

Provides LocalStreamedResponse that implements Pydantic AI's StreamedResponse protocol.
Handles tool call extraction from streamed text.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

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

logger = get_logger("pyai_stream", level=appConfiguration.LoggerConfiguration.AgentLevel)


class LocalStreamedResponse:
    """
    Streamed response for local LLM runners.

    Implements the StreamedResponse protocol expected by Pydantic AI.
    Handles tool call extraction from accumulated text after streaming completes.
    """

    def __init__(
        self,
        runner: LLMRunner,
        gen_kwargs: Dict[str, Any],
        model_name_str: str,
        available_tools: List[ToolDefinition] = None,
    ):
        """
        Initialize streamed response.

        Args:
            runner: The LLM runner instance.
            gen_kwargs: Generation keyword arguments for the runner.
            model_name_str: Model name string for identification.
            available_tools: List of available tools for extraction.
        """
        self._runner = runner
        self._gen_kwargs = gen_kwargs
        self._model_name_str = model_name_str
        self._available_tools = available_tools or []
        self._timestamp = datetime.now(timezone.utc)
        self._usage: Optional[RequestUsage] = None
        self._parts: List[ModelResponsePart] = []
        self._accumulated_text: str = ""
        self._stream_started: bool = False

    def _extract_tool_calls(self, text: str) -> List[ToolCallPart]:
        """
        Extract tool calls from accumulated text.

        Args:
            text: The accumulated response text.

        Returns:
            List of ToolCallPart objects found in the text.
        """
        tool_calls = []
        tool_names = {tool.name for tool in self._available_tools}
        matches = []

        # Pattern 1: JSON in code blocks
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        for match in re.finditer(code_block_pattern, text, re.DOTALL):
            try:
                json_str = match.group(1).strip()
                obj = json.loads(json_str)
                if "tool" in obj or "name" in obj:
                    matches.append(obj)
            except json.JSONDecodeError:
                continue

        # Pattern 2: Bare JSON objects using brace matching
        if not matches:
            brace_depth = 0
            start_idx = None

            for i, char in enumerate(text):
                if char == "{":
                    if brace_depth == 0:
                        start_idx = i
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx is not None:
                        try:
                            json_str = text[start_idx : i + 1]
                            obj = json.loads(json_str)
                            if "tool" in obj or "name" in obj:
                                matches.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = None

        # Convert matches to ToolCallPart objects
        for obj in matches:
            tool_name = obj.get("tool") or obj.get("name")
            tool_args = obj.get("args") or obj.get("arguments", {})

            if tool_name and tool_name in tool_names:
                if not isinstance(tool_args, dict):
                    tool_args = {}

                tool_calls.append(
                    ToolCallPart(
                        tool_name=tool_name,
                        args=tool_args,
                        tool_call_id=f"call_{len(tool_calls)}",
                    )
                )

        return tool_calls

    def _clean_text_from_tool_calls(self, text: str) -> str:
        """Remove tool call JSON from text."""
        text = re.sub(r'```(?:json)?\s*\{[^`]+\}\s*```', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^{}]*"tool"\s*:[^{}]*\}', '', text)
        text = re.sub(r'\{[^{}]*"name"\s*:[^{}]*\}', '', text)
        return text.strip()

    async def __aiter__(self):
        """
        Stream events from the LLM.

        Yields Pydantic AI streaming events (PartStartEvent, PartDeltaEvent).
        After streaming completes, extracts tool calls from accumulated text.
        """
        from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPartDelta

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
                        delta=TextPartDelta(content_delta=chunk.text),
                    )

            if chunk.finished:
                if chunk.usage:
                    self._usage = RequestUsage(
                        input_tokens=chunk.usage.get("prompt_tokens", 0),
                        output_tokens=chunk.usage.get("completion_tokens", 0),
                    )

        # Post-process: extract tool calls from accumulated text
        if self._available_tools and self._accumulated_text:
            tool_calls = self._extract_tool_calls(self._accumulated_text)

            if tool_calls:
                # Clean the text
                cleaned_text = self._clean_text_from_tool_calls(self._accumulated_text)

                # Rebuild parts: tool calls first, then cleaned text
                new_parts = list(tool_calls)

                if cleaned_text:
                    new_parts.append(TextPart(content=cleaned_text))

                self._parts = new_parts

                logger.info(
                    f"[LocalStreamedResponse] Extracted {len(tool_calls)} tool calls"
                )

        self._stream_started = True

    def get(self) -> ModelResponse:
        """
        Build the final ModelResponse after streaming completes.

        Returns:
            Complete ModelResponse with all accumulated content.
        """
        parts = self._parts if self._parts else [TextPart(content=self._accumulated_text)]

        return ModelResponse(
            parts=parts,
            model_name=self._model_name_str,
            timestamp=self._timestamp,
            usage=self._usage or RequestUsage(input_tokens=0, output_tokens=0),
        )

    def usage(self) -> RequestUsage:
        """Get current usage stats."""
        return self._usage or RequestUsage(input_tokens=0, output_tokens=0)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name_str

    @property
    def timestamp(self) -> datetime:
        """Get the response timestamp."""
        return self._timestamp