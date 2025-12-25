"""
Pydantic AI Model implementation for local LLM runners.

Provides LocalLLMModel that implements the Pydantic AI Model protocol,
bridging Pydantic AI agents with llama.cpp and transformers backends.

The Model is responsible for:
- Converting messages between Pydantic AI and runner formats
- Including tool definitions in prompts (via prompt engineering for local models)
- Parsing tool calls from model responses
- Returning properly formatted ModelResponse objects
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
)
from pydantic_ai.usage import RequestUsage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.settings import ModelSettings

from practorflow.llm.base.llm_runner import LLMRunner
from practorflow.llm.pyai.message_converter import MessageConverter
from practorflow.llm.pyai.stream_response import LocalStreamedResponse

from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

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

    Uses prompt engineering to enable tool calling for local models
    that don't have native function calling support.
    """

    def __init__(
        self,
        runner: LLMRunner,
        system_prompt: Optional[str] = None,
    ):
        self._runner = runner
        self._model_name = runner.model_name
        self._system_prompt = system_prompt
        self._converter = MessageConverter()
        logger.info(f"[LocalLLMModel] Initialized with model: {self._model_name}")

    @property
    def model_name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def system(self) -> str:
        return "local"

    def _build_tool_prompt(
        self,
        function_tools: List[ToolDefinition],
        output_tools: List[ToolDefinition] = None,
    ) -> str:
        """
        Build system prompt with tool descriptions for prompt engineering.

        Args:
            function_tools: List of available function tools.
            output_tools: List of output tools (for structured output).

        Returns:
            System prompt instructing model how to use tools.
        """
        output_tools = output_tools or []
        function_tools = function_tools or []

        if not function_tools and not output_tools:
            return ""

        prompt_parts = []

        if function_tools:
            tool_descriptions = []

            for tool in function_tools:
                params_desc = []

                if hasattr(tool, "parameters_json_schema"):
                    schema = tool.parameters_json_schema
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])

                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        is_required = param_name in required
                        req_str = " (REQUIRED)" if is_required else " (optional)"
                        params_desc.append(
                            f"    - {param_name} ({param_type}){req_str}: {param_desc}"
                        )

                params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"
                tool_desc = f"- {tool.name}: {tool.description}\n  Parameters:\n{params_str}"
                tool_descriptions.append(tool_desc)

            prompt_parts.append("AVAILABLE TOOLS:\n" + "\n\n".join(tool_descriptions))

        tool_names = [t.name for t in function_tools]

        instruction = f"""You have access to the following tools: {', '.join(tool_names)}

{chr(10).join(prompt_parts)}

IMPORTANT: When you need to use a tool, respond with ONLY a JSON object in this exact format:
{{"tool": "TOOL_NAME", "args": {{"param_name": "value"}}}}

Do not include any other text when calling a tool.
After receiving tool results, provide your final answer based on the information."""

        return instruction

    def _extract_tool_calls(
        self,
        text: str,
        available_tools: List[ToolDefinition],
    ) -> List[ToolCallPart]:
        """
        Extract tool calls from model's text output.

        Args:
            text: The model's response text.
            available_tools: List of available tools to validate against.

        Returns:
            List of ToolCallPart objects found in the text.
        """
        tool_calls = []
        tool_names = {tool.name for tool in available_tools}
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

            if not tool_name or tool_name not in tool_names:
                continue

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
        """Remove tool call JSON from text response."""
        text = re.sub(r'```(?:json)?\s*\{[^`]+\}\s*```', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^{}]*"tool"\s*:[^{}]*\}', '', text)
        text = re.sub(r'\{[^{}]*"name"\s*:[^{}]*\}', '', text)
        return text.strip()

    def _has_tool_results(self, messages: List[ModelMessage]) -> bool:
        """Check if messages contain tool results."""
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return True
        return False

    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Make a non-streaming request to the local LLM.

        Args:
            messages: List of ModelMessage from Pydantic AI.
            model_settings: Optional model settings (temperature, etc.).
            model_request_parameters: Parameters including tool definitions.

        Returns:
            ModelResponse with the LLM's response.
        """
        conversion = self._converter.convert_messages(messages, self._system_prompt)

        temperature = model_settings.get("temperature") if model_settings else None
        top_p = model_settings.get("top_p") if model_settings else None

        has_function_tools = bool(model_request_parameters.function_tools)
        has_output_tools = bool(model_request_parameters.output_tools)
        has_tools = has_function_tools or has_output_tools

        # Build system prompt with tool instructions if tools are available
        system_parts = []
        if conversion.system_prompt:
            system_parts.append(conversion.system_prompt)

        if has_tools:
            tool_prompt = self._build_tool_prompt(
                model_request_parameters.function_tools,
                model_request_parameters.output_tools,
            )
            system_parts.append(tool_prompt)
            logger.debug(
                f"[LocalLLMModel] Added tool prompt for "
                f"{len(model_request_parameters.function_tools or [])} tools"
            )

        final_system_prompt = "\n\n".join(system_parts) if system_parts else None

        gen_kwargs = {
            "messages": conversion.messages,
            "instructions": final_system_prompt,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        logger.debug(f"[LocalLLMModel] Request with {len(conversion.messages)} messages")

        result = await self._runner.generate(**gen_kwargs)

        # Extract response text
        reply = result.get("reply", "")
        if isinstance(reply, dict):
            choices = reply.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                reply = message.get("content", "") or ""

        parts: List[ModelResponsePart] = []

        # Extract tool calls if tools are available
        if has_tools:
            all_tools = list(model_request_parameters.function_tools or []) + list(
                model_request_parameters.output_tools or []
            )
            tool_calls = self._extract_tool_calls(reply, all_tools)

            if tool_calls:
                parts.extend(tool_calls)
                for tc in tool_calls:
                    logger.info(f"[LocalLLMModel] Tool call: {tc.tool_name}({tc.args})")
                reply = self._clean_text_from_tool_calls(reply)

        # Add remaining text as TextPart
        if reply and reply.strip():
            parts.append(TextPart(content=reply))

        if not parts:
            parts.append(TextPart(content=""))

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

        Args:
            messages: List of ModelMessage from Pydantic AI.
            model_settings: Optional model settings.
            model_request_parameters: Parameters including tool definitions.
            run_context: Optional run context (required by Pydantic AI protocol).

        Yields:
            StreamedResponse that can be iterated for chunks.
        """
        conversion = self._converter.convert_messages(messages, self._system_prompt)

        temperature = model_settings.get("temperature") if model_settings else None
        top_p = model_settings.get("top_p") if model_settings else None

        has_function_tools = bool(model_request_parameters.function_tools)
        has_output_tools = bool(model_request_parameters.output_tools)
        has_tools = has_function_tools or has_output_tools

        system_parts = []
        if conversion.system_prompt:
            system_parts.append(conversion.system_prompt)

        if has_tools:
            tool_prompt = self._build_tool_prompt(
                model_request_parameters.function_tools,
                model_request_parameters.output_tools,
            )
            system_parts.append(tool_prompt)

        final_system_prompt = "\n\n".join(system_parts) if system_parts else None

        gen_kwargs = {
            "messages": conversion.messages,
            "instructions": final_system_prompt,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        logger.debug(
            f"[LocalLLMModel] Stream request with {len(conversion.messages)} messages"
        )

        all_tools = []
        if has_tools:
            all_tools = list(model_request_parameters.function_tools or []) + list(
                model_request_parameters.output_tools or []
            )

        stream = LocalStreamedResponse(
            runner=self._runner,
            gen_kwargs=gen_kwargs,
            model_name_str=self.model_name,
            available_tools=all_tools,
        )

        try:
            yield stream
        finally:
            pass