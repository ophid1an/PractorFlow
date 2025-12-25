"""
Message conversion between Pydantic AI and local LLM runner formats.

Pydantic AI uses structured message types (ModelRequest, ModelResponse)
while the local runners expect simple List[Dict[str, str]] with role/content.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
)


@dataclass
class ConversionResult:
    """Result of message conversion."""

    messages: List[Dict[str, str]]
    system_prompt: Optional[str] = None


class MessageConverter:
    """
    Converts between Pydantic AI message formats and runner message format.

    Runner format: List[Dict[str, str]] with keys 'role' and 'content'
    Supported roles: 'system', 'user', 'assistant'
    """

    @staticmethod
    def extract_text_content(part: Any) -> str:
        """Extract text content from a message part."""
        if isinstance(part, str):
            return part
        if isinstance(part, TextPart):
            return part.content
        if isinstance(part, UserPromptPart):
            return part.content if isinstance(part.content, str) else str(part.content)
        if isinstance(part, SystemPromptPart):
            return part.content
        if isinstance(part, ToolReturnPart):
            # Format tool returns clearly for the model
            content = part.content
            if isinstance(content, str):
                return f"[Tool Result: {part.tool_name}]\n{content}"
            return f"[Tool Result: {part.tool_name}]\n{str(content)}"
        if isinstance(part, RetryPromptPart):
            return part.content if isinstance(part.content, str) else str(part.content)
        if isinstance(part, ToolCallPart):
            # Tool calls from assistant - format as indication
            args_str = str(part.args) if part.args else "{}"
            return f"[Calling tool: {part.tool_name}({args_str})]"
        return str(part)

    @classmethod
    def convert_model_request(cls, request: ModelRequest) -> List[Dict[str, str]]:
        """
        Convert a ModelRequest to runner message format.

        ModelRequest contains parts that can be system prompts, user prompts,
        tool returns, or retry prompts. We consolidate into appropriate roles.

        Args:
            request: Pydantic AI ModelRequest.

        Returns:
            List of message dicts for the runner.
        """
        system_parts = []
        user_parts = []

        for part in request.parts:
            if isinstance(part, SystemPromptPart):
                system_parts.append(cls.extract_text_content(part))
            elif isinstance(part, (UserPromptPart, RetryPromptPart)):
                user_parts.append(cls.extract_text_content(part))
            elif isinstance(part, ToolReturnPart):
                # Tool returns go to user context
                user_parts.append(cls.extract_text_content(part))
            else:
                # Fallback for unknown parts
                user_parts.append(cls.extract_text_content(part))

        # Combine parts
        messages = []

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        if user_parts:
            messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        return messages

    @classmethod
    def convert_model_response(cls, response: ModelResponse) -> Dict[str, str]:
        """
        Convert a ModelResponse to runner message format.

        ModelResponse contains parts that are text or tool calls from assistant.

        Args:
            response: Pydantic AI ModelResponse.

        Returns:
            Message dict for the runner.
        """
        content_parts = []

        for part in response.parts:
            content_parts.append(cls.extract_text_content(part))

        return {"role": "assistant", "content": "\n".join(content_parts)}

    @classmethod
    def convert_messages(
        cls, messages: List[ModelMessage], system_prompt: Optional[str] = None
    ) -> ConversionResult:
        """
        Convert a list of Pydantic AI messages to runner format.

        Args:
            messages: List of ModelMessage (ModelRequest or ModelResponse).
            system_prompt: Optional additional system prompt to prepend.

        Returns:
            ConversionResult with converted messages and extracted system prompt.
        """
        runner_messages: List[Dict[str, str]] = []
        collected_system: List[str] = []

        if system_prompt:
            collected_system.append(system_prompt)

        for msg in messages:
            if isinstance(msg, ModelRequest):
                converted = cls.convert_model_request(msg)
                for m in converted:
                    if m["role"] == "system":
                        collected_system.append(m["content"])
                    else:
                        runner_messages.append(m)
            elif isinstance(msg, ModelResponse):
                runner_messages.append(cls.convert_model_response(msg))

        # Combine all system prompts
        final_system = "\n\n".join(collected_system) if collected_system else None

        return ConversionResult(messages=runner_messages, system_prompt=final_system)

    @classmethod
    def create_simple_messages(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Create runner messages from simple prompt string.

        Utility for direct usage without Pydantic AI message types.

        Args:
            prompt: User prompt string.
            system_prompt: Optional system prompt.
            history: Optional conversation history.

        Returns:
            List of message dicts for runner.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        return messages