"""
Tool registry for managing available tools.

Provides centralized tool registration, lookup, and execution.
"""

from typing import Dict, List, Optional, Any
from llm.tools.base import BaseTool, ToolResult
from logger.logger import get_logger
from settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class ToolRegistry:
    """
    Registry for managing LLM tools.

    Handles tool registration, lookup, and execution with
    support for document scope management.
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._document_scope: Optional[set] = None
        self._last_result: Optional[ToolResult] = None

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool
        logger.info(f"[ToolRegistry] Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"[ToolRegistry] Unregistered tool: {tool_name}")
            return True
        return False

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Tool name to lookup

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tool instances."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all registered tools.

        Returns:
            List of tool schemas in function calling format
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def set_document_scope(self, document_ids: Optional[set]) -> None:
        """
        Set document scope for knowledge-aware tools.

        Args:
            document_ids: Set of document IDs to scope searches to,
                         or None to search all documents
        """
        if document_ids is not None:
            self._document_scope = set(document_ids)
            logger.debug(
                f"[ToolRegistry] Document scope set: {len(self._document_scope)} documents"
            )
        else:
            self._document_scope = None
            logger.debug("[ToolRegistry] Document scope cleared (all documents)")

    def get_document_scope(self) -> Optional[set]:
        """Get current document scope."""
        return self._document_scope

    def clear_document_scope(self) -> None:
        """Clear document scope to search all documents."""
        self._document_scope = None
        logger.info("[ToolRegistry] Document scope cleared")

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Parameters to pass to tool

        Returns:
            ToolResult from execution
        """
        tool = self._tools.get(tool_name)

        if not tool:
            result = ToolResult(success=False, error=f"Tool not found: {tool_name}")
            self._last_result = result
            return result

        # Inject document scope for knowledge-aware tools
        if hasattr(tool, "set_document_scope"):
            tool.set_document_scope(self._document_scope)

        result = tool(**kwargs)
        self._last_result = result

        return result

    def get_last_result(self) -> Optional[ToolResult]:
        """Get the result from the last tool execution."""
        return self._last_result

    def clear_last_result(self) -> None:
        """Clear the last tool result."""
        self._last_result = None

    def has_pending_context(self) -> bool:
        """Check if there's a successful tool result to use as context."""
        return (
            self._last_result is not None
            and self._last_result.success
            and self._last_result.data is not None
        )

    def consume_context(self) -> Optional[str]:
        """
        Get and clear pending context from last tool execution.

        Returns:
            Context string if available, None otherwise
        """
        if not self.has_pending_context():
            return None

        context = self._last_result.to_context_string()
        self._last_result = None
        return context

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __repr__(self) -> str:
        tools = ", ".join(self._tools.keys())
        return f"ToolRegistry(tools=[{tools}])"
