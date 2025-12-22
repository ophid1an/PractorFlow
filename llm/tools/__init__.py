"""
LLM Tools module.

Provides tool infrastructure for LLM function calling and RAG.
"""

from llm.tools.base import BaseTool, ToolParameter, ToolResult
from llm.tools.tool_registry import ToolRegistry
from llm.tools.knowledge_search import KnowledgeSearchTool
from llm.tools.base_web_search import DuckDuckGoSearchTool
from llm.tools.serpapi_web_search import SerpAPISearchTool

__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "KnowledgeSearchTool",
    "DuckDuckGoSearchTool",
    "SerpAPISearchTool",
]