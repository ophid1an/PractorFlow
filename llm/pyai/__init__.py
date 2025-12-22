"""
Pydantic AI integration for local LLM runners.

Provides a custom Model implementation that bridges Pydantic AI agents
with the local LLM inference library (llama.cpp and transformers backends).

Usage:
    from llm.pyai import LocalLLMModel, KnowledgeDeps, search_knowledge

    # Create model from runner
    model = LocalLLMModel(runner)

    # Use with Pydantic AI agent
    agent = Agent(model=model)
    result = await agent.run("Hello!")

    # With knowledge search tool
    agent = Agent(model=model, deps_type=KnowledgeDeps)
    agent.tool(search_knowledge)
"""

from llm.pyai.model import LocalLLMModel
from llm.pyai.stream_response import LocalStreamedResponse
from llm.pyai.message_converter import MessageConverter
from llm.pyai.tools import (
    KnowledgeDeps,
    search_knowledge,
    search_knowledge_generic,
    register_knowledge_tools,
    format_search_results,
)

__all__ = [
    "LocalLLMModel",
    "LocalStreamedResponse",
    "MessageConverter",
    "KnowledgeDeps",
    "search_knowledge",
    "search_knowledge_generic",
    "register_knowledge_tools",
    "format_search_results",
]