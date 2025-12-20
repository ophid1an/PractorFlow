"""
Pydantic AI integration for local LLM runners.

Provides a custom Model implementation that bridges Pydantic AI agents
with the local LLM inference library (llama.cpp and transformers backends).

Usage:
    from llm.pyai import LocalLLMModel, create_knowledge_search_tool

    # Create model from runner
    model = LocalLLMModel(runner)

    # Use with Pydantic AI agent
    agent = Agent(model=model)
    result = await agent.run("Hello!")

    # With knowledge search tool
    tool = create_knowledge_search_tool(runner)
    agent = Agent(model=model, tools=[tool])
"""

from llm.pyai.model import LocalLLMModel
from llm.pyai.stream_response import LocalStreamedResponse
from llm.pyai.message_converter import MessageConverter
from llm.pyai.tools import create_knowledge_search_tool, KnowledgeSearchParams

__all__ = [
    "LocalLLMModel",
    "MessageConverter",
    "create_knowledge_search_tool",
    "KnowledgeSearchParams",
    "LocalStreamedResponse"
]