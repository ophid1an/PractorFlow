"""
Usage examples for Pydantic AI integration with local LLM runners.

This file demonstrates various ways to use the LocalLLMModel with
Pydantic AI agents, including RAG integration and hybrid function calling.

Note: These examples are for reference and documentation purposes.
      Run them in an async context (e.g., with asyncio.run()).
"""

import asyncio
from typing import Optional, List
from pydantic import BaseModel, Field

# Pydantic AI imports
from pydantic_ai import Agent, Tool

# Local LLM imports
from llm import LLMConfig, ModelPool, create_runner
from llm.knowledge import ChromaKnowledgeStore, ChromaKnowledgeStoreConfig
from llm.pyai import (
    LocalLLMModel,
    create_knowledge_search_tool,
)
from llm.pyai.tools import inject_search_context, MultiToolRegistry


# =============================================================================
# Example 0: Function Calling Detection (NEW - HYBRID MODE)
# =============================================================================

async def example_function_calling_detection():
    """
    NEW: Demonstrate hybrid function calling detection.
    
    Shows how the system automatically detects if a model supports
    native function calling or falls back to prompt engineering.
    """
    print("="*60)
    print("Function Calling Detection")
    print("="*60)
    
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
    )
    
    pool = ModelPool.get_instance(max_models=1)
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        
        # Automatic detection
        supports_fc = runner.supports_function_calling()
        
        print(f"\nModel: {runner.model_name}")
        print(f"Backend: {handle.backend}")
        print(f"Supports native function calling: {supports_fc}")
        print(f"Mode: {'NATIVE' if supports_fc else 'PROMPT_ENGINEERING'}")
        
        # The LocalLLMModel will automatically use the appropriate mode
        model = LocalLLMModel(runner)
        print(f"\nLocalLLMModel initialized with {('native' if supports_fc else 'prompt engineering')} mode")


# =============================================================================
# Example 1: Basic Usage - Simple Agent
# =============================================================================

async def example_basic_usage():
    """
    Basic example: Create a Pydantic AI agent with local LLM.
    
    This demonstrates the simplest integration without RAG.
    """
    # Configure model
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
        temperature=0.7,
    )
    
    # Get model from pool
    pool = ModelPool.get_instance(max_models=1)
    
    async with pool.acquire_context(config) as handle:
        # Create runner and model
        runner = create_runner(handle)
        model = LocalLLMModel(runner)
        
        # Create Pydantic AI agent
        agent = Agent(
            model=model,
            instructions="You are a helpful assistant. Be concise.",
        )
        
        # Run the agent
        result = await agent.run("What is the capital of France?")
        print(f"Response: {result.output}")
        
        # Access usage statistics
        print(f"Usage: {result.usage()}")


# =============================================================================
# Example 2: Streaming Response
# =============================================================================

async def example_streaming():
    """
    Streaming example: Get response chunks as they're generated.
    """
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)
        
        agent = Agent(
            model=model,
            instructions="You are a storyteller. Write engaging content.",
        )
        q = "Tell me a short story about a robot."
        print(f"Question: {q}")
        # Stream the response
        print("Streaming response:")
        try:
            async with agent.run_stream(q) as response:
                async for chunk in response.stream_text():
                    print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\nStreaming error: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Example 3: RAG with Native Tool (Agent-Controlled Search) - UPDATED
# =============================================================================

async def example_rag_native_tool():
    """
    RAG with native tool: Agent decides when to search.
    
    UPDATED: Now uses hybrid function calling - automatically detects
    if model supports native function calling or uses prompt engineering.
    """
    # Setup knowledge store
    kb_config = ChromaKnowledgeStoreConfig(
        persist_directory="./chroma_db",
        embedding_model_name="all-MiniLM-L6-v2",
    )
    knowledge_store = ChromaKnowledgeStore(kb_config)
    
    # Add some documents (if not already present)
    # knowledge_store.add_document_from_file("./docs/product_manual.pdf")
    
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
        temperature=0.3,  # Lower temp for more reliable tool calling
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store)
        model = LocalLLMModel(runner)
        
        # Check which mode will be used
        print(f"\nFunction calling mode: {'NATIVE' if runner.supports_function_calling() else 'PROMPT_ENGINEERING'}")
        
        # Create search tool
        search_tool = create_knowledge_search_tool(runner)
        
        # Create agent with tool
        agent = Agent(
            model=model,
            instructions=(
                "You are a helpful assistant with access to a knowledge base. "
                "When asked about documents or specific information, use the search_knowledge tool to find relevant information."
            ),
            tools=[search_tool],
        )
        
        # The agent will decide to use the tool (hybrid mode handles the rest)
        q = "What documents are available in the knowledge base? List all documents you can find."
        print(f"Question: {q}")
        result = await agent.run(q)
        print(f"Response: {result.output}")


# =============================================================================
# Example 4: RAG with Context Injection (Manual Search)
# =============================================================================

async def example_rag_context_injection():
    """
    RAG with context injection: Pre-search before agent run.
    
    You control when and what to search. The context is
    automatically included in the next generation.
    """
    kb_config = ChromaKnowledgeStoreConfig(persist_directory="./chroma_db")
    knowledge_store = ChromaKnowledgeStore(kb_config)
    
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store)
        model = LocalLLMModel(runner)
        
        agent = Agent(
            model=model,
            instructions="You are a helpful assistant. Answer based on the provided context.",
        )
        
        # Manually inject search context
        search_result = await inject_search_context(
            runner,
            query="documents content overview",  # Generic query
            top_k=5,
        )
        
        if search_result.success:
            print(f"Found {search_result.metadata.get('results_count', 0)} relevant sections")
        
        # Run agent - context is automatically included
        result = await agent.run("What information is available in the knowledge base?")
        print(f"Response: {result.output}")


# =============================================================================
# Example 5: Scoped Search (Session Documents)
# =============================================================================

async def example_scoped_search():
    """
    Scoped search: Limit search to specific documents.
    
    Useful when working with session-specific documents,
    like files uploaded in a chat session.
    """
    from llm.pyai.tools import create_scoped_knowledge_search_tool
    
    kb_config = ChromaKnowledgeStoreConfig(persist_directory="./chroma_db")
    knowledge_store = ChromaKnowledgeStore(kb_config)
    
    # Get actual document IDs from the knowledge store
    documents = knowledge_store.list_documents()
    if not documents:
        print("No documents in knowledge store. Please add documents first.")
        return
    
    # Use real document IDs from the store (take first 2)
    session_document_ids = {doc["id"] for doc in documents[:2]}
    print(f"Scoping search to documents: {session_document_ids}")
    
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store)
        model = LocalLLMModel(runner)
        
        # Create scoped search tool
        scoped_search = create_scoped_knowledge_search_tool(
            runner,
            document_ids=session_document_ids,
            name="search_session_docs",
            description="Search documents uploaded in this session.",
        )
        
        agent = Agent(
            model=model,
            instructions="Answer questions using only the uploaded documents.",
            tools=[scoped_search],
        )
        
        result = await agent.run("What content is available in these documents?")
        print(f"Response: {result.output}")


# =============================================================================
# Example 6: Structured Output (FIXED - Using Tool-based approach)
# =============================================================================

class DocumentSummary(BaseModel):
    """Structured output for document information."""
    title: str = Field(description="The title or name of the document")
    summary: str = Field(description="A brief summary of the document content")
    topics: List[str] = Field(description="List of main topics covered in the document")


async def example_structured_output():
    """
    Structured output: Get typed responses from the agent.
    
    Uses Pydantic AI's output_type with a tool-based approach for models 
    without native function calling support.
    """
    kb_config = ChromaKnowledgeStoreConfig(persist_directory="./chroma_db")
    knowledge_store = ChromaKnowledgeStore(kb_config)
    
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
        temperature=0.1,  # Low temperature for more deterministic output
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store)
        model = LocalLLMModel(runner)
        
        search_tool = create_knowledge_search_tool(runner)
        
        # Create agent with search tool and structured output type
        agent = Agent(
            model=model,
            instructions=(
                "You are a document analyst. Search the knowledge base and provide a structured summary. "
                "First use search_knowledge to find documents, then return a DocumentSummary."
            ),
            tools=[search_tool],
            output_type=DocumentSummary,
        )
        
        try:
            result = await agent.run("Search the knowledge base and summarize the main document.")
            summary: DocumentSummary = result.output
            print(f"Title: {summary.title}")
            print(f"Summary: {summary.summary}")
            print(f"Topics: {', '.join(summary.topics)}")
        except Exception as e:
            print(f"Structured output failed: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Example 7: Multi-Tool Agent
# =============================================================================

async def example_multi_tool():
    """
    Multi-tool agent: Combine knowledge search with other tools.
    
    Shows how to use MultiToolRegistry for managing multiple tools.
    """
    kb_config = ChromaKnowledgeStoreConfig(persist_directory="./chroma_db")
    knowledge_store = ChromaKnowledgeStore(kb_config)
    
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
        temperature=0.3,
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store)
        model = LocalLLMModel(runner)
        
        # Use MultiToolRegistry for convenient tool management
        tools = MultiToolRegistry(runner)
        tools.add_knowledge_search(
            name="search_docs",
            description="Search the document knowledge base.",
        )
        
        # Add a custom calculation tool with proper builtins
        async def calculate(expression: str) -> str:
            """Evaluate a mathematical expression safely."""
            try:
                # Allow safe math operations
                allowed_names = {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len, "pow": pow,
                    "int": int, "float": float,
                }
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        tools.add_custom_tool("calculate", calculate)
        
        agent = Agent(
            model=model,
            instructions=(
                "You are a helpful assistant with access to documents and a calculator. "
                "Use search_docs to find information and calculate for math operations."
            ),
            tools=tools.get_tools(),
        )
        
        result = await agent.run(
            "Calculate the sum of 10, 20, 30, 40, and 50."
        )
        print(f"Response: {result.output}")


# =============================================================================
# Example 8: Conversation with History
# =============================================================================

async def example_conversation():
    """
    Conversation: Multi-turn dialogue with context.
    
    Shows how to maintain conversation history across turns.
    """
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)
        
        agent = Agent(
            model=model,
            instructions="You are a helpful coding assistant.",
        )
        
        # First turn
        result1 = await agent.run("How do I read a file in Python?")
        print(f"Assistant: {result1.output}\n")
        
        # Second turn - continues conversation
        result2 = await agent.run(
            "How do I handle errors in that code?",
            message_history=result1.all_messages(),
        )
        print(f"Assistant: {result2.output}\n")
        
        # Third turn
        result3 = await agent.run(
            "Can you show me a complete example?",
            message_history=result2.all_messages(),
        )
        print(f"Assistant: {result3.output}")


# =============================================================================
# Example 9: Custom Model Settings
# =============================================================================

async def example_custom_settings():
    """
    Custom settings: Override temperature and other parameters.
    """
    from llm.pyai.model import LocalModelSettings
    
    config = LLMConfig(
        model_name="bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
    )
    
    pool = ModelPool.get_instance()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        model = LocalLLMModel(runner)
        
        # Creative agent with high temperature
        creative_settings = LocalModelSettings(
            temperature=0.9,
            top_p=0.95,
        )
        
        creative_agent = Agent(
            model=model,
            instructions="You are a creative writer. Be imaginative and expressive.",
            model_settings=creative_settings,
        )
        
        # Precise agent with low temperature
        precise_settings = LocalModelSettings(
            temperature=0.1,
            top_p=0.5,
        )
        
        precise_agent = Agent(
            model=model,
            instructions="You are a factual assistant. Be precise and accurate.",
            model_settings=precise_settings,
        )
        
        # Compare outputs
        creative_result = await creative_agent.run("Describe a sunset.")
        print(f"Creative: {creative_result.output}\n")
        
        precise_result = await precise_agent.run("Describe a sunset.")
        print(f"Precise: {precise_result.output}")


# =============================================================================
# Run Examples
# =============================================================================

async def run_all_examples():
    """Run all examples sequentially."""
    examples = [
        ("Function Calling Detection", example_function_calling_detection),
        ("Basic Usage", example_basic_usage),
        ("Streaming", example_streaming),
        ("RAG Native Tool", example_rag_native_tool),
        ("RAG Context Injection", example_rag_context_injection),
        ("Scoped Search", example_scoped_search),
        ("Structured Output", example_structured_output),
        ("Multi-Tool", example_multi_tool),
        ("Conversation", example_conversation),
        ("Custom Settings", example_custom_settings),
    ]
    
    
    for name, example_fn in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        try:
            await example_fn()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Run a specific example
    # asyncio.run(example_function_calling_detection())
    
    # Or run all examples
    asyncio.run(run_all_examples())