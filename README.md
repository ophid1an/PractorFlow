# PractorFlow

**Private, Self-Hosted AI for Organizations That Can't Share Their Data**

PractorFlow is an open-source AI service that organizations can deploy inside their own infrastructure with full control. Built for regulated industries, consulting firms, and enterprises handling confidential data—where privacy is not a feature, but a requirement.

## 🎯 The Problem

Many companies want the productivity gains of tools like ChatGPT or Claude but legally, contractually, or ethically cannot send sensitive data to third-party AI APIs. For AI to be adopted responsibly in these environments, it must run where the data already lives.

## 💡 The Solution

PractorFlow is a production-ready, self-hosted AI service designed for real business workflows. Think of it as a private internal AI assistant that never leaves your environment.

### What This Enables

- **AI assistants that never leave your environment** - All inference and data processing happens on your infrastructure
- **Full ownership of data, prompts, and outputs** - Complete control over your AI interactions
- **Support for agentic workflows** - AI that reasons across documents and tasks
- **No vendor lock-in** - Built on open standards and open-source models

## 🌟 Key Features

- **Multiple Backend Support**: Run GGUF models via llama.cpp or HuggingFace models via transformers
- **RAG with ChromaDB**: Persistent document storage with Small-to-Big chunking strategy for secure document ingestion and retrieval
- **Async Model Pooling**: Efficient resource management with LRU eviction for production deployments
- **Session Management**: Persistent conversation history and document context management
- **Pydantic AI Integration**: Drop-in compatibility with modern AI orchestration and agent frameworks
- **Web Search Tools**: Built-in DuckDuckGo (free) and SerpAPI (premium) support
- **Native Function Calling**: Automatic detection and support for models with native tool calling
- **Streaming Generation**: Real-time token streaming with async support
- **Document Processing**: Support for PDF, DOCX, PPTX, XLSX, images, and more via Docling
- **Comprehensive Logging**: Configurable logging levels per component for production monitoring

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for GPU acceleration)

## 🚀 Installation

```bash
pip install -r requirements.txt
```

The requirements.txt includes llama-cpp-python with CUDA 12.1 support. If you need a different CUDA version or CPU-only installation, modify the llama-cpp-python line in requirements.txt accordingly.

## ⚙️ Configuration

The library uses environment variables for configuration. Configuration is loaded from `config/options/options.env`:

```bash
# Logging Levels
LOG_RUNNER_LEVEL=INFO           # Runner component logging
LOG_DOC_LEVEL=INFO              # Document processing logging
LOG_KNOWLEDGE_LEVEL=INFO        # Knowledge store logging
LOG_MODEL_POOL_LEVEL=INFO       # Model pool logging
LOG_TOOL_LEVEL=INFO             # Tool execution logging
LOG_AGENT_LEVEL=INFO            # Agent/Pydantic AI logging

# LLM Model Configuration
LLM_MODEL=bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf
LLM_BACKEND=llama_cpp           # 'llama_cpp' or 'transformers'
LLM_DEVICE=auto                 # 'auto', 'cuda', 'cpu', or specific device
LLM_DTYPE=auto                  # 'auto', 'float32', 'float16', 'bfloat16', or None
LLM_MAX_NEW_TOKENS=2048         # Maximum tokens to generate
LLM_TEMPERATURE=0.7             # Sampling temperature (0.0-2.0)
LLM_TOP_P=0.9                   # Nucleus sampling parameter
LLM_MODELS_DIR=./models         # Directory for cached models

# Llama.cpp Specific Settings
LLM_GPU_LAYERS=-1               # Number of layers on GPU (-1 = all)
LLM_N_CTX=32768                 # Context window size
LLM_N_BATCH=2048                # Batch size for prompt processing

# Transformers Specific Settings
# LLM_QUANTIZATION=4bit         # Optional: '4bit' or '8bit' quantization

# Generation Settings
# LLM_STOP_TOKENS=</s>,<|endoftext|>  # Comma-separated stop tokens
LLM_MAX_SEARCH_RESULTS=5        # Default results for knowledge search

# Knowledge Database Configuration (ChromaDB)
KB_TYPE=chromadb                           # Knowledge store type
KB_CHROMA_PERSIST_DIRECTORY=./chroma_db   # Persistent storage location
KB_CHROMA_RETRIEVE_COLLECTION=knowledge_retrieval  # Retrieval chunks collection
KB_CHROMA_CONTEXT_COLLECTION=knowledge_context     # Context chunks collection
KB_CHROMA_DOCUMENT_COLLECTION=knowledge_documents  # Documents collection
KB_CHROMA_BATCH_SIZE=100                   # Batch size for operations
KB_CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2  # SentenceTransformer model
KB_CHROMA_EMBEDDING_MODEL_DIR=./models     # Embedding model cache directory

# Small-to-Big Chunking Strategy
KB_CHROMA_RETRIEVAL_CHUNK_SIZE=128        # Small chunk size for retrieval
KB_CHROMA_RETRIEVAL_CHUNK_OVERLAP=20      # Overlap for retrieval chunks
KB_CHROMA_CONTEXT_CHUNK_SIZE=1024         # Large chunk size for context
KB_CHROMA_CONTEXT_CHUNK_OVERLAP=100       # Overlap for context chunks
```

Copy `config/options/options.env` and modify values as needed for your setup.

## 🎯 Quick Start

### Basic Usage

```python
import asyncio
from llm import ModelPool, create_runner
from settings.app_settings import appConfiguration

async def main():
    config = appConfiguration.ModelConfiguration
    pool = ModelPool(max_models=1)
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        result = await runner.generate(prompt="What is Python?")
        print(result['reply'])

asyncio.run(main())
```

### Streaming Generation

```python
async with pool.acquire_context(config) as handle:
    runner = create_runner(handle)
    
    async for chunk in runner.generate_stream(prompt="Explain recursion"):
        if not chunk.finished:
            print(chunk.text, end="", flush=True)
        else:
            print(f"\n\nLatency: {chunk.latency_seconds:.2f}s")
```

### Secure Document Ingestion with RAG

```python
from llm.knowledge import ChromaKnowledgeStore
from settings.app_settings import appConfiguration

# Initialize knowledge store (all data stays local)
kb_config = appConfiguration.KnowledgeChromaConfiguration
knowledge_store = ChromaKnowledgeStore(kb_config)

# Add documents - data never leaves your infrastructure
doc_info = knowledge_store.add_document_from_file("confidential_report.pdf")

# Create runner with knowledge store
async with pool.acquire_context(config) as handle:
    runner = create_runner(handle, knowledge_store=knowledge_store)
    
    # Set document scope for session-specific context
    runner.set_document_scope({doc_info['id']})
    
    # Search and generate - entirely within your environment
    search_result = runner.search("What are the key findings?")
    result = await runner.generate(prompt="Summarize the recommendations")
    print(result['reply'])
```

### Agentic Workflows with Pydantic AI

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from llm.pyai import LocalLLMModel, KnowledgeDeps, search_knowledge

@dataclass
class Deps:
    knowledge_store: ChromaKnowledgeStore
    document_scope: set[str] | None = None

async with pool.acquire_context(config) as handle:
    runner = create_runner(handle)
    model = LocalLLMModel(runner)
    
    # Create agent that can reason across documents
    agent = Agent(
        model=model, 
        deps_type=KnowledgeDeps,
        system_prompt="You are a secure AI assistant that helps analyze internal documents."
    )
    agent.tool(search_knowledge)
    
    # Run with dependencies - all processing stays local
    deps = KnowledgeDeps(knowledge_store=knowledge_store)
    result = await agent.run(
        "What patterns do you see across our quarterly reports?",
        deps=deps
    )
    print(result.output)
```

### Web Search Integration (Optional)

```python
from llm.tools import DuckDuckGoSearchTool

@dataclass
class WebSearchDeps:
    web_search_tool: DuckDuckGoSearchTool

agent = Agent(model=model, deps_type=WebSearchDeps)

@agent.tool
async def search_web(ctx: RunContext[WebSearchDeps], query: str) -> str:
    """Search the web for current information."""
    result = ctx.deps.web_search_tool.execute(query=query, max_results=5)
    return result.data if result.success else "No results found"

deps = WebSearchDeps(web_search_tool=DuckDuckGoSearchTool())
result = await agent.run(
    "What are industry best practices for data governance?",
    deps=deps
)
```

## 📚 Examples

Comprehensive examples are provided in:
- `sample.py` - Basic usage, streaming, RAG workflows
- `pyai-examples.py` - Pydantic AI integration patterns

Run examples:
```bash
# Basic examples
python sample.py

# With document
python sample.py path/to/document.pdf

# Pydantic AI examples
python pyai-examples.py
```

## 🏗️ Architecture

### Core Components

```
llm/
├── base/               # Abstract base classes
│   ├── llm_runner.py   # Base runner interface
│   └── session.py      # Session management
├── pool/               # Model pooling
│   ├── model_pool.py   # Async model pool with LRU
│   └── model_handle.py # Model wrapper
├── knowledge/          # RAG components
│   ├── knowledge_store.py        # Abstract store interface
│   ├── chroma_knowledge_store.py # ChromaDB implementation
│   └── chroma_knowledge_config.py
├── document/           # Document processing
│   ├── document_loader.py # File parsing & chunking
│   └── embeddings.py      # Embedding models
├── tools/              # Tool system
│   ├── base.py               # Tool interface
│   ├── tool_registry.py      # Tool management
│   ├── knowledge_search.py   # RAG search tool
│   ├── base_web_search.py    # DuckDuckGo tool
│   └── serpapi_web_search.py # SerpAPI tool
├── pyai/               # Pydantic AI integration
│   ├── model.py              # LocalLLMModel implementation
│   ├── stream_response.py    # Streaming support
│   ├── message_converter.py  # Message format conversion
│   └── tools.py              # Pydantic AI tool helpers
├── llama_cpp_runner.py    # Llama.cpp backend
├── transformers_runner.py # Transformers backend
├── factory.py             # Runner factory
└── llm_config.py          # Configuration dataclass
```

### Small-to-Big Chunking Strategy

PractorFlow implements an efficient RAG strategy for secure document retrieval:

1. **Small Chunks (Retrieval)**: Documents are split into small chunks (128 chars) with embeddings for precise similarity search
2. **Large Chunks (Context)**: Parent chunks (1024 chars) provide rich context for the LLM
3. **Parent-Child Linking**: Small chunks reference their parent chunks
4. **Search Flow**: Search small chunks → Deduplicate by parent → Return parent chunks

This approach balances embedding quality with context richness while maintaining data locality.

## 🔧 Advanced Usage

### Custom Backend Selection

```python
# Use transformers backend with quantization
from llm import LLMConfig

config = LLMConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    backend="transformers",
    device="auto",
    dtype="auto",
    quantization="4bit"  # Optional 4-bit/8-bit quantization
)
```

### Model Pool for Production Deployments

```python
from llm import ModelPool

# Initialize pool with multiple models for concurrent requests
pool = ModelPool(max_models=3)

# Preload models at startup
await pool.preload(config1)
await pool.preload(config2)

# Acquire models concurrently
async with pool.acquire_context(config) as handle:
    # Model is automatically released on exit
    runner = create_runner(handle, knowledge_store)
    result = await runner.generate(prompt="Hello")
```

### Session-Based Document Scope

```python
# Add multiple documents for a specific session/user
session_doc_ids = set()
for filepath in ["contract.pdf", "policy.docx", "memo.txt"]:
    doc_info = knowledge_store.add_document_from_file(filepath)
    session_doc_ids.add(doc_info['id'])

# Scope search to session-specific documents
runner.set_document_scope(session_doc_ids)

# All searches are isolated to this session's documents
result = runner.search("What are the terms?")
```

### Native Function Calling

The library automatically detects and uses native function calling when supported by the model:

```python
# Check if model supports native function calling
if runner.supports_function_calling():
    print("Model has native tool calling support")

# Define tools in OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_data",
            "description": "Retrieve customer information from internal database",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"}
                },
                "required": ["customer_id"]
            }
        }
    }
]

# Generate with tools
result = await runner.generate(
    prompt="What's the status of customer ABC123?",
    tools=tools
)

# Check for tool calls
if 'tool_calls' in result:
    for tc in result['tool_calls']:
        print(f"Tool: {tc['tool_name']}, Args: {tc['args']}")
```

## 🧪 Testing

```bash
# Run basic examples
python sample.py

# Run Pydantic AI examples
python pyai-examples.py

# Test with document
python sample.py path/to/test.pdf
```

## 📊 Performance Tips

1. **GPU Acceleration**: Set `LLM_GPU_LAYERS=-1` to offload all layers to GPU
2. **Context Size**: Adjust `LLM_N_CTX` based on your model and GPU memory
3. **Batch Size**: Tune `LLM_N_BATCH` for optimal throughput
4. **Quantization**: Use 4-bit quantization for large models on limited hardware
5. **Model Pooling**: Set appropriate `max_models` based on available VRAM
6. **Embedding Model**: Use `all-MiniLM-L6-v2` for speed or `all-mpnet-base-v2` for quality

## 🎯 Who This Is For

- **Regulated Industries**: Healthcare, finance, legal—where data privacy is mandated
- **Consulting Firms**: Working with confidential client data
- **Enterprises**: Organizations with strict data governance requirements
- **Government Agencies**: Public sector with security clearance needs
- **Research Organizations**: Academic institutions with sensitive research data

## 🤝 Contributing

Contributions are welcome! We appreciate your help in building a better private AI service for organizations.

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and guidelines
- Code style and architecture principles
- Privacy considerations
- How to submit pull requests
- Areas where we need help

For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Apache 2.0 allows you to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Use patents granted by contributors

With the requirement to:
- 📄 Include license and copyright notice
- 📝 State significant changes made

## 🙏 Acknowledgments

- **llama.cpp** - Efficient GGUF model inference
- **Transformers** - HuggingFace model support
- **ChromaDB** - Vector database for RAG
- **Docling** - Advanced document parsing
- **Pydantic AI** - Type-safe agent framework
- **sentence-transformers** - High-quality embeddings

Built on modern open-source models including Qwen, Mistral, and others.

## 📧 Support

For questions, issues, or feature requests, please [open an issue](https://github.com/yourusername/practorflow/issues) on GitHub.

---

**Privacy by Design**: PractorFlow is built for organizations where AI must run where the data already lives. All inference, document processing, and reasoning happens entirely within your infrastructure. No data ever leaves your environment.