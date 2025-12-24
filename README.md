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

### Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for GPU acceleration)

### Basic Installation

```bash
pip install -r requirements.txt
```

The requirements.txt includes llama-cpp-python with CUDA 12.1 support. If you need a different CUDA version or CPU-only installation, modify the llama-cpp-python line in requirements.txt accordingly.

### Enabling Qwen3 and Mistral3 Support

PractorFlow supports the latest Qwen3VL and Mistral3 models, which require a specialized build of llama-cpp-python.

#### Why Special Support is Needed

- The official llama-cpp-python releases do not yet include support for Qwen3VL and Mistral3 architectures
- These newer models use updated attention mechanisms and vision encoders
- Attempting to load these models with standard llama-cpp-python will result in unsupported architecture errors

#### Installation Options

**Option 1: Prebuilt Wheels (Recommended)**

For Python 3.12 (CUDA 12.8, Linux/WSL2):

```bash
# Edit requirements.txt - comment out the standard llama-cpp-python line:
# llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Uncomment this line:
llama-cpp-python @ https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.18-cu128-AVX2-linux-20251220/llama_cpp_python-0.3.18-cp312-cp312-linux_x86_64.whl

# Then install:
pip install -r requirements.txt
```

For Python 3.10 (CUDA 12.8, Linux/WSL2):

```bash
# Use the Python 3.10 wheel instead:
llama-cpp-python @ https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.18-cu128-AVX2-linux-20251220/llama_cpp_python-0.3.18-cp310-cp310-linux_x86_64.whl
```

**Option 2: Build from Source (Advanced)**

If prebuilt wheels don't work for your system:

1. Check your CUDA version:
```bash
nvcc --version
```

2. Determine your GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

3. Build and install (replace '89' with your GPU's compute capability):
```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=$(which nvcc)" \
pip install git+https://github.com/JamePeng/llama-cpp-python.git --no-cache-dir
```

**Common GPU Compute Capabilities:**
- RTX 4090/4080/4070/4060: 89
- RTX 3090/3080/3070/3060: 86
- RTX 2080 Ti/2080: 75
- GTX 1080 Ti/1080: 61

#### Verification

After installation, verify the custom build:

```bash
# Check llama-cpp-python version
python -c "import llama_cpp; print(f'llama-cpp-python version: {llama_cpp.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test model loading (replace with your model path)
python -c "
from llama_cpp import Llama
model = Llama(model_path='path/to/Qwen3VL-8B-Instruct-Q4_K_M.gguf', n_gpu_layers=-1)
print('Model loaded successfully!')
"
```

#### Troubleshooting

**Error: "Unsupported architecture"**
- Solution: Ensure you're using the custom llama-cpp-python wheel, not the standard version

**Error: "CUDA out of memory"**
- Reduce context window size (n_ctx) in your model configuration
- Reduce GPU layers (n_gpu_layers) or use a smaller quantization
- Close other GPU-intensive applications

**Error: "Model loading takes too long"**
- First load is always slower due to model download
- Subsequent loads use cached models from `./models` directory
- Consider using `pool.preload()` at server startup

**Error: "ImportError: cannot import name 'Llama'"**
- Reinstall llama-cpp-python:
```bash
pip uninstall llama-cpp-python
pip install --no-cache-dir <wheel-url-or-build-command>
```

**Error: "CMAKE_CUDA_COMPILER not found"**
- Install CUDA toolkit:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

#### Performance Tips

1. **Quantization Selection:**
   - Q4_K_M: Best balance of quality/speed (recommended)
   - Q5_K_M: Slightly better quality, slower
   - Q3_K_M: Faster, lower quality (good for testing)

2. **Context Window:**
   - Use smaller contexts (8192-16384) if you don't need full 32K
   - Larger contexts consume more VRAM and slow inference

3. **Batch Size:**
   - Increase n_batch for faster prompt processing
   - Decrease if you hit VRAM limits

4. **GPU Layers:**
   - Start with -1 (all layers on GPU)
   - If VRAM limited, try 30-40 layers
   - Monitor with `nvidia-smi` to find optimal balance

#### Model Sources

**Qwen3VL Models:**
- [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- Multimodal (text + vision) capabilities
- 8B parameters, fits on 24GB GPU with Q4 quantization

**Mistral3 Models:**
- [mistralai/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF)
- Text-only, improved architecture over Mistral 7B
- 8B parameters, excellent quality/efficiency ratio

#### Switching Back to Standard Models

To revert to standard llama-cpp-python for other models:

```bash
# Edit requirements.txt - comment out custom wheel
# Uncomment standard version:
llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Reinstall:
pip install --force-reinstall -r requirements.txt
```

#### Additional Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Custom llama-cpp-python builds](https://github.com/JamePeng/llama-cpp-python)
- [GGUF Model Hub](https://huggingface.co/models?search=gguf)
- [Qwen Documentation](https://qwen.readthedocs.io/)
- [Mistral Documentation](https://docs.mistral.ai/)

**Note:** The custom llama-cpp-python builds are community-maintained until official support is merged. Always verify the source and use official releases when available for production deployments.

## ⚙️ Configuration

PractorFlow uses environment variables for configuration, loaded from three separate files in the `config/options/` directory:

### Configuration Files

#### 1. Logger Configuration (`config/options/logger.env`)

Controls logging levels for different components:

```bash
# Logging Levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_RUNNER_LEVEL=INFO           # Runner component logging
LOG_DOC_LEVEL=INFO              # Document processing logging
LOG_KNOWLEDGE_LEVEL=INFO        # Knowledge store logging
LOG_MODEL_POOL_LEVEL=INFO       # Model pool logging
LOG_TOOL_LEVEL=INFO             # Tool execution logging
LOG_AGENT_LEVEL=INFO            # Agent/Pydantic AI logging
```

#### 2. Model Configuration (`config/options/model.env`)

Configures the LLM model and inference settings:

```bash
# Model Selection
LLM_MODEL=bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf # Model name
LLM_BACKEND=llama_cpp         # 'llama_cpp' or 'transformers'
LLM_DEVICE=auto                  # 'auto', 'cuda', 'cpu', or specific device
LLM_DTYPE=auto                   # 'auto', 'float32', 'float16', 'bfloat16', or None
LLM_MAX_NEW_TOKENS=2048          # Maximum tokens to generate
LLM_TEMPERATURE=0.7              # Sampling temperature (0.0-2.0)
LLM_TOP_P=0.9                    # Nucleus sampling parameter
LLM_MODELS_DIR=./models          # Directory for cached models

# Llama.cpp Specific Settings (when LLM_BACKEND=llama_cpp)
LLM_GPU_LAYERS=-1                # Number of layers on GPU (-1 = all)
LLM_N_CTX=32768                  # Context window size
LLM_N_BATCH=2048                 # Batch size for prompt processing

# Transformers Specific Settings (when LLM_BACKEND=transformers)
# LLM_QUANTIZATION=4bit            # Optional: '4bit' or '8bit' quantization

# Generation Settings
# LLM_STOP_TOKENS=</s>,<|endoftext|>  # Comma-separated stop tokens
LLM_MAX_SEARCH_RESULTS=5         # Default results for knowledge search

# =============================================================================
# Transformers Backend Optimizations
# =============================================================================
# These settings only apply when LLM_BACKEND=transformers
# They are ignored for llama_cpp backend (GGUF models)

# Enable torch.compile() for faster inference (PyTorch 2.0+)
# Compiles the model into optimized kernels for your GPU
# First inference triggers JIT compilation, subsequent calls are faster
# Set to "false" to disable (useful for debugging or unsupported models)
# Default: true
# LLM_USE_TORCH_COMPILE=true

# torch.compile() optimization mode
# - "default"         : Fast compile (~10s), decent runtime speed
# - "reduce-overhead" : Balanced compile (~30s), good runtime speed (recommended)
# - "max-autotune"    : Slow compile (~3-5min), fastest runtime speed
# Default: reduce-overhead
# LLM_COMPILE_MODE=reduce-overhead

# Run warmup inference immediately after model loading
# Triggers JIT compilation during startup instead of on first request
# Improves first request latency at the cost of longer startup time
# Set to "false" if you prefer faster startup over first-request latency
# Default: true
# LLM_WARMUP_ON_LOAD=true
```

#### 3. Knowledge Database Configuration (`config/options/knowledge.env`)

Configures the ChromaDB-based knowledge store:

```bash
# Knowledge Database Type
KB_TYPE=chromadb                           # Knowledge store type

# ChromaDB Settings
KB_CHROMA_PERSIST_DIRECTORY=./chroma_db   # Persistent storage location
KB_CHROMA_RETRIEVE_COLLECTION=knowledge_retrieval  # Retrieval chunks collection
KB_CHROMA_CONTEXT_COLLECTION=knowledge_context     # Context chunks collection
KB_CHROMA_DOCUMENT_COLLECTION=knowledge_documents  # Documents collection
KB_CHROMA_BATCH_SIZE=100                   # Batch size for operations

# Embedding Model
KB_CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2  # SentenceTransformer model
KB_CHROMA_EMBEDDING_MODEL_DIR=./models      # Embedding model cache directory

# Small-to-Big Chunking Strategy
KB_CHROMA_RETRIEVAL_CHUNK_SIZE=128        # Small chunk size for retrieval
KB_CHROMA_RETRIEVAL_CHUNK_OVERLAP=20      # Overlap for retrieval chunks
KB_CHROMA_CONTEXT_CHUNK_SIZE=1024         # Large chunk size for context
KB_CHROMA_CONTEXT_CHUNK_OVERLAP=100       # Overlap for context chunks
```

### Sample Model Configurations

Sample configurations for various models are provided in `config/options/samples/models/`. You can use these as templates or copy them directly to `config/options/model.env`:

**Available Sample Configurations:**

1. **Qwen2-1.5B-Instruct-GGUF** (`model-Qwen2-1.5B-Instruct-GGUF.env`)
   - Small, fast model for testing
   - Backend: llama_cpp
   - Quantization: Q4_K_M

2. **Qwen2.5-7B** (`model-Qwen2.5-7B.env`)
   - Balanced quality/performance
   - Backend: transformers
   - Quantization: 4bit

3. **Qwen3-VL-8B-Instruct-GGUF** (`model-Qwen3-VL-8B-Instruct-GGUF.env`)
   - Multimodal (text + vision)
   - Backend: llama_cpp
   - Requires custom llama-cpp-python build (see Installation section)

4. **Ministral-3-8B-Instruct-2512-GGUF** (`model-Ministral-3-8B-Instruct-2512-GGUF.env`)
   - Mistral's latest architecture
   - Backend: llama_cpp
   - Requires custom llama-cpp-python build (see Installation section)

5. **GPT-OSS-20B** (`model-gpt-oss-20b.env`)
   - Large open-source model
   - Backend: transformers
   - Quantization: 4bit
   - Requires significant resources

**Using Sample Configurations:**

To use a sample configuration, copy it to your active model configuration file:

```bash
# Example: Use Qwen2.5-7B configuration
cp config/options/samples/models/model-Qwen2.5-7B.env config/options/model.env
```

Or manually copy the contents into `config/options/model.env` and modify as needed.

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
from llm.knowledge.chroma_knowledge_store import ChromaKnowledgeStore
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

## 🗃️ Architecture

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

## 📄 License

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