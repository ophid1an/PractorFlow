# Contributing to PractorFlow

Thank you for your interest in contributing to PractorFlow! We welcome contributions from the community to help build a better private AI service for organizations.

## 🎯 Project Vision

PractorFlow aims to provide a production-ready, self-hosted AI service for organizations that cannot share their data with third-party APIs. When contributing, please keep this vision in mind:

- **Privacy First**: All features should support fully local, private deployment
- **Production Ready**: Code should be robust, well-tested, and deployable
- **No Vendor Lock-in**: Use open standards and avoid proprietary dependencies
- **Real Business Workflows**: Focus on practical enterprise use cases

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/practorflow.git
   cd practorflow
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp config/options/options.env.example config/options/options.env
   # Edit options.env with your settings
   ```

4. **Run Tests**
   ```bash
   # Run basic examples
   python sample.py
   
   # Run Pydantic AI examples
   python pyai-examples.py
   ```

## 📝 How to Contribute

### Reporting Issues

- **Search First**: Check if the issue already exists
- **Be Specific**: Include error messages, logs, and reproduction steps
- **Environment Details**: Python version, OS, GPU/CPU info
- **Privacy Context**: If relevant, explain the privacy/security concern

### Suggesting Features

When proposing new features, consider:
- Does it align with the privacy-first mission?
- Is it useful for regulated industries or enterprise deployments?
- Does it introduce external dependencies or data leakage risks?
- Can it be implemented without vendor lock-in?

### Pull Requests

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the existing code style (PEP 8 for Python)
   - Use snake_case for functions/variables
   - Use PascalCase for classes
   - Add docstrings to all public functions and classes
   - Keep functions focused and single-purpose

3. **Test Your Changes**
   - Ensure existing examples still work
   - Test with both llama.cpp and transformers backends
   - Verify privacy: no external API calls unless explicitly optional

4. **Commit with Clear Messages**
   ```bash
   git commit -m "feat: add support for new document format"
   git commit -m "fix: resolve memory leak in model pool"
   git commit -m "docs: update RAG configuration examples"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   - Provide a clear description of changes
   - Reference any related issues
   - Explain testing performed

## 🏗️ Code Guidelines

### Architecture Principles

- **Modularity**: Keep components loosely coupled
- **Abstraction**: Use base classes for extensibility
- **Async-First**: Support async operations where blocking would hurt performance
- **Configuration**: Make behavior configurable via environment variables

### Code Style

```python
# Good: Clear, documented, single-purpose
async def generate_with_context(
    runner: LLMRunner,
    prompt: str,
    document_ids: set[str]
) -> Dict[str, Any]:
    """
    Generate response with document context.
    
    Args:
        runner: LLM runner instance
        prompt: User prompt
        document_ids: Set of document IDs for context
        
    Returns:
        Generation result with reply and metadata
    """
    runner.set_document_scope(document_ids)
    runner.search(prompt)
    return await runner.generate(prompt=prompt)
```

### Privacy Considerations

**CRITICAL**: Before adding any feature, verify:
- ✅ All data processing happens locally
- ✅ No telemetry or usage tracking
- ✅ No external API calls (unless explicitly opt-in)
- ✅ Credentials stored securely (never in code)
- ✅ Logs don't leak sensitive information

### Testing Checklist

- [ ] Code runs without errors
- [ ] Existing examples still work
- [ ] Works with llama.cpp backend
- [ ] Works with transformers backend
- [ ] No external data leakage
- [ ] Logs are appropriately leveled
- [ ] Documentation updated

## 📚 Areas for Contribution

### High Priority

- **Vector Store Backends**: Add support for Pinecone, Weaviate, Qdrant
- **Document Formats**: Expand support for additional file types
- **Deployment Guides**: Docker, Kubernetes, production best practices
- **Performance Optimization**: Reduce latency, improve throughput
- **API Server Example**: FastAPI implementation for production deployment

### Medium Priority

- **Evaluation Tools**: Built-in testing and quality assessment
- **Multi-modal Support**: Vision model integration
- **Additional Embedding Models**: Support for more embedding providers
- **Monitoring**: Metrics and observability tools
- **Migration Tools**: Import from other AI platforms

### Documentation

- **Tutorials**: Step-by-step guides for common workflows
- **Use Case Examples**: Industry-specific implementations
- **Architecture Diagrams**: Visual documentation
- **Security Guides**: Best practices for secure deployment
- **Performance Tuning**: Optimization recommendations

## 🎓 Learning Resources

### Understanding the Codebase

- **Entry Point**: Start with `sample.py` for basic usage patterns
- **Core Abstraction**: `llm/base/llm_runner.py` defines the runner interface
- **Model Pool**: `llm/pool/model_pool.py` handles resource management
- **RAG Implementation**: `llm/knowledge/chroma_knowledge_store.py` for document storage
- **Pydantic AI Bridge**: `llm/pyai/model.py` for agent integration

### Key Concepts

- **Small-to-Big Chunking**: See `llm/document/document_loader.py`
- **Tool Registry**: See `llm/tools/tool_registry.py`
- **Async Patterns**: Study context managers in `model_pool.py`
- **Message Conversion**: See `llm/pyai/message_converter.py`

## 🤝 Code of Conduct

- **Be Respectful**: Treat all contributors with respect
- **Be Constructive**: Provide actionable feedback
- **Be Patient**: Remember everyone is learning
- **Be Inclusive**: Welcome contributors of all backgrounds and skill levels

## 📧 Questions?

- **General Questions**: Open a GitHub discussion
- **Bug Reports**: Create an issue with the bug template
- **Feature Requests**: Create an issue with the feature template
- **Security Issues**: Email security@practorflow.org (do not open public issues)

## 🙏 Thank You

PractorFlow is built on the shoulders of giants. We're deeply grateful to the teams and communities behind:

### Core Infrastructure
- **[llama.cpp Team](https://github.com/ggerganov/llama.cpp)** - For making GGUF inference efficient and accessible
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - For the comprehensive transformer library and model hub
- **[ChromaDB Team](https://github.com/chroma-core/chroma)** - For the excellent vector database
- **[Pydantic AI](https://github.com/pydantic/pydantic-ai)** - For the type-safe agent framework

### Document Processing
- **[Docling Team](https://github.com/DS4SD/docling)** - For advanced document parsing capabilities
- **[sentence-transformers](https://github.com/UKPLab/sentence-transformers)** - For high-quality embedding models

### Models & Research
- **[Qwen Team (Alibaba)](https://github.com/QwenLM/Qwen)** - For excellent open-source language models
- **[Mistral AI](https://mistral.ai/)** - For powerful open-weight models
- **[The GGUF Community](https://huggingface.co/TheBloke)** - For quantized model conversions

### Supporting Libraries
- **PyTorch Team** - For the foundational deep learning framework
- **NumPy & SciPy Communities** - For numerical computing foundations
- **Python Core Developers** - For the language that makes this possible

### Special Recognition
- **Open Source AI Community** - For democratizing AI and proving that privacy-preserving AI is possible
- **All Contributors** - Every PR, issue report, and discussion helps make PractorFlow better

Your contributions, whether code, documentation, bug reports, or discussions, help organizations adopt AI responsibly while maintaining data privacy. Thank you for being part of this mission.

---

**Remember**: Every contribution, no matter how small, helps organizations maintain their data privacy while gaining AI productivity. You're helping build the future of responsible AI deployment.