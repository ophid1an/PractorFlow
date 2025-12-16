from llm import LLMConfig, create_runner, Session
import sys


def main():
    # Parse command line arguments
    document_file = None
    if len(sys.argv) > 1:
        document_file = sys.argv[1]
    
    # Configuration for Qwen3-VL-8B-Instruct with Q4_K_M quantization
    # Model will be auto-downloaded from HuggingFace
    config = LLMConfig(
        model_name="Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        backend="llama_cpp",
        n_ctx=8192,
        n_gpu_layers=-1,  # Use all GPU layers (fits in RTX 4070 8GB)
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=1024,
    )
    
    print(f"Loading model: {config.model_name}")
    print(f"Backend: {config.backend}")
    print(f"Device: {config.device}")
    print()
    
    runner = create_runner(config)
    print("Model loaded successfully!")
    print()
    
    # Example 1: Single prompt (backward compatibility)
    print("=" * 60)
    print("EXAMPLE 1: Single prompt")
    print("=" * 60)
    prompt = "What is Qwen3-VL in one sentence?"
    print(f"Prompt: {prompt}")
    print()
    
    result = runner.generate(prompt=prompt)
    print(f"Response: {result['text']}")
    print(f"Tokens: {result['usage']['total_tokens']}")
    print(f"Latency: {result['latency_seconds']:.2f}s")
    print()
    
    # Example 2: Messages API (chat history)
    print("=" * 60)
    print("EXAMPLE 2: Messages API with chat history")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "What are vision-language models?"},
    ]
    
    print(f"User: {messages[0]['content']}")
    result1 = runner.generate(messages=messages)
    print(f"Assistant: {result1['text']}")
    print()
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": result1['text']})
    
    # Continue conversation
    messages.append({"role": "user", "content": "What can they do?"})
    print(f"User: {messages[-1]['content']}")
    result2 = runner.generate(messages=messages)
    print(f"Assistant: {result2['text']}")
    print()
    
    # Example 3: Messages with instructions
    print("=" * 60)
    print("EXAMPLE 3: Messages with system instructions")
    print("=" * 60)
    
    instructions = "You are a helpful AI assistant. Keep answers concise."
    messages = [
        {"role": "user", "content": "Explain OCR briefly."},
    ]
    
    print(f"Instructions: {instructions}")
    print(f"User: {messages[0]['content']}")
    result3 = runner.generate(messages=messages, instructions=instructions)
    print(f"Assistant: {result3['text']}")
    print()
    
    # Example 4: Document loading with RAG (if document provided)
    if document_file:
        print("=" * 60)
        print("EXAMPLE 4: Document loading with RAG")
        print("=" * 60)
        
        # Create session
        session = Session(session_id="qwen3vl_session")
        
        # Create runner with session
        runner_with_session = create_runner(config, session=session)
        
        # Load document - auto chunks and embeds
        print(f"Loading document: {document_file}")
        runner_with_session.load_document(document_file)
        
        print(f"Vector store has {runner_with_session.vector_store.count()} chunks")
        print()
        
        # Ask question - uses semantic search
        question = "What is this document about?"
        print(f"Question: {question}")
        result4 = runner_with_session.generate(
            prompt=question,
            use_context=True
        )
        
        print(f"Response: {result4['text']}")
        if 'context_sources' in result4:
            print(f"\nSources used:")
            for src in result4['context_sources']:
                print(f"  - {src['filename']} (similarity: {src.get('similarity', src.get('best_similarity', 0)):.3f})")
        print()
        
        # Example 5: Another question with limits
        print("=" * 60)
        print("EXAMPLE 5: RAG with context limits")
        print("=" * 60)
        
        question2 = "Summarize the key points."
        print(f"Question: {question2}")
        result5 = runner_with_session.generate(
            prompt=question2,
            use_context=True,
            max_context_docs=5
        )
        
        print(f"Response: {result5['text']}")
        if 'context_sources' in result5:
            print(f"\nRetrieved {len(result5['context_sources'])} chunks")
        print()
        
        # Show stats
        print("=" * 60)
        print("VECTOR STORE STATISTICS")
        print("=" * 60)
        stats = runner_with_session.vector_store.get_stats()
        print(f"Total chunks: {stats['total_entries']}")
        print(f"Embedding dimension: {stats['dimension']}")
        print()
        
    else:
        print()
        print("=" * 60)
        print("INFO: No document file provided")
        print("=" * 60)
        print("To test RAG, run:")
        print("  python test_qwen3vl.py README.md")
        print("  python test_qwen3vl.py document.pdf")
        print()


if __name__ == "__main__":
    main()