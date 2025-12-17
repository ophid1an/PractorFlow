"""
Sample usage of LLM module with async support, model pooling, and RAG.

Demonstrates:
- Model pooling with async acquire/release
- Async generation (non-blocking)
- Async streaming generation
- Messages API with chat history
- Document management via KnowledgeStore
- Knowledge search tool with scoped search
"""

import asyncio
import sys

from llm import LLMConfig, ModelPool, create_runner
from llm.knowledge import ChromaKnowledgeStore, ChromaKnowledgeStoreConfig


async def main():
    # Parse command line arguments
    document_file = None
    if len(sys.argv) > 1:
        document_file = sys.argv[1]
    
    config = LLMConfig()
    
    print(f"Loading model: {config.model_name}")
    print(f"Backend: {config.backend}")
    print(f"Device: {config.device}")
    print()
    
    # Initialize knowledge store
    knowledge_config = ChromaKnowledgeStoreConfig(
        persist_directory="./knowledge_db",
        embedding_model_name="all-MiniLM-L6-v2",
    )
    knowledge_store = ChromaKnowledgeStore(knowledge_config)
    print(f"Knowledge store: {knowledge_store.count_documents()} documents, {knowledge_store.count_chunks()} chunks")
    print()
    
    # Initialize model pool
    pool = ModelPool(max_models=1)
    
    # Preload model at startup
    print("Preloading model...")
    await pool.preload(config)
    print("Model preloaded!")
    print()
    
    # Example 1: Single prompt (async)
    print("=" * 60)
    print("EXAMPLE 1: Single prompt (async)")
    print("=" * 60)
    prompt = "What is Python in one sentence?"
    print(f"Prompt: {prompt}")
    print()
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        result = await runner.generate(prompt=prompt)
        print(f"Response: {result['reply']}")
        print(f"Latency: {result['latency_seconds']:.2f}s")
    print()
    
    # Example 2: Messages API (chat history)
    print("=" * 60)
    print("EXAMPLE 2: Messages API with chat history")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "What are the main programming paradigms?"},
    ]
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        
        print(f"User: {messages[0]['content']}")
        result1 = await runner.generate(messages=messages)
        print(f"Assistant: {result1['reply']}")
        print()
        
        # Add assistant response to history
        assistant_text = result1['reply']
        if isinstance(assistant_text, dict) and 'choices' in assistant_text:
            assistant_text = assistant_text['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": assistant_text})
        
        # Continue conversation
        messages.append({"role": "user", "content": "Which one is best for beginners?"})
        print(f"User: {messages[-1]['content']}")
        result2 = await runner.generate(messages=messages)
        print(f"Assistant: {result2['reply']}")
    print()
    
    # Example 3: Streaming generation (async)
    print("=" * 60)
    print("EXAMPLE 3: Async streaming generation")
    print("=" * 60)
    
    prompt = "Explain what a list comprehension is in Python."
    print(f"Prompt: {prompt}")
    print()
    print("Streaming response: ", end="", flush=True)
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        async for chunk in runner.generate_stream(prompt=prompt):
            if not chunk.finished:
                print(chunk.text, end="", flush=True)
            else:
                print()
                print()
                print(f"Finish reason: {chunk.finish_reason}")
                print(f"Latency: {chunk.latency_seconds:.2f}s")
                if chunk.usage:
                    print(f"Usage: {chunk.usage}")
    print()
    
    # Example 4: Streaming with instructions
    print("=" * 60)
    print("EXAMPLE 4: Streaming with system instructions")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "What is recursion?"},
    ]
    instructions = "You are a computer science teacher. Use simple examples."
    
    print(f"Instructions: {instructions}")
    print(f"User: {messages[0]['content']}")
    print()
    print("Streaming response: ", end="", flush=True)
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        async for chunk in runner.generate_stream(messages=messages, instructions=instructions):
            if not chunk.finished:
                print(chunk.text, end="", flush=True)
            else:
                print()
                print()
                print(f"Finish reason: {chunk.finish_reason}")
                print(f"Latency: {chunk.latency_seconds:.2f}s")
    print()
    
    # Example 5: RAG with document (if provided)
    if document_file:
        print("=" * 60)
        print("EXAMPLE 5: RAG with document")
        print("=" * 60)
        
        print(f"Adding document: {document_file}")
        doc_info = knowledge_store.add_document_from_file(document_file)
        
        print(f"Document ID: {doc_info['id']}")
        print(f"Retrieval chunks: {doc_info['retrieval_chunk_count']}")
        print(f"Context chunks: {doc_info['context_chunk_count']}")
        print()
        
        async with pool.acquire_context(config) as handle:
            runner = create_runner(handle, knowledge_store=knowledge_store)
            runner.set_document_scope({doc_info['id']})
            
            question = "What is this document about?"
            print(f"Question: {question}")
            
            search_result = runner.search(question)
            print(f"Search found {search_result.metadata.get('results_count', 0)} results")
            
            result = await runner.generate(prompt=question)
            
            print(f"Response<{result['latency_seconds']:.2f}s>: {result['reply']}")
            if 'search_metadata' in result:
                print(f"Documents used: {result['search_metadata'].get('document_ids', [])}")
        print()
        
        # Example 6: Streaming with RAG
        print("=" * 60)
        print("EXAMPLE 6: Async streaming with RAG context")
        print("=" * 60)
        
        question = "Summarize the key points from this document."
        print(f"Question: {question}")
        
        async with pool.acquire_context(config) as handle:
            runner = create_runner(handle, knowledge_store=knowledge_store)
            runner.set_document_scope({doc_info['id']})
            
            search_result = runner.search(question)
            print(f"Search found {search_result.metadata.get('results_count', 0)} results")
            print()
            print("Streaming response: ", end="", flush=True)
            
            async for chunk in runner.generate_stream(prompt=question):
                if not chunk.finished:
                    print(chunk.text, end="", flush=True)
                else:
                    print()
                    print()
                    print(f"Finish reason: {chunk.finish_reason}")
                    print(f"Latency: {chunk.latency_seconds:.2f}s")
                    if chunk.context_used:
                        print(f"Context used: {len(chunk.context_used)} chars")
        print()
        
        # Knowledge store stats
        print("=" * 60)
        print("KNOWLEDGE STORE STATISTICS")
        print("=" * 60)
        stats = knowledge_store.get_stats()
        print(f"Documents: {stats['documents']}")
        print(f"Retrieval chunks: {stats['retrieval_chunks']}")
        print(f"Context chunks: {stats['context_chunks']}")
        print()
    else:
        print()
        print("=" * 60)
        print("INFO: No document file provided")
        print("=" * 60)
        print("To test RAG, run: python sample.py <document_file>")
        print()
    
    # Pool stats
    print("=" * 60)
    print("MODEL POOL STATISTICS")
    print("=" * 60)
    pool_stats = pool.get_stats()
    print(f"Max models: {pool_stats['max_models']}")
    print(f"Loaded models: {pool_stats['loaded_models']}")
    for model_info in pool_stats['models']:
        print(f"  - {model_info['model_name']} (refs: {model_info['ref_count']})")
    print()
    
    # Usage summary
    print("=" * 60)
    print("USAGE SUMMARY")
    print("=" * 60)
    print("""
# Model Pool (async):
pool = ModelPool(max_models=2)
await pool.preload(config)

async with pool.acquire_context(config) as handle:
    runner = create_runner(handle, knowledge_store)
    result = await runner.generate(prompt="Hello")

# Async Streaming:
async for chunk in runner.generate_stream(prompt="Hello"):
    if not chunk.finished:
        print(chunk.text, end="")

# RAG:
runner.set_document_scope({doc_id})
runner.search("query")
result = await runner.generate(prompt="question")
""")

    # Cleanup
    await pool.unload_all()


if __name__ == "__main__":
    asyncio.run(main())