"""
Knowledge Store Integration Sample

Tests:
1. Loading a document into persistent KnowledgeStore
2. Loading a document into temporary storage (with TTL)
3. On-demand loading from KnowledgeStore into temp storage
4. Generation with RAG using both sources
"""

from llm import LLMConfig, create_runner
from llm.knowledge import ChromaKnowledgeStore, ChromaKnowledgeStoreConfig


def main():
    # Get user inputs
    print("=" * 60)
    print("KNOWLEDGE STORE INTEGRATION TEST")
    print("=" * 60)
    print()
    
    prompt = input("Enter your question/prompt: ").strip()
    if not prompt:
        print("Error: Prompt cannot be empty")
        return
    
    print()
    knowledge_file = input("Enter path to knowledge file (persistent storage): ").strip()
    if not knowledge_file:
        print("Error: Knowledge file path cannot be empty")
        return
    
    print()
    temp_file = input("Enter path to temp file (TTL storage): ").strip()
    if not temp_file:
        print("Error: Temp file path cannot be empty")
        return
    
    print()
    print("=" * 60)
    print("INITIALIZING")
    print("=" * 60)
    print()
    
    # Initialize knowledge store
    print("[1/4] Initializing knowledge store...")
    ks_config = ChromaKnowledgeStoreConfig(
        persist_directory="./test_knowledge_db",
        embedding_model_name="all-MiniLM-L6-v2"
    )
    knowledge_store = ChromaKnowledgeStore(ks_config)
    print()
    
    # Initialize LLM runner with knowledge store
    print("[2/4] Initializing LLM runner...")
    config = LLMConfig()
    runner = create_runner(config, knowledge_store=knowledge_store)
    print()
    
    # Load document into knowledge store (persistent)
    print("=" * 60)
    print("LOADING DOCUMENTS")
    print("=" * 60)
    print()
    
    print(f"[3/4] Loading into knowledge store: {knowledge_file}")
    try:
        knowledge_doc = knowledge_store.add_document_from_file(knowledge_file)
        print(f"  Document ID: {knowledge_doc['id']}")
        print(f"  Filename: {knowledge_doc['filename']}")
        print(f"  Retrieval chunks: {knowledge_doc['retrieval_chunk_count']}")
        print(f"  Context chunks: {knowledge_doc['context_chunk_count']}")
    except Exception as e:
        print(f"  Error loading knowledge file: {e}")
        return
    print()
    
    # Load document from knowledge store into temp storage
    print(f"  Loading from knowledge store into temp storage...")
    try:
        load_result = runner.load_document_from_knowledge(
            document_id=knowledge_doc['id'],
            ttl_hours=24
        )
        print(f"  Retrieval chunks loaded: {load_result['retrieval_chunks_loaded']}")
        print(f"  Context chunks loaded: {load_result['context_chunks_loaded']}")
        print(f"  TTL: {load_result['ttl_hours']} hours")
    except Exception as e:
        print(f"  Error loading from knowledge store: {e}")
        return
    print()
    
    # Load temp document directly (with TTL)
    print(f"[4/4] Loading into temp storage: {temp_file}")
    try:
        temp_doc = runner.load_document(temp_file, ttl_hours=1)
        print(f"  Document ID: {temp_doc['id']}")
        print(f"  Filename: {temp_doc['filename']}")
        retrieval_count = len(temp_doc.get('retrieval_chunks', []))
        context_count = len(temp_doc.get('context_chunks', []))
        print(f"  Retrieval chunks: {retrieval_count}")
        print(f"  Context chunks: {context_count}")
        print(f"  TTL: 1 hour")
    except Exception as e:
        print(f"  Error loading temp file: {e}")
        return
    print()
    
    # Show storage stats
    print("=" * 60)
    print("STORAGE STATS")
    print("=" * 60)
    print()
    
    print("Knowledge Store (persistent):")
    ks_stats = knowledge_store.get_stats()
    print(f"  Documents: {ks_stats['documents']}")
    print(f"  Retrieval chunks: {ks_stats['retrieval_chunks']}")
    print(f"  Context chunks: {ks_stats['context_chunks']}")
    print()
    
    print("Temp Vector Store:")
    print(f"  Retrieval chunks: {runner.vector_store.count()}")
    print(f"  Context chunks: {len(runner.context_store)}")
    print()
    
    # Generate response
    print("=" * 60)
    print("GENERATION")
    print("=" * 60)
    print()
    
    print(f"Prompt: {prompt}")
    print()
    print("Generating response...")
    print()
    
    try:
        result = runner.generate(
            prompt=prompt,
            use_context=True,
            max_context_docs=10
        )
        
        print("Response:")
        print("-" * 40)
        print(result['text'])
        print("-" * 40)
        print()
        
        print("Usage:")
        print(f"  Prompt tokens: {result['usage']['prompt_tokens']}")
        print(f"  Completion tokens: {result['usage']['completion_tokens']}")
        print(f"  Total tokens: {result['usage']['total_tokens']}")
        print(f"  Latency: {result['latency_seconds']:.2f}s")
        print()
        
        if 'context_sources' in result:
            print("Context sources used:")
            for src in result['context_sources']:
                print(f"  - {src['filename']} (similarity: {src['similarity']:.3f}, chars: {src['chars']})")
        print()
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
