"""
Sample usage of LLM module with tool-based RAG.

Demonstrates:
- Basic generation
- Messages API with chat history
- Document management via KnowledgeStore
- Knowledge search tool with scoped search
- LLM generation with search context
"""

from llm import LLMConfig, create_runner, Session
from llm.knowledge import ChromaKnowledgeStore, ChromaKnowledgeStoreConfig
import base64
import io
import sys


def main():
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
    
    # Create runner with knowledge store
    runner = create_runner(config, knowledge_store=knowledge_store)
    print("Model loaded successfully!")
    print()
    
    # Example 1: Single prompt (backward compatibility)
    print("=" * 60)
    print("EXAMPLE 1: Single prompt")
    print("=" * 60)
    prompt = "What is Python in one sentence?"
    print(f"Prompt: {prompt}")
    print()
    
    result = runner.generate(prompt=prompt)
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
    
    print(f"User: {messages[0]['content']}")
    result1 = runner.generate(messages=messages)
    print(f"Assistant: {result1['reply']}")
    print()
    
    # Add assistant response to history (extract text from llama.cpp response)
    assistant_text = result1['reply']
    if isinstance(assistant_text, dict) and 'choices' in assistant_text:
        assistant_text = assistant_text['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": assistant_text})
    
    # Continue conversation
    messages.append({"role": "user", "content": "Which one is best for beginners?"})
    print(f"User: {messages[-1]['content']}")
    result2 = runner.generate(messages=messages)
    print(f"Assistant: {result2['reply']}")
    print()
    
    # Example 3: Messages with instructions
    print("=" * 60)
    print("EXAMPLE 3: Messages with system instructions")
    print("=" * 60)
    
    instructions = "You are a helpful Python tutor. Keep answers concise and beginner-friendly."
    messages = [
        {"role": "user", "content": "How do I define a function in Python?"},
    ]
    
    print(f"Instructions: {instructions}")
    print(f"User: {messages[0]['content']}")
    result3 = runner.generate(messages=messages, instructions=instructions)
    print(f"Assistant: {result3['reply']}")
    print()
    
    # Example 4: Document loading with RAG (if document provided)
    if document_file:
        print("=" * 60)
        print("EXAMPLE 4: Add document to knowledge store")
        print("=" * 60)
        
        # Add document to knowledge store
        print(f"Adding document: {document_file}")
        doc_info = knowledge_store.add_document_from_file(document_file)
        
        print(f"Document ID: {doc_info['id']}")
        print(f"Retrieval chunks: {doc_info['retrieval_chunk_count']}")
        print(f"Context chunks: {doc_info['context_chunk_count']}")
        print()
        
        # Set document scope for search
        runner.set_document_scope({doc_info['id']})
        
        # Search and ask question
        question = "What is this document about?"
        print(f"Question: {question}")
        
        # Search for relevant content
        search_result = runner.search(question)
        print(f"Search found {search_result.metadata.get('results_count', 0)} results")
        
        # Generate with search context
        result4 = runner.generate(prompt=question)
        
        print(f"Response<{result4['latency_seconds']:.2f}s>: {result4['reply']}")
        if 'search_metadata' in result4:
            print(f"\nDocuments used: {result4['search_metadata'].get('document_ids', [])}")
        print()
        
        # Example 5: Document loading from base64
        print("=" * 60)
        print("EXAMPLE 5: Document loading from base64")
        print("=" * 60)
        
        try:
            print(f"Converting {document_file} to base64...")
            with open(document_file, "rb") as f:
                file_bytes = f.read()
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
            
            print("Adding document from base64...")
            doc_info2 = knowledge_store.add_document_from_base64(
                base64_data=base64_data,
                filename=f"base64_{document_file}"
            )
            
            print(f"Document ID: {doc_info2['id']}")
            print(f"Knowledge store now has {knowledge_store.count_chunks()} chunks")
            print()
            
            # Ask another question with both documents in scope
            runner.set_document_scope({doc_info['id'], doc_info2['id']})
            
            question2 = "Summarize the key points from this document."
            print(f"Question: {question2}")
            
            runner.search(question2)
            result5 = runner.generate(prompt=question2)
            
            print(f"Response<{result5['latency_seconds']:.2f}s>: {result5['reply']}")
            if 'search_metadata' in result5:
                print(f"\nResults count: {result5['search_metadata'].get('results_count', 0)}")
            print()
        except Exception as e:
            print(f"Error loading from base64: {e}")
            print()
        
        # Example 6: Document loading from data URI
        print("=" * 60)
        print("EXAMPLE 6: Document loading from data URI")
        print("=" * 60)
        
        try:
            print(f"Converting {document_file} to data URI...")
            with open(document_file, "rb") as f:
                file_bytes = f.read()
                base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
                data_uri = f"data:application/octet-stream;base64,{base64_encoded}"
            
            print("Adding document from data URI...")
            doc_info3 = knowledge_store.add_document_from_base64(
                base64_data=data_uri,
                filename=f"datauri_{document_file}"
            )
            
            print(f"Document ID: {doc_info3['id']}")
            print(f"Knowledge store now has {knowledge_store.count_chunks()} chunks")
            print()
        except Exception as e:
            print(f"Error loading from data URI: {e}")
            print()
        
        # Example 7: Document loading from stream (simulated FastAPI upload)
        print("=" * 60)
        print("EXAMPLE 7: Document loading from stream (simulated)")
        print("=" * 60)
        
        try:
            print(f"Simulating stream upload for {document_file}...")
            
            with open(document_file, "rb") as f:
                file_bytes = f.read()
                file_stream = io.BytesIO(file_bytes)
            
            print("Adding document from stream...")
            doc_info4 = knowledge_store.add_document_from_stream(
                file_stream=file_stream,
                filename=f"stream_{document_file}",
                mime_type="application/pdf"
            )
            
            print(f"Document ID: {doc_info4['id']}")
            print(f"Knowledge store now has {knowledge_store.count_chunks()} chunks")
            print()
            
            # Query with all documents in scope
            all_doc_ids = {doc['id'] for doc in knowledge_store.list_documents()}
            runner.set_document_scope(all_doc_ids)
            
            question3 = "What are the main topics covered?"
            print(f"Question: {question3}")
            
            runner.search(question3)
            result6 = runner.generate(prompt=question3)
            
            print(f"Response<{result6['latency_seconds']:.2f}s>: {result6['reply']}")
            if 'search_metadata' in result6:
                print(f"\nDocuments referenced: {result6['search_metadata'].get('document_ids', [])}")
            print()
        except Exception as e:
            print(f"Error loading from stream: {e}")
            print()
        
        # Clear document scope
        runner.clear_document_scope()
        
        # Show knowledge store statistics
        print("=" * 60)
        print("KNOWLEDGE STORE STATISTICS")
        print("=" * 60)
        stats = knowledge_store.get_stats()
        print(f"Documents: {stats['documents']}")
        print(f"Retrieval chunks: {stats['retrieval_chunks']}")
        print(f"Context chunks: {stats['context_chunks']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print()
        
        # List all documents
        print("Documents in store:")
        for doc in knowledge_store.list_documents():
            print(f"  - {doc['filename']} (ID: {doc['id']})")
        print()
        
    else:
        print()
        print("=" * 60)
        print("INFO: No document file provided")
        print("=" * 60)
        print("To test RAG with document loading, run:")
        print("  python sample.py file.docx")
        print("  python sample.py README.md")
        print("  python sample.py document.pdf")
        print()
    
    # Usage summary
    print("=" * 60)
    print("USAGE SUMMARY")
    print("=" * 60)
    print("""
# Document Management (via KnowledgeStore):
knowledge_store.add_document_from_file("path/to/file.pdf")
knowledge_store.add_document_from_bytes(data, "file.pdf")
knowledge_store.add_document_from_base64(base64_data, "file.pdf")
knowledge_store.add_document_from_stream(stream, "file.pdf")  # FastAPI UploadFile
knowledge_store.list_documents()
knowledge_store.delete_document(doc_id)

# LLM with RAG (via Runner):
runner = create_runner(config, knowledge_store=knowledge_store)
runner.set_document_scope({doc_id1, doc_id2})  # Scope search to specific docs
runner.search("your query")                     # Search knowledge base
runner.generate(prompt="your question")         # Generate with search context
runner.clear_document_scope()                   # Search all documents
""")


if __name__ == "__main__":
    main()