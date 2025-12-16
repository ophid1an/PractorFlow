from llm import LLMConfig, create_runner, Session
import base64
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
    
    runner = create_runner(config)
    print("Model loaded successfully!")
    print()
    
    # # Example 1: Single prompt (backward compatibility)
    # print("=" * 60)
    # print("EXAMPLE 1: Single prompt")
    # print("=" * 60)
    # prompt = "What is Python in one sentence?"
    # print(f"Prompt: {prompt}")
    # print()
    
    # result = runner.generate(prompt=prompt)
    # print(f"Response: {result['text']}")
    # print(f"Tokens: {result['usage']['total_tokens']}")
    # print(f"Latency: {result['latency_seconds']:.2f}s")
    # print()
    
    # # Example 2: Messages API (chat history)
    # print("=" * 60)
    # print("EXAMPLE 2: Messages API with chat history")
    # print("=" * 60)
    
    # messages = [
    #     {"role": "user", "content": "What are the main programming paradigms?"},
    # ]
    
    # print(f"User: {messages[0]['content']}")
    # result1 = runner.generate(messages=messages)
    # print(f"Assistant: {result1['text']}")
    # print()
    
    # # Add assistant response to history
    # messages.append({"role": "assistant", "content": result1['text']})
    
    # # Continue conversation
    # messages.append({"role": "user", "content": "Which one is best for beginners?"})
    # print(f"User: {messages[-1]['content']}")
    # result2 = runner.generate(messages=messages)
    # print(f"Assistant: {result2['text']}")
    # print()
    
    # # Example 3: Messages with instructions
    # print("=" * 60)
    # print("EXAMPLE 3: Messages with system instructions")
    # print("=" * 60)
    
    # instructions = "You are a helpful Python tutor. Keep answers concise and beginner-friendly."
    # messages = [
    #     {"role": "user", "content": "How do I define a function in Python?"},
    # ]
    
    # print(f"Instructions: {instructions}")
    # print(f"User: {messages[0]['content']}")
    # result3 = runner.generate(messages=messages, instructions=instructions)
    # print(f"Assistant: {result3['text']}")
    # print()
    
    # Example 4: Document loading with RAG (if document provided)
    if document_file:
        print("=" * 60)
        print("EXAMPLE 4: Document loading with RAG")
        print("=" * 60)
        
        # Create session
        session = Session(session_id="example_session")
        
        # Create runner with session
        runner_with_session = create_runner(config, session=session)
        
        # Load document - now automatically chunks and embeds
        print(f"Loading document: {document_file}")
        runner_with_session.load_document(document_file)
        
        print(f"Vector store has {runner_with_session.vector_store.count()} chunks")
        print()
        
        # Ask question - will use semantic search on vector store
        question = "What is this document about?"
        print(f"Question: {question}")
        result4 = runner_with_session.generate(
            prompt=question
        )
        
        print(f"Response: {result4['reply']}")
        if 'context_sources' in result4:
            print(f"\nSources used:")
            for src in result4['context_sources']:
                print(f"  - {src['filename']} (similarity: {src['similarity']:.3f}, chars: {src['chars']})")
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
            
            print("Loading document from base64...")
            runner_with_session.load_document_from_base64(
                base64_data=base64_data,
                filename=document_file
            )
            
            print(f"Vector store now has {runner_with_session.vector_store.count()} chunks")
            print()
            
            # Ask another question
            question2 = "Summarize the key points from this document."
            print(f"Question: {question2}")
            result5 = runner_with_session.generate(
                prompt=question2
            )
            
            print(f"Response: {result5['reply']}")
            if 'context_sources' in result5:
                print(f"\nRetrieved {len(result5['context_sources'])} relevant chunks")
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
            
            print("Loading document from data URI...")
            runner_with_session.load_document_from_base64(
                base64_data=data_uri,
                filename=document_file
            )
            
            print(f"Vector store now has {runner_with_session.vector_store.count()} chunks")
            print()
        except Exception as e:
            print(f"Error loading from data URI: {e}")
            print()
        
        # Example 7: Document loading from stream (simulated)
        print("=" * 60)
        print("EXAMPLE 7: Document loading from stream (simulated)")
        print("=" * 60)
        
        try:
            print(f"Simulating stream upload for {document_file}...")
            
            import io
            with open(document_file, "rb") as f:
                file_bytes = f.read()
                file_stream = io.BytesIO(file_bytes)
            
            print("Loading document from stream...")
            runner_with_session.load_document_from_stream(
                file_stream=file_stream,
                filename=document_file,
                mime_type="application/pdf"
            )
            
            print(f"Vector store now has {runner_with_session.vector_store.count()} chunks")
            print()
            
            # Query the document
            question3 = "What are the main topics covered?"
            print(f"Question: {question3}")
            result6 = runner_with_session.generate(
                prompt=question3
            )
            
            print(f"Response: {result6['reply']}")
            if 'context_sources' in result6:
                print(f"\nTop relevant chunks:")
                for src in result6['context_sources'][:3]:
                    print(f"  - {src['filename']} (page {src.get('page', '?')}, similarity: {src['similarity']:.3f})")
            print()
        except Exception as e:
            print(f"Error loading from stream: {e}")
            print()
        
        # Show vector store statistics
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
        print("To test RAG with document loading, run:")
        print("  python sample.py file.docx")
        print("  python sample.py README.md")
        print("  python sample.py document.pdf")
        print()


if __name__ == "__main__":
    main()