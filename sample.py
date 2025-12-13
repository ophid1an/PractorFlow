from llm import LLMConfig, create_runner


def main():
    config = LLMConfig()
    
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
    prompt = "What is Python in one sentence?"
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
        {"role": "user", "content": "What are the main programming paradigms?"},
    ]
    
    print(f"User: {messages[0]['content']}")
    result1 = runner.generate(messages=messages)
    print(f"Assistant: {result1['text']}")
    print()
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": result1['text']})
    
    # Continue conversation
    messages.append({"role": "user", "content": "Which one is best for beginners?"})
    print(f"User: {messages[-1]['content']}")
    result2 = runner.generate(messages=messages)
    print(f"Assistant: {result2['text']}")
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
    print(f"Assistant: {result3['text']}")
    print()


if __name__ == "__main__":
    main()