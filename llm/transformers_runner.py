import time
import os
from typing import Dict, Any, Optional, List

import torch
from transformers import AutoModel, AutoTokenizer

from llm.llm_config import LLMConfig
from llm.base.llm_runner import LLMRunner
from llm.base.session import Session


class TransformersRunner(LLMRunner):
    """Runner for models using HuggingFace Transformers with truly unlimited document context support."""

    def __init__(self, config: LLMConfig, session: Optional[Session] = None):
        """
        Initialize Transformers runner with optional session.
        
        Args:
            config: LLM configuration
            session: Optional Session object for context and document management
        """
        super().__init__(config, session)

        os.makedirs(config.models_dir, exist_ok=True)

        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=config.models_dir,
        )
        
        print(f"Loading model {self.model_name}...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            cache_dir=config.models_dir,
            trust_remote_code=True,
        )
        self.model.eval()

        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_context_length = self.model.config.max_position_embeddings
            print(f"[transformers] Model context window: {self.max_context_length}")
        else:
            self.max_context_length = None
            print("[transformers] Model context window: unlimited")

        print("Model loaded successfully!")

    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
        use_context: bool = True,
        max_context_docs: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text from messages or prompt with unlimited document context.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            prompt: Single prompt string (for backward compatibility)
            instructions: System-level instructions/prompt
            temperature: Sampling temperature (uses config default if None)
            top_p: Nucleus sampling parameter (uses config default if None)
            use_context: Whether to include document context (default: True)
            max_context_docs: Maximum documents to include (default: None = unlimited)
            max_context_chars: Maximum characters for context (default: None = unlimited)

        Returns:
            Dictionary with keys: text, usage, latency_seconds, context_used (if context was added)
        """
        start_time = time.time()

        # Retrieve relevant context if enabled
        context = None
        context_sources = []
        
        if use_context and self.context_enabled:
            query = self._extract_query(messages, prompt)
            
            if query:
                print(f"[Context] Searching {len(self._active_documents)} documents...")
                context, context_sources = self._retrieve_context(
                    query, 
                    max_docs=max_context_docs,
                    max_chars=max_context_chars
                )
                
                if context:
                    print(f"[Context] Retrieved {len(context_sources)} documents ({len(context)} chars)")

        # Build final prompt (with context if retrieved)
        final_prompt = self._build_prompt(messages, prompt, instructions, context)

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        input_ids = self.tokenizer.encode(final_prompt, return_tensors="pt")
        input_length = input_ids.shape[1]

        input_ids = input_ids.to(self.device)

        # Build generation kwargs
        gen_kwargs = {
            "do_sample": True,
            "temperature": temp,
            "top_p": tp,
            "use_cache": True,
        }
        
        # Only add max_new_tokens if explicitly set
        if self.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = self.max_new_tokens
        
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        generated_ids = output_ids[:, input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        completion_tokens = generated_ids.shape[1]
        total_tokens = input_length + completion_tokens

        latency = time.time() - start_time

        response = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "latency_seconds": latency,
        }
        
        # Add context information if used
        if context:
            response["context_used"] = context
            response["context_sources"] = context_sources
        
        return response

    def _extract_query(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Extract the user query for context retrieval."""
        if messages:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    return msg["content"]
        elif prompt:
            return prompt
        return None

    def _retrieve_context(
        self,
        query: str,
        max_docs: Optional[int] = None,
        max_chars: Optional[int] = None
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """
        Retrieve relevant context from documents using simple keyword matching.
        UNLIMITED by default - includes all matching documents in full.
        
        Args:
            query: User query
            max_docs: Maximum number of documents to include (None = unlimited)
            max_chars: Maximum total characters (None = unlimited)
            
        Returns:
            Tuple of (context_string, list_of_source_dicts)
        """
        documents = self._active_documents
        
        if not documents:
            return None, []
        
        # Simple keyword-based scoring
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            content_lower = doc["content"].lower()
            content_words = set(content_lower.split())
            
            # Score based on matching words
            matches = len(query_words.intersection(content_words))
            
            # Boost score if query appears as phrase
            if query.lower() in content_lower:
                matches += 10
            
            if matches > 0:
                scored_docs.append((matches, doc))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Apply max_docs limit if specified
        if max_docs is not None:
            scored_docs = scored_docs[:max_docs]
        
        # Build context from documents
        context_parts = []
        sources = []
        total_chars = 0
        
        for score, doc in scored_docs:
            content = doc["content"]
            
            # Apply max_chars limit if specified
            if max_chars is not None and total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 0:
                    content = content[:remaining]
                else:
                    break
            
            context_parts.append(f"[{doc['filename']}]\n{content}")
            sources.append({
                "id": doc["id"],
                "filename": doc["filename"],
                "score": score
            })
            total_chars += len(content)
            
            # Stop if max_chars limit reached
            if max_chars is not None and total_chars >= max_chars:
                break
        
        if not context_parts:
            return None, []
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _build_prompt(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Build final prompt from messages/prompt, instructions, and context."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")
        
        # Build from messages
        if messages is not None:
            return self._format_messages(messages, instructions, context)
        
        # Build from single prompt
        parts = []
        
        if instructions:
            parts.append(instructions)
        
        if context:
            parts.append(f"\n---DOCUMENT CONTEXT---\n{context}\n---END CONTEXT---\n")
        
        parts.append(prompt)
        
        return "\n\n".join(parts)

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Format messages into a prompt string with optional context."""
        parts = []
        
        if instructions:
            parts.append(f"Instructions: {instructions}\n")
        
        if context:
            parts.append(f"---DOCUMENT CONTEXT---")
            parts.append(context)
            parts.append(f"---END CONTEXT---\n")
            parts.append("Use the above context to answer the user's question when relevant.\n")
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        parts.append("Assistant:")
        return "\n".join(parts)