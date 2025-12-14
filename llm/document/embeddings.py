"""
Embedding models for document vectorization.

Uses the loaded LLM model itself for embeddings.
"""

from typing import List, Union, Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for batches of text."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass


class LLMEmbeddingModel(BaseEmbeddingModel):
    """
    Extract embeddings from the LLM model itself.
    
    Works with:
    - llama-cpp models (uses embedding extraction)
    - transformers models (uses hidden states)
    """
    
    def __init__(self, llm_runner):
        """
        Initialize LLM-based embeddings.
        
        Args:
            llm_runner: LLMRunner instance (LlamaCppRunner or TransformersRunner)
        """
        self.runner = llm_runner
        self.backend = llm_runner.config.backend
        self._dimension = None
        
        print(f"[Embeddings] Using {self.backend} model for embeddings")
        
        # Initialize dimension
        self._initialize_dimension()
    
    def _initialize_dimension(self):
        """Determine embedding dimension from model."""
        if self.backend == "llama_cpp":
            # For llama-cpp, get dimension from model
            try:
                # Get embedding for a test string
                test_emb = self._embed_llama_cpp("test")
                self._dimension = len(test_emb)
                print(f"[Embeddings] Llama.cpp embedding dimension: {self._dimension}")
            except Exception as e:
                print(f"[Embeddings] Error determining dimension: {e}")
                raise
        
        elif self.backend == "transformers":
            # For transformers, use hidden size
            if hasattr(self.runner.model.config, 'hidden_size'):
                self._dimension = self.runner.model.config.hidden_size
                print(f"[Embeddings] Transformers embedding dimension: {self._dimension}")
            else:
                print("[Embeddings] Warning: Could not determine dimension, using 768")
                self._dimension = 768  # Default fallback
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Embeddings as numpy array. Shape: (dimension,) for single text,
            (n_texts, dimension) for list of texts
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Generate embeddings based on backend
        if self.backend == "llama_cpp":
            embeddings = self._embed_llama_cpp_batch(texts)
        elif self.backend == "transformers":
            embeddings = self._embed_transformers_batch(texts)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        # Return single vector for single input
        if is_single:
            return embeddings[0]
        
        return embeddings
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for large batches of text.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress (not implemented for LLM embeddings)
            
        Returns:
            Embeddings as numpy array of shape (n_texts, dimension)
        """
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.backend == "llama_cpp":
                batch_emb = self._embed_llama_cpp_batch(batch)
            elif self.backend == "transformers":
                batch_emb = self._embed_transformers_batch(batch)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            all_embeddings.append(batch_emb)
            
            if show_progress:
                print(f"[Embeddings] Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return np.vstack(all_embeddings)
    
    def _embed_llama_cpp(self, text: str) -> np.ndarray:
        """Extract embedding from llama-cpp model for single text."""
        # Use llama-cpp's embedding API
        result = self.runner.model.create_embedding(text)
        embedding_data = result['data'][0]['embedding']
        
        # Handle nested list structure from llama-cpp
        if isinstance(embedding_data[0], list):
            embedding_data = embedding_data[0]
        
        embedding = np.array(embedding_data, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _embed_llama_cpp_batch(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from llama-cpp model for batch."""
        embeddings = []
        
        for text in texts:
            emb = self._embed_llama_cpp(text)
            embeddings.append(emb)
        
        return np.vstack(embeddings)
    
    def _embed_transformers_batch(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from transformers model."""
        import torch
        
        # Tokenize
        inputs = self.runner.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # Reasonable limit for embeddings
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.runner.device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.runner.model(**inputs, output_hidden_states=True)
            
            # Use last hidden state
            last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            
            # Mean pooling over sequence length (excluding padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Mask out padding tokens
            masked_hidden = last_hidden_state * attention_mask
            
            # Sum and average
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
            sum_mask = attention_mask.sum(dim=1)  # (batch, 1)
            
            embeddings = sum_hidden / sum_mask  # (batch, hidden_dim)
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dimension