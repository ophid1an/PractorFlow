"""
Embedding models for document vectorization.

Supports:
- SentenceTransformer models (recommended)
- LLM-based embeddings (llama-cpp or transformers)
"""

from typing import List, Union, Any, Optional
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


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Embedding model using sentence-transformers.
    
    Recommended models:
    - all-MiniLM-L6-v2: Fast, 384 dims, good quality
    - all-mpnet-base-v2: Slower, 768 dims, better quality
    - bge-small-en-v1.5: Fast, 384 dims, excellent quality
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize SentenceTransformer embedding model.
        
        Args:
            model_name: Model name from HuggingFace
            device: Device to use (None for auto-detect)
            cache_dir: Cache directory for model files
        """
        from sentence_transformers import SentenceTransformer
        
        print(f"[Embeddings] Loading SentenceTransformer: {model_name}")
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir
        )
        self.model_name = model_name
        self._dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"[Embeddings] Loaded. Dimension: {self._dimension}")
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Embeddings as numpy array
        """
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for batches of text.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings as numpy array of shape (n_texts, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dimension


class LLMEmbeddingModel(BaseEmbeddingModel):
    """
    Extract embeddings from the LLM model itself.
    
    Works with:
    - llama-cpp models (uses embedding extraction)
    - transformers models (uses hidden states)
    """
    
    def __init__(self, llm_model: Any, backend: str, tokenizer: Any = None, device: str = None):
        """
        Initialize LLM-based embeddings.
        
        Args:
            llm_model: The LLM model instance (Llama or transformers model)
            backend: Backend type ("llama_cpp" or "transformers")
            tokenizer: Tokenizer (required for transformers backend)
            device: Device for transformers backend
        """
        self.model = llm_model
        self.backend = backend
        self.tokenizer = tokenizer
        self.device = device
        self._dimension = None
        
        print(f"[Embeddings] Using {self.backend} model for embeddings")
        
        self._initialize_dimension()
    
    def _initialize_dimension(self):
        """Determine embedding dimension from model."""
        if self.backend == "llama_cpp":
            try:
                test_emb = self._embed_llama_cpp("test")
                self._dimension = len(test_emb)
                print(f"[Embeddings] Llama.cpp embedding dimension: {self._dimension}")
            except Exception as e:
                print(f"[Embeddings] Error determining dimension: {e}")
                raise
        
        elif self.backend == "transformers":
            if hasattr(self.model.config, 'hidden_size'):
                self._dimension = self.model.config.hidden_size
                print(f"[Embeddings] Transformers embedding dimension: {self._dimension}")
            else:
                print("[Embeddings] Warning: Could not determine dimension, using 768")
                self._dimension = 768
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        if self.backend == "llama_cpp":
            embeddings = self._embed_llama_cpp_batch(texts)
        elif self.backend == "transformers":
            embeddings = self._embed_transformers_batch(texts)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        if is_single:
            return embeddings[0]
        
        return embeddings
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings for batches of text."""
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
        result = self.model.create_embedding(text)
        embedding_data = result['data'][0]['embedding']
        
        if isinstance(embedding_data[0], list):
            embedding_data = embedding_data[0]
        
        embedding = np.array(embedding_data, dtype=np.float32)
        
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
        
        if not self.tokenizer:
            raise ValueError("Tokenizer required for transformers backend")
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            
            masked_hidden = last_hidden_state * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            
            embeddings = sum_hidden / sum_mask
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dimension