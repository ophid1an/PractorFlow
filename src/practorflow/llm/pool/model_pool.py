"""
ModelPool - Async model pool with LRU eviction.

Manages model lifecycle for API server usage:
- Caches loaded models by configuration
- Reference counting for safe concurrent access
- LRU eviction when max models reached
- Async context manager for clean acquire/release
"""

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional, AsyncIterator

import torch
from transformers import GenerationConfig

from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.pool.model_handle import ModelHandle

from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger(
    "model_pool", level=appConfiguration.LoggerConfiguration.ModelPoolLevel
)
verbose_runner = appConfiguration.LoggerConfiguration.RunnerLevel == "DEBUG"


class ModelPool:
    """
    Async model pool with configurable max concurrent models.

    Usage:
        pool = ModelPool(max_models=2)

        async with pool.acquire(config) as handle:
            runner = create_runner(handle, knowledge_store)
            result = runner.generate(prompt="Hello")
        # Auto-release on exit
    """

    _instance: Optional["ModelPool"] = None
    _lock: asyncio.Lock = None

    def __init__(self, max_models: int = 1):
        """
        Initialize model pool.

        Args:
            max_models: Maximum number of models to keep loaded (default: 1)
        """
        self.max_models = max_models
        self._models: Dict[str, ModelHandle] = {}
        self._lock = asyncio.Lock()
        self._loading_locks: Dict[str, asyncio.Lock] = {}

        logger.info(f"[ModelPool] Initialized with max_models={max_models}")

    @classmethod
    def get_instance(cls, max_models: int = 1) -> "ModelPool":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(max_models=max_models)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance._models.clear()
            cls._instance = None

    def _compute_config_hash(self, config: LLMConfig) -> str:
        """Compute hash from config for cache key."""
        key_fields = {
            "model_name": config.model_name,
            "backend": config.backend,
            "device": config.device,
            "dtype": str(config.dtype),
            "quantization": config.quantization,
            "n_gpu_layers": config.n_gpu_layers,
            "n_ctx": config.n_ctx,
            "n_batch": config.n_batch,
            "use_torch_compile": getattr(config, "use_torch_compile", False),
            "compile_mode": getattr(config, "compile_mode", "default"),
        }
        key_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def _get_loading_lock(self, config_hash: str) -> asyncio.Lock:
        """Get or create a loading lock for a specific config."""
        async with self._lock:
            if config_hash not in self._loading_locks:
                self._loading_locks[config_hash] = asyncio.Lock()
            return self._loading_locks[config_hash]

    async def _evict_lru(self) -> bool:
        """
        Evict least recently used model that is not in use.

        Returns:
            True if a model was evicted, False otherwise
        """
        lru_hash = None
        lru_time = None

        for config_hash, handle in self._models.items():
            if handle.is_in_use:
                continue
            if lru_time is None or handle.last_used_at < lru_time:
                lru_time = handle.last_used_at
                lru_hash = config_hash

        if lru_hash is None:
            return False

        handle = self._models.pop(lru_hash)
        logger.info(
            f"[ModelPool] Evicted model: {handle.config.model_name} (hash: {lru_hash[:8]}...)"
        )

        del handle.model
        if handle.tokenizer:
            del handle.tokenizer

        return True

    def _load_llama_cpp_model(self, config: LLMConfig) -> ModelHandle:
        """Load a llama.cpp model."""
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        os.makedirs(config.models_dir, exist_ok=True)

        model_path = None
        filename = os.path.basename(config.model_name)
        for dirpath, _, filenames in os.walk(config.models_dir):
            if filename in filenames:
                model_path = os.path.join(dirpath, filename)
                break

        if not model_path:
            if config.model_name.count("/") < 1:
                raise ValueError(
                    "[ModelPool] model_name must be 'repo_id/filename.gguf'"
                )

            repo_id, filename = config.model_name.rsplit("/", 1)
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=config.models_dir,
                token=os.environ.get("HF_TOKEN"),
            )

        if not os.path.isfile(model_path):
            raise ValueError(f"[ModelPool] Model path is not a file: {model_path}")

        with open(model_path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("[ModelPool] File is not a valid GGUF file")

        logger.info(f"[ModelPool] Loading llama.cpp model: {model_path}")

        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "embedding": False,
            "verbose": verbose_runner,
        }

        if config.n_batch is not None:
            llama_kwargs["n_batch"] = config.n_batch

        model = Llama(**llama_kwargs)
        model.set_cache(None)

        max_ctx = model.n_ctx() if hasattr(model, "n_ctx") else config.n_ctx

        config_hash = self._compute_config_hash(config)

        return ModelHandle(
            config=config,
            backend="llama_cpp",
            model=model,
            tokenizer=None,
            max_context_length=max_ctx,
            config_hash=config_hash,
        )

    def _compile_model(self, model, config: LLMConfig):
        """
        Apply torch.compile() to the model for faster inference.
        
        Args:
            model: The transformers model to compile
            config: LLM configuration
            
        Returns:
            Compiled model or original if compilation fails
        """
        use_compile = getattr(config, "use_torch_compile", True)
        compile_mode = getattr(config, "compile_mode", "reduce-overhead")
        
        if not use_compile:
            logger.info("[ModelPool] torch.compile() disabled by config")
            return model
        
        torch_version = tuple(map(int, torch.__version__.split("+")[0].split(".")[:2]))
        if torch_version < (2, 0):
            logger.warning(
                f"[ModelPool] torch.compile() requires PyTorch 2.0+, "
                f"found {torch.__version__}"
            )
            return model
        
        if not torch.cuda.is_available():
            logger.info("[ModelPool] CUDA not available, skipping torch.compile()")
            return model
        
        try:
            logger.info(f"[ModelPool] Compiling model with mode='{compile_mode}'...")
            
            compiled_model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=False,
                dynamic=True,
            )
            
            logger.info("[ModelPool] Model compiled successfully")
            return compiled_model
            
        except Exception as e:
            logger.warning(f"[ModelPool] torch.compile() failed: {e}")
            logger.warning("[ModelPool] Falling back to eager mode")
            return model

    def _warmup_model(self, model, tokenizer, device, config: LLMConfig) -> None:
        """
        Run warmup inference to trigger JIT compilation.
        
        This ensures the first real request doesn't pay the compilation cost.
        
        Args:
            model: The loaded model
            tokenizer: The tokenizer
            device: Device the model is on
            config: LLM configuration for generation parameters
        """
        logger.info("[ModelPool] Running warmup inference...")
        
        try:
            warmup_text = "Hello, how are you?"
            
            inputs = tokenizer(
                warmup_text,
                return_tensors="pt",
                padding=True,
            ).to(device)
            
            do_sample = config.temperature > 0
            
            gen_config_kwargs = {
                "max_new_tokens": min(config.max_new_tokens, 10),  # Limit warmup tokens
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            if do_sample:
                gen_config_kwargs["temperature"] = config.temperature
                gen_config_kwargs["top_p"] = config.top_p
            
            warmup_gen_config = GenerationConfig(**gen_config_kwargs)
            
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    generation_config=warmup_gen_config,
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            logger.info("[ModelPool] Warmup complete")
            
        except Exception as e:
            logger.warning(f"[ModelPool] Warmup failed (non-fatal): {e}")

    def _load_transformers_model(self, config: LLMConfig) -> ModelHandle:
        """Load a transformers model with optimizations."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.makedirs(config.models_dir, exist_ok=True)

        logger.info(f"[ModelPool] Loading transformers model: {config.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.models_dir,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "cache_dir": config.models_dir,
            "trust_remote_code": True,
        }

        if config.dtype == "auto" or config.dtype is None:
            model_kwargs["dtype"] = "auto"
        else:
            model_kwargs["dtype"] = config.dtype

        if config.device == "auto":
            model_kwargs["device_map"] = "auto"
        elif config.device != "cpu":
            model_kwargs["device_map"] = config.device

        if config.quantization:
            try:
                from transformers import BitsAndBytesConfig

                if config.quantization == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                elif config.quantization == "8bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                logger.info(f"[ModelPool] Using {config.quantization} quantization")
            except ImportError:
                logger.warning(
                    "[ModelPool] bitsandbytes not available, skipping quantization"
                )

        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

        if "device_map" not in model_kwargs and config.device == "cpu":
            model = model.to(config.device)

        model.eval()

        if hasattr(model, "device"):
            device = model.device
        else:
            device = next(model.parameters()).device

        model = self._compile_model(model, config)

        warmup_on_load = getattr(config, "warmup_on_load", True)
        if warmup_on_load and getattr(config, "use_torch_compile", True):
            self._warmup_model(model, tokenizer, device, config)

        if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
            max_ctx = model.config.max_position_embeddings
        else:
            max_ctx = config.n_ctx

        config_hash = self._compute_config_hash(config)

        return ModelHandle(
            config=config,
            backend="transformers",
            model=model,
            tokenizer=tokenizer,
            max_context_length=max_ctx,
            config_hash=config_hash,
        )

    async def _load_model(self, config: LLMConfig) -> ModelHandle:
        """Load model in thread pool to avoid blocking."""
        loop = asyncio.get_running_loop()

        if config.backend == "llama_cpp":
            handle = await loop.run_in_executor(
                None, self._load_llama_cpp_model, config
            )
        elif config.backend == "transformers":
            handle = await loop.run_in_executor(
                None, self._load_transformers_model, config
            )
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

        return handle

    async def acquire(self, config: LLMConfig) -> ModelHandle:
        """
        Acquire a model handle.

        Returns cached model if available, otherwise loads new model.
        May evict LRU model if max_models reached.

        Args:
            config: Model configuration

        Returns:
            ModelHandle for the requested model
        """
        config_hash = self._compute_config_hash(config)

        async with self._lock:
            if config_hash in self._models:
                handle = self._models[config_hash]
                handle.acquire()
                logger.info(
                    f"[ModelPool] Cache hit: {config.model_name} (refs: {handle.ref_count})"
                )
                return handle

        loading_lock = await self._get_loading_lock(config_hash)

        async with loading_lock:
            async with self._lock:
                if config_hash in self._models:
                    handle = self._models[config_hash]
                    handle.acquire()
                    return handle

                while len(self._models) >= self.max_models:
                    evicted = await self._evict_lru()
                    if not evicted:
                        raise RuntimeError(
                            f"[ModelPool] Cannot load model: max_models={self.max_models} "
                            f"reached and all models are in use"
                        )

            logger.info(f"[ModelPool] Loading new model: {config.model_name}")
            handle = await self._load_model(config)

            async with self._lock:
                self._models[config_hash] = handle
                handle.acquire()
                logger.info(
                    f"[ModelPool] Model loaded and cached: {config.model_name} (refs: {handle.ref_count})"
                )

            return handle

    async def release(self, handle: ModelHandle) -> None:
        """
        Release a model handle.

        Decrements reference count. Model stays cached for reuse.

        Args:
            handle: ModelHandle to release
        """
        async with self._lock:
            handle.release()
            logger.info(
                f"[ModelPool] Released: {handle.config.model_name} (refs: {handle.ref_count})"
            )

    @asynccontextmanager
    async def acquire_context(self, config: LLMConfig) -> AsyncIterator[ModelHandle]:
        """
        Async context manager for acquiring/releasing models.

        Usage:
            async with pool.acquire_context(config) as handle:
                runner = create_runner(handle, knowledge_store)
                result = runner.generate(prompt="Hello")
        """
        handle = await self.acquire(config)
        try:
            yield handle
        finally:
            await self.release(handle)

    async def preload(self, config: LLMConfig) -> None:
        """
        Preload a model for faster first request.

        Useful for warming up at server startup.

        Args:
            config: Model configuration to preload
        """
        handle = await self.acquire(config)
        await self.release(handle)
        logger.info(f"[ModelPool] Preloaded: {config.model_name}")

    async def unload(self, config: LLMConfig) -> bool:
        """
        Explicitly unload a model.

        Only succeeds if model is not in use.

        Args:
            config: Model configuration to unload

        Returns:
            True if unloaded, False if not found or in use
        """
        config_hash = self._compute_config_hash(config)

        async with self._lock:
            if config_hash not in self._models:
                return False

            handle = self._models[config_hash]
            if handle.is_in_use:
                logger.info(
                    f"[ModelPool] Cannot unload {config.model_name}: still in use"
                )
                return False

            del self._models[config_hash]
            del handle.model
            if handle.tokenizer:
                del handle.tokenizer

            logger.info(f"[ModelPool] Unloaded: {config.model_name}")
            return True

    async def unload_all(self, force: bool = False) -> int:
        """
        Unload all models.

        Args:
            force: If True, unload even models in use

        Returns:
            Number of models unloaded
        """
        async with self._lock:
            count = 0
            hashes_to_remove = []

            for config_hash, handle in self._models.items():
                if handle.is_in_use and not force:
                    continue
                hashes_to_remove.append(config_hash)

            for config_hash in hashes_to_remove:
                handle = self._models.pop(config_hash)
                del handle.model
                if handle.tokenizer:
                    del handle.tokenizer
                count += 1

            logger.info(f"[ModelPool] Unloaded {count} models")
            return count

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            "max_models": self.max_models,
            "loaded_models": len(self._models),
            "models": [
                {
                    "model_name": h.config.model_name,
                    "backend": h.backend,
                    "ref_count": h.ref_count,
                    "config_hash": h.config_hash[:8],
                    "created_at": h.created_at.isoformat(),
                    "last_used_at": h.last_used_at.isoformat(),
                }
                for h in self._models.values()
            ],
        }

    def __repr__(self) -> str:
        return f"ModelPool(max_models={self.max_models}, loaded={len(self._models)})"