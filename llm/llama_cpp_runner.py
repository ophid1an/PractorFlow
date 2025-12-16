import time
import os
from typing import Dict, Any, Optional, List

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from llm.llm_config import LLMConfig
from llm.base.llm_runner import LLMRunner
from llm.base.session import Session
from llm.knowledge.knowledge_store import KnowledgeStore
from llm.tools.knowledge_search import KnowledgeSearchTool


class LlamaCppRunner(LLMRunner):
    """Runner for GGUF models using llama-cpp-python with tool-based RAG."""

    def __init__(
        self,
        config: LLMConfig,
        session: Optional[Session] = None,
        knowledge_store: Optional[KnowledgeStore] = None
    ):
        super().__init__(config, session, knowledge_store)
        
        os.makedirs(config.models_dir, exist_ok=True)
        model_path = self._resolve_model_path()

        print(f"[LlamaCppRunner] Loading model: {model_path}")

        if not os.path.isfile(model_path):
            raise ValueError(f"[LlamaCppRunner] Model path is not a file: {model_path}")

        with open(model_path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("[LlamaCppRunner] File is not a valid GGUF file")

        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "embedding": False,
            "verbose": True,
        }
        
        if config.n_batch is not None:
            llama_kwargs["n_batch"] = config.n_batch

        self.model = Llama(**llama_kwargs)
        self.model.set_cache(None)
        
        if hasattr(self.model, 'n_ctx'):
            self.max_context_length = self.model.n_ctx()
        else:
            self.max_context_length = config.n_ctx

        print(f"[LlamaCppRunner] Model loaded. Context window: {self.max_context_length}")
        
        # Register knowledge search tool if knowledge store is available
        if self.knowledge_store:
            knowledge_tool = KnowledgeSearchTool(
                knowledge_store=self.knowledge_store,
                default_top_k=config.max_search_results
            )
            self.tool_registry.register(knowledge_tool)
            print("[LlamaCppRunner] Knowledge search tool registered")

    def _resolve_model_path(self) -> str:
        """Resolve model path from config or download from HuggingFace."""
        if self.config.local_model_path:
            if not os.path.exists(self.config.local_model_path):
                raise ValueError(f"[LlamaCppRunner] Path does not exist: {self.config.local_model_path}")
            return self.config.local_model_path

        filename = os.path.basename(self.model_name)
        found = self._find_file_recursive(self.config.models_dir, filename)
        if found:
            self.config.local_model_path = found
            return found

        if self.model_name.count("/") < 1:
            raise ValueError("[LlamaCppRunner] model_name must be 'repo_id/filename.gguf'")

        repo_id, filename = self.model_name.rsplit("/", 1)
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=self.config.models_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        self.config.local_model_path = model_path
        return model_path

    def _find_file_recursive(self, root_dir: str, filename: str) -> Optional[str]:
        """Find a file recursively in directory tree."""
        for dirpath, _, filenames in os.walk(root_dir):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def get_chat_reply_structure(self) -> Optional[str]:
        """Get the chat template from model metadata."""
        try:
            metadata = self.model.metadata
            if metadata and "tokenizer.chat_template" in metadata:
                return metadata["tokenizer.chat_template"]
        except Exception:
            pass
        return None

    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
    ) -> Dict[str, Any]:
        """Generate with optional context from prior search() call."""
        start_time = time.time()

        # Get pending context from search tool
        context = self._consume_pending_context()
        context_metadata = None
        
        if context:
            # Get metadata from last tool result
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            
            print(f"[LlamaCppRunner] Using search context ({len(context)} chars)")

        # Build chat messages
        chat_messages = self._build_chat_messages(messages, prompt, instructions, context)

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        print("[LlamaCppRunner] Generating...")
        
        completion_kwargs = {
            "messages": chat_messages,
            "temperature": temp,
            "top_p": tp,
            "max_tokens": self.max_new_tokens,
        }
        
        if self.config.stop_tokens:
            completion_kwargs["stop"] = self.config.stop_tokens
        
        result = self.model.create_chat_completion(**completion_kwargs)

        latency = time.time() - start_time

        response = {
            "reply": result,
            "latency_seconds": latency,
        }
        
        if context:
            response["context_used"] = context
            if context_metadata:
                response["search_metadata"] = context_metadata
        
        return response

    def _build_chat_messages(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages array for create_chat_completion API."""
        if messages is not None and prompt is not None:
            raise ValueError("Cannot provide both messages and prompt")
        if messages is None and prompt is None:
            raise ValueError("Must provide either messages or prompt")
        
        chat_messages = []
        
        # Build system message
        system_parts = []
        
        if instructions:
            system_parts.append(instructions)
        
        if context:
            context_instruction = (
                "You have been provided with relevant information from documents below. "
                "Use this information to answer the user's question accurately. "
                "Reference specific details from the context when relevant."
            )
            system_parts.append(context_instruction)
            system_parts.append(f"\n=== DOCUMENT CONTEXT ===\n{context}\n=== END CONTEXT ===\n")
        
        if system_parts:
            chat_messages.append({
                "role": "system",
                "content": "\n\n".join(system_parts)
            })
        
        # Add conversation messages
        if messages is not None:
            for msg in messages:
                chat_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        else:
            chat_messages.append({
                "role": "user",
                "content": prompt
            })
        
        return chat_messages