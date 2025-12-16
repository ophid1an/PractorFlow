import time
import os
from typing import Dict, Any, Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.llm_config import LLMConfig
from llm.base.llm_runner import LLMRunner
from llm.base.session import Session
from llm.knowledge.knowledge_store import KnowledgeStore
from llm.tools.knowledge_search import KnowledgeSearchTool


class TransformersRunner(LLMRunner):
    """
    Runner for HuggingFace Transformers models with tool-based RAG.
    
    Supports models like:
    - OpenAI GPT-OSS (openai/gpt-oss-20b, openai/gpt-oss-120b)
    - Qwen2 Instruct models
    - Llama models
    - And other AutoModelForCausalLM compatible models
    """

    def __init__(
        self,
        config: LLMConfig,
        session: Optional[Session] = None,
        knowledge_store: Optional[KnowledgeStore] = None
    ):
        super().__init__(config, session, knowledge_store)
        
        os.makedirs(config.models_dir, exist_ok=True)

        print(f"[TransformersRunner] Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=config.models_dir,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build model loading kwargs
        model_kwargs = {
            "cache_dir": config.models_dir,
            "trust_remote_code": True,
        }
        
        # Handle dtype - use "auto" for models like GPT-OSS that have specific requirements
        if config.dtype == "auto" or config.dtype is None:
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = config.dtype
        
        # Handle device mapping
        if config.device == "auto":
            model_kwargs["device_map"] = "auto"
        elif config.device != "cpu":
            model_kwargs["device_map"] = config.device
        
        # Handle quantization
        if config.quantization:
            if config.quantization == "4bit":
                model_kwargs["load_in_4bit"] = True
            elif config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True

        # Load model
        print(f"[TransformersRunner] Loading with kwargs: {model_kwargs}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs and config.device == "cpu":
            self.model = self.model.to(config.device)
        
        self.model.eval()
        
        # Determine actual device
        if hasattr(self.model, 'device'):
            self._device = self.model.device
        else:
            self._device = next(self.model.parameters()).device

        # Get context length
        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_context_length = self.model.config.max_position_embeddings
            print(f"[TransformersRunner] Context window: {self.max_context_length}")
        else:
            self.max_context_length = config.n_ctx
            print(f"[TransformersRunner] Context window (from config): {self.max_context_length}")

        print(f"[TransformersRunner] Model loaded on {self._device}")
        
        # Register knowledge search tool if knowledge store is available
        if self.knowledge_store:
            knowledge_tool = KnowledgeSearchTool(
                knowledge_store=self.knowledge_store,
                default_top_k=config.max_search_results
            )
            self.tool_registry.register(knowledge_tool)
            print("[TransformersRunner] Knowledge search tool registered")

    def get_chat_reply_structure(self) -> Optional[str]:
        """Get the chat template from tokenizer."""
        try:
            if hasattr(self.tokenizer, 'chat_template'):
                return self.tokenizer.chat_template
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
            last_result = self.tool_registry.get_last_result()
            if last_result and last_result.metadata:
                context_metadata = last_result.metadata
            self.tool_registry.clear_last_result()
            
            print(f"[TransformersRunner] Using search context ({len(context)} chars)")

        # Build chat messages
        chat_messages = self._build_chat_messages(messages, prompt, instructions, context)

        temp = self._get_temperature(temperature)
        tp = self._get_top_p(top_p)

        print("[TransformersRunner] Generating...")
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = self._format_messages_fallback(chat_messages)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length - self.max_new_tokens if self.max_context_length else None
        ).to(self._device)
        
        input_length = inputs['input_ids'].shape[1]
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temp > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = tp
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode response (only new tokens)
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        latency = time.time() - start_time

        response = {
            "reply": response_text,
            "latency_seconds": latency,
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": len(generated_tokens),
                "total_tokens": input_length + len(generated_tokens),
            }
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
        """Build chat messages for the model."""
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
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback message formatting for tokenizers without chat template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)