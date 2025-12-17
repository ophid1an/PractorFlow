from dataclasses import dataclass, field
import os
from typing import List, Optional
from dotenv import load_dotenv

from llm.knowledge.chroma_knowledge_config import ChromaKnowledgeStoreConfig
from llm.llm_config import LLMConfig

load_dotenv(dotenv_path="config/options/options.env", override=True)


def parse_list_string(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None

    tokens = [t.strip() for t in value.split(",") if t.strip()]
    return tokens or None


def load_knowledge_chroma_config() -> Optional[ChromaKnowledgeStoreConfig]:
    kbtype = os.getenv("KB_TYPE", "chromadb")
    if kbtype == "chromadb":
        knowledgeStore = ChromaKnowledgeStoreConfig(
            persist_directory=os.getenv("KB_CHROMA_PERSIST_DIRECTORY", "chroma_db"),
            retrieval_collection_name=os.getenv(
                "KB_CHROMA_RETRIEVE_COLLECTION", "knowledge_retrieval"
            ),
            context_collection_name=os.getenv(
                "KB_CHROMA_CONTEXT_COLLECTION", "knowledge_context"
            ),
            documents_collection_name=os.getenv(
                "KB_CHROMA_DOCUMENT_COLLECTION", "knowledge_documents"
            ),
            batch_size=int(os.getenv("KB_CHROMA_BATCH_SIZE", "100")),
            embedding_model_name=os.getenv(
                "KB_CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            embedding_cache_dir=os.getenv("KB_CHROMA_EMBEDDING_MODEL_DIR", "./models"),
            retrieval_chunk_size=int(
                os.getenv("KB_CHROMA_RETRIEVAL_CHUNK_SIZE", "128")
            ),
            retrieval_chunk_overlap=int(
                os.getenv("KB_CHROMA_RETRIEVAL_CHUNK_OVERLAP", "20")
            ),
            context_chunk_size=int(os.getenv("KB_CHROMA_CONTEXT_CHUNK_SIZE", "1024")),
            context_chunk_overlap=int(
                os.getenv("KB_CHROMA_CONTEXT_CHUNK_OVERLAP", "100")
            ),
        )
    else:
        knowledgeStore = None
    return knowledgeStore


@dataclass
class LoggerConfig:
    RunnerLevel: str = os.getenv("LOG_RUNNER_LEVEL", "INFO")
    DocumentLevel: str = os.getenv("LOG_DOC_LEVEL", "INFO")
    KnowledgeLevel: str = os.getenv("LOG_KNOWLEDGE_LEVEL", "INFO")
    ModelPoolLevel: str = os.getenv("LOG_MODEL_POOL_LEVEL", "INFO")
    ToolLevel: str = os.getenv("LOG_TOOL_LEVEL", "INFO")


@dataclass
class AppConfig:
    LoggerConfiguration: LoggerConfig = LoggerConfig()
    ModelConfiguration: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model_name=os.getenv(
                "LLM_MODEL",
                "Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_k_m.gguf",
            ),
            device=os.getenv("LLM_DEVICE", "auto"),
            dtype=os.getenv("LLM_DTYPE", "auto"),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "2048")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
            quantization=os.getenv("LLM_QUANTIZATION"),
            models_dir=os.getenv("LLM_MODELS_DIR", "auto"),
            n_gpu_layers=int(os.getenv("LLM_GPU_LAYERS", "-1")),
            n_ctx=int(os.getenv("LLM_N_CTX", "32768")),
            n_batch=int(os.getenv("LLM_N_BATCH", "2048")),
            backend=os.getenv("LLM_BACKEND", "llama_cpp"),
            stop_tokens=parse_list_string(os.getenv("LLM_STOP_TOKENS")),
            max_search_results=int(os.getenv("LLM_MAX_SEARCH_RESULTS", "5")),
        )
    )
    KnowledgeChromaConfiguration: Optional[ChromaKnowledgeStoreConfig] = field(
        default_factory=lambda: load_knowledge_chroma_config()
    )


appConfiguration: AppConfig = AppConfig()
