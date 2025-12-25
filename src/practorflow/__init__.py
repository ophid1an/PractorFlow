"""
PractorFlow - Private AI Service for local LLM inference with RAG support.

A self-hosted AI service for organizations that cannot share their data
with third-party APIs. Supports llama.cpp and transformers backends with
ChromaDB-based knowledge storage for RAG workflows.
"""

__version__ = "0.0.1"
__author__ = "Vasileios Bouzoukos"

# LLM module - core inference components
from practorflow.llm import (
    LLMConfig,
    LLMRunner,
    StreamChunk,
    ModelHandle,
    ModelPool,
    TransformersRunner,
    LlamaCppRunner,
    create_runner,
)

# Logger
from practorflow.logger import get_logger, ColorFormatter

# Settings
from practorflow.settings.app_settings import (
    AppConfig,
    LoggerConfig,
    appConfiguration,
    load_configuration,
)

__all__ = [
    # Version
    "__version__",
    # LLM
    "LLMConfig",
    "LLMRunner",
    "StreamChunk",
    "ModelHandle",
    "ModelPool",
    "TransformersRunner",
    "LlamaCppRunner",
    "create_runner",
    # Logger
    "get_logger",
    "ColorFormatter",
    # Settings
    "AppConfig",
    "LoggerConfig",
    "appConfiguration",
    "load_configuration",
]