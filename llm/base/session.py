from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Union
import uuid


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # "user" or "assistant"
    content: Union[str, List[Dict[str, Any]]]  # Can be string or structured content
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: str = "message"
    status: str = "completed"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with agent SDKs."""
        return {
            "id": self.id,
            "role": self.role,
            "type": self.type,
            "status": self.status,
            "content": self.content
        }
    
    def get_text_content(self) -> str:
        """Extract text content from message."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Extract text from structured content
            texts = []
            for item in self.content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
            return "\n".join(texts)
        return str(self.content)


@dataclass
class Session:
    """Conversation session with message history."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    instructions: str = None  # System-level instructions
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)