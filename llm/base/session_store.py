from abc import ABC, abstractmethod
from llm.base.session import Session

class SessionStore(ABC):
    """Base class for session storage. Extend this for persistent storage (DB, file, etc)."""
    
    @abstractmethod
    def get(self, session_id: str) -> Session:
        """Get a session by ID. Implementation decides behavior if not found."""
        pass
    
    @abstractmethod
    def save(self, session: Session):
        """Save a session. Implementation handles persistence."""
        pass
    
    @abstractmethod
    def delete(self, session_id: str):
        """Delete a session. Implementation handles cleanup."""
        pass
    
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass