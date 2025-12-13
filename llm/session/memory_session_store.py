from typing import Dict
from llm.base.session import Session
from llm.base.session_store import SessionStore


class InMemorySessionStore(SessionStore):
    """In-memory session store implementation."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
    
    def get(self, session_id: str) -> Session:
        """Get or create a session if it doesn't exist."""
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]
    
    def save(self, session: Session):
        """Save session to memory."""
        self._sessions[session.session_id] = session
    
    def delete(self, session_id: str):
        """Delete a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists in memory."""
        return session_id in self._sessions
