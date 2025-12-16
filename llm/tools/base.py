"""
Base tool interface for LLM tool calling.

Defines the abstract interface that all tools must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None  # Allowed values


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_context_string(self) -> str:
        """Convert result to string suitable for LLM context."""
        if not self.success:
            return f"Tool execution failed: {self.error}"
        
        if isinstance(self.data, str):
            return self.data
        
        if isinstance(self.data, dict):
            return self._format_dict(self.data)
        
        if isinstance(self.data, list):
            return self._format_list(self.data)
        
        return str(self.data)
    
    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary for readable output."""
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_list(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    
    def _format_list(self, lst: List[Any], indent: int = 0) -> str:
        """Format list for readable output."""
        lines = []
        prefix = "  " * indent
        for i, item in enumerate(lst, 1):
            if isinstance(item, dict):
                lines.append(f"{prefix}{i}.")
                lines.append(self._format_dict(item, indent + 1))
            else:
                lines.append(f"{prefix}{i}. {item}")
        return "\n".join(lines)


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools are callable units that can be invoked by the LLM runner
    to perform specific actions like searching knowledge, fetching data, etc.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what the tool does.
        This is provided to the LLM to help it decide when to use the tool.
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """List of parameters the tool accepts."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with success status and data or error
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        Validate provided parameters against tool definition.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        provided = set(kwargs.keys())
        
        for param in self.parameters:
            if param.required and param.name not in provided:
                if param.default is None:
                    return f"Missing required parameter: {param.name}"
        
        defined_names = {p.name for p in self.parameters}
        unknown = provided - defined_names
        if unknown:
            return f"Unknown parameters: {', '.join(unknown)}"
        
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool schema in OpenAI/Anthropic function calling format.
        
        Returns:
            Dictionary representing the tool schema
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    def __call__(self, **kwargs) -> ToolResult:
        """Allow tool to be called directly."""
        validation_error = self.validate_parameters(**kwargs)
        if validation_error:
            return ToolResult(success=False, error=validation_error)
        
        return self.execute(**kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"