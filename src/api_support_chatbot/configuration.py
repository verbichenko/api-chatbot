"""Configuration management for the API Support Chatbot."""

import os
from enum import Enum
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class MCPTransport(Enum):
    """MCP transport types."""
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    WEBSOCKET = "websocket"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection."""
    
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    timeout: int = 30
    
    def to_connection_dict(self) -> Dict[str, Any]:
        """Convert to connection dictionary format for MCP client."""
        if self.transport == MCPTransport.STREAMABLE_HTTP:
            return {
                "url": self.url,
                "transport": self.transport.value,
                "timeout": self.timeout
            }
        elif self.transport == MCPTransport.STDIO:
            return {
                "command": self.command,
                "args": self.args or [],
                "transport": self.transport.value,
                "timeout": self.timeout
            }
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")


class Configuration(BaseModel):
    """Main configuration for the API Support Chatbot."""
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        description="Azure OpenAI service endpoint"
    )
    azure_openai_api_key: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""),
        description="Azure OpenAI API key"
    )
    azure_openai_api_version: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        description="Azure OpenAI API version"
    )
    azure_openai_deployment_name: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini"),
        description="Azure OpenAI deployment name"
    )
    azure_openai_model_name: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4.1-mini"),
        description="Azure OpenAI model name"
    )
    
    # MCP Server Configuration
    mcp_servers: Dict[str, MCPServerConfig] = Field(
        default_factory=lambda: {
            "api_support": MCPServerConfig(
                url=os.getenv("MCP_API_SUPPORT_SERVER_URL", "http://localhost:9000/mcp/"),
                transport=MCPTransport(os.getenv("MCP_API_SUPPORT_SERVER_TRANSPORT", "streamable_http"))
            )
        },
        description="MCP server configurations"
    )
    
    # Chatbot Configuration
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")),
        description="Maximum number of retries for model calls"
    )
    max_concurrent_requests: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
        description="Maximum number of concurrent requests"
    )
    request_timeout: int = Field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "30")),
        description="Request timeout in seconds"
    )
    enable_clarification: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_CLARIFICATION", "true").lower() == "true",
        description="Whether to enable clarification requests"
    )
    
    # Model Configuration
    model_temperature: float = Field(
        default=0.1,
        description="Temperature for model responses"
    )
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for model responses"
    )
    
    @classmethod
    def from_env(cls) -> "Configuration":
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create configuration from LangGraph runnable config."""
        if config is None or "configurable" not in config:
            return cls.from_env()
        
        configurable = config["configurable"]
        if isinstance(configurable, dict):
            return cls(**configurable)
        elif isinstance(configurable, Configuration):
            return configurable
        else:
            return cls.from_env()
    
    def get_azure_openai_params(self) -> Dict[str, Any]:
        """Get Azure OpenAI connection parameters."""
        return {
            "": self.azure_openai_endpoint,
            "api_key": self.azure_openai_api_key,
            "api_version": self.azure_openai_api_version,
            "azure_deployment": self.azure_openai_deployment_name,
            "model": self.azure_openai_model_name,
            "temperature": self.model_temperature,
            "max_tokens": self.max_tokens,
        }
    
    def get_mcp_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP server connections dictionary."""
        return {
            name: server.to_connection_dict() 
            for name, server in self.mcp_servers.items()
        }