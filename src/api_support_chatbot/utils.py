"""Utility functions for the API Support Chatbot."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def get_today_str() -> str:
    """Get today's date as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:8]


def extract_human_messages(messages: List[BaseMessage]) -> List[str]:
    """Extract text content from human messages."""
    return [
        msg.content for msg in messages 
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
    ]


def extract_last_human_message(messages: List[BaseMessage]) -> Optional[str]:
    """Extract the last human message content."""
    human_messages = extract_human_messages(messages)
    return human_messages[-1] if human_messages else None


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def validate_mcp_connection_config(config: Dict[str, Any]) -> bool:
    """Validate MCP connection configuration."""
    required_fields = ["transport"]
    
    for field in required_fields:
        if field not in config:
            return False
    
    transport = config["transport"]
    
    if transport == "streamable_http":
        return "url" in config
    elif transport == "stdio":
        return "command" in config
    elif transport in ["websocket", "sse"]:
        return "url" in config
    
    return False


def safe_get_tool_name(tool: Any) -> str:
    """Safely extract tool name from tool object."""
    if hasattr(tool, 'name'):
        return tool.name
    elif hasattr(tool, '__name__'):
        return tool.__name__
    else:
        return str(tool)

async def run_with_timeout(coro, timeout: float):
    """Run a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout} seconds")


def create_error_message(error: Exception, context: str = "") -> str:
    """Create a formatted error message."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"{context}: {error_type} - {error_msg}"
    else:
        return f"{error_type}: {error_msg}"


def log_agent_action(agent_name: str, action: str, details: Dict[str, Any] = None) -> None:
    """Log agent actions for debugging and monitoring."""

    # TODO add logs using logging module

    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "agent": agent_name,
        "action": action
    }
    
    if details:
        log_entry.update(details)
    # In a production environment, this would log to a proper logging system
    print(f"[{timestamp}] {agent_name}: {action}", flush=True)
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}", flush=True)
