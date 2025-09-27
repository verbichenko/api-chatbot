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


def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


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


def format_sources(sources: List[str]) -> str:
    """Format sources for display in responses."""
    if not sources:
        return ""
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        formatted_sources.append(f"{i}. {source}")
    
    return "\n\n**Sources:**\n" + "\n".join(formatted_sources)


def combine_sources(source_lists: List[List[str]]) -> List[str]:
    """Combine multiple source lists, removing duplicates while preserving order."""
    seen = set()
    combined = []
    
    for source_list in source_lists:
        for source in source_list:
            if source not in seen:
                seen.add(source)
                combined.append(source)
    
    return combined


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_request_item_summary(request_items: List[Any]) -> str:
    """Format a summary of request items for logging/debugging."""
    if not request_items:
        return "No request items"
    
    summary_lines = []
    for item in request_items:
        summary_lines.append(f"- {item.id}: {item.category} ({item.priority}) - {truncate_text(item.request_text, 100)}")
    
    return "\n".join(summary_lines)


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


def format_conversation_context(messages: List[BaseMessage], max_messages: int = 10) -> str:
    """Format recent conversation context for agents."""
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    context_lines = []
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            context_lines.append(f"Customer: {msg.content}")
        elif isinstance(msg, AIMessage):
            context_lines.append(f"Assistant: {msg.content}")
    
    return "\n".join(context_lines)


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
        return f"Error in {context}: {error_type} - {error_msg}"
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


def calculate_confidence_score(factors: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Calculate a confidence score based on multiple factors."""
    if not factors:
        return 0.0
    
    if weights is None:
        weights = {key: 1.0 for key in factors.keys()}
    
    weighted_sum = sum(factors[key] * weights.get(key, 1.0) for key in factors.keys())
    total_weight = sum(weights.get(key, 1.0) for key in factors.keys())
    
    if total_weight == 0:
        return 0.0
    
    confidence = weighted_sum / total_weight
    return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1