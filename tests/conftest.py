"""Test configuration and fixtures."""

import pytest
from unittest.mock import Mock
from api_support_chatbot.configuration import Configuration, MCPServerConfig, MCPTransport


@pytest.fixture
def mock_configuration():
    """Create a mock configuration for testing."""
    return Configuration(
        azure_openai_endpoint="https://test.openai.azure.com/",
        azure_openai_api_key="test-key",
        azure_openai_api_version="2024-02-01",
        azure_openai_deployment_name="test-gpt-4",
        azure_openai_model_name="gpt-4",
        mcp_servers={
            "test_server": MCPServerConfig(
                url="http://localhost:8000/mcp/",
                transport=MCPTransport.STREAMABLE_HTTP
            )
        },
        max_retries=2,
        max_concurrent_requests=3,
        request_timeout=10,
        enable_clarification=True
    )


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    client = Mock()
    client.get_tools.return_value = []
    return client


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    from langchain_core.messages import HumanMessage, AIMessage
    
    return [
        HumanMessage(content="How do I authenticate with your API?"),
        AIMessage(content="I'll help you with API authentication. Let me gather some information."),
        HumanMessage(content="I'm specifically interested in OAuth2 implementation.")
    ]