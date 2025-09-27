"""Tests for configuration management."""

import os
import pytest
from unittest.mock import patch

from api_support_chatbot.configuration import Configuration, MCPServerConfig, MCPTransport


class TestConfiguration:
    """Tests for Configuration class."""
    
    def test_configuration_from_env(self):
        """Test configuration creation from environment variables."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'AZURE_OPENAI_DEPLOYMENT_NAME': 'test-deployment',
            'MAX_RETRIES': '5',
            'ENABLE_CLARIFICATION': 'false'
        }):
            config = Configuration.from_env()
            
            assert config.azure_openai_endpoint == 'https://test.openai.azure.com/'
            assert config.azure_openai_api_key == 'test-key'
            assert config.azure_openai_deployment_name == 'test-deployment'
            assert config.max_retries == 5
            assert config.enable_clarification is False
    
    def test_get_azure_openai_params(self, mock_configuration):
        """Test Azure OpenAI parameters extraction."""
        params = mock_configuration.get_azure_openai_params()
        
        assert 'azure_endpoint' in params
        assert 'api_key' in params
        assert 'api_version' in params
        assert 'azure_deployment' in params
        assert 'model' in params
        assert params['azure_endpoint'] == 'https://test.openai.azure.com/'
        assert params['api_key'] == 'test-key'
    
    def test_get_mcp_connections(self, mock_configuration):
        """Test MCP connections extraction."""
        connections = mock_configuration.get_mcp_connections()
        
        assert 'test_server' in connections
        assert connections['test_server']['transport'] == 'streamable_http'
        assert connections['test_server']['url'] == 'http://localhost:8000/mcp/'


class TestMCPServerConfig:
    """Tests for MCPServerConfig class."""
    
    def test_streamable_http_connection(self):
        """Test streamable HTTP connection configuration."""
        config = MCPServerConfig(
            url="http://localhost:9000/mcp/",
            transport=MCPTransport.STREAMABLE_HTTP
        )
        
        connection_dict = config.to_connection_dict()
        
        assert connection_dict['url'] == "http://localhost:9000/mcp/"
        assert connection_dict['transport'] == 'streamable_http'
        assert connection_dict['timeout'] == 30
    
    def test_stdio_connection(self):
        """Test stdio connection configuration."""
        config = MCPServerConfig(
            command="python",
            args=["server.py"],
            transport=MCPTransport.STDIO
        )
        
        connection_dict = config.to_connection_dict()
        
        assert connection_dict['command'] == "python"
        assert connection_dict['args'] == ["server.py"]
        assert connection_dict['transport'] == 'stdio'
    
    def test_unsupported_transport(self):
        """Test unsupported transport raises error."""
        config = MCPServerConfig(transport=MCPTransport.WEBSOCKET)
        
        with pytest.raises(ValueError, match="Unsupported transport"):
            config.to_connection_dict()