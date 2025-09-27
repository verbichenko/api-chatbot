"""Tests for utility functions."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from api_support_chatbot.utils import (
    get_today_str,
    generate_request_id,
    generate_conversation_id,
    extract_human_messages,
    extract_last_human_message,
    format_sources,
    combine_sources,
    truncate_text,
    validate_mcp_connection_config,
    calculate_confidence_score
)


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_today_str(self):
        """Test today string generation."""
        today = get_today_str()
        assert len(today) == 10  # YYYY-MM-DD format
        assert today.count('-') == 2
        
        # Verify it's a valid date
        datetime.strptime(today, "%Y-%m-%d")
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        id1 = generate_request_id()
        id2 = generate_request_id()
        
        assert len(id1) == 8
        assert len(id2) == 8
        assert id1 != id2  # Should be unique
    
    def test_generate_conversation_id(self):
        """Test conversation ID generation."""
        id1 = generate_conversation_id()
        id2 = generate_conversation_id()
        
        assert len(id1) > 8  # Full UUID
        assert len(id2) > 8
        assert id1 != id2  # Should be unique
    
    def test_extract_human_messages(self, sample_messages):
        """Test human message extraction."""
        human_messages = extract_human_messages(sample_messages)
        
        assert len(human_messages) == 2
        assert "How do I authenticate with your API?" in human_messages
        assert "I'm specifically interested in OAuth2 implementation." in human_messages
    
    def test_extract_last_human_message(self, sample_messages):
        """Test last human message extraction."""
        last_message = extract_last_human_message(sample_messages)
        
        assert last_message == "I'm specifically interested in OAuth2 implementation."
    
    def test_extract_last_human_message_empty(self):
        """Test last human message extraction with empty list."""
        last_message = extract_last_human_message([])
        assert last_message is None
    
    def test_format_sources(self):
        """Test source formatting."""
        sources = ["API Documentation", "Stack Overflow", "GitHub Issue #123"]
        formatted = format_sources(sources)
        
        assert "**Sources:**" in formatted
        assert "1. API Documentation" in formatted
        assert "2. Stack Overflow" in formatted
        assert "3. GitHub Issue #123" in formatted
    
    def test_format_sources_empty(self):
        """Test source formatting with empty list."""
        formatted = format_sources([])
        assert formatted == ""
    
    def test_combine_sources(self):
        """Test source combination with duplicate removal."""
        sources1 = ["Source A", "Source B"]
        sources2 = ["Source B", "Source C"]
        sources3 = ["Source C", "Source D"]
        
        combined = combine_sources([sources1, sources2, sources3])
        
        assert len(combined) == 4
        assert combined == ["Source A", "Source B", "Source C", "Source D"]
    
    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "This is a very long text that should be truncated " * 10
        truncated = truncate_text(long_text, 50)
        
        assert len(truncated) <= 50
        assert truncated.endswith("...")
    
    def test_truncate_text_short(self):
        """Test text truncation with short text."""
        short_text = "Short text"
        truncated = truncate_text(short_text, 50)
        
        assert truncated == short_text
    
    def test_validate_mcp_connection_config_streamable_http(self):
        """Test MCP connection config validation for streamable HTTP."""
        config = {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp/"
        }
        
        assert validate_mcp_connection_config(config) is True
    
    def test_validate_mcp_connection_config_stdio(self):
        """Test MCP connection config validation for stdio."""
        config = {
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"]
        }
        
        assert validate_mcp_connection_config(config) is True
    
    def test_validate_mcp_connection_config_invalid(self):
        """Test MCP connection config validation with invalid config."""
        config = {
            "transport": "streamable_http"
            # Missing required 'url' field
        }
        
        assert validate_mcp_connection_config(config) is False
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        factors = {
            "clarity": 0.8,
            "completeness": 0.6,
            "accuracy": 0.9
        }
        
        weights = {
            "clarity": 1.0,
            "completeness": 0.5,
            "accuracy": 1.5
        }
        
        confidence = calculate_confidence_score(factors, weights)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high given the factors
    
    def test_calculate_confidence_score_no_weights(self):
        """Test confidence score calculation without weights."""
        factors = {
            "factor1": 0.8,
            "factor2": 0.6
        }
        
        confidence = calculate_confidence_score(factors)
        
        assert confidence == 0.7  # Average of the factors
    
    def test_calculate_confidence_score_empty(self):
        """Test confidence score calculation with empty factors."""
        confidence = calculate_confidence_score({})
        assert confidence == 0.0