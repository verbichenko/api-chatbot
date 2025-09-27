"""Main entry point for the API Support Chatbot."""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from api_support_chatbot.chatbot import create_chatbot_graph
from api_support_chatbot.configuration import Configuration


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


async def main() -> None:
    """Main function to run the chatbot."""
    # Load environment
    load_environment()
    
    # Create configuration
    config = Configuration.from_env()
    
    # Validate configuration
    if not config.azure_openai_endpoint or not config.azure_openai_api_key:
        raise ValueError("Azure OpenAI configuration is required. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.")
    
    # Create the chatbot graph
    graph = create_chatbot_graph()
    
    print("API Support Chatbot initialized successfully!")
    print(f"Using Azure OpenAI endpoint: {config.azure_openai_endpoint}")
    print(f"Configured MCP servers: {list(config.mcp_servers.keys())}")
    print("\nChatbot is ready to handle API support requests.")
    
    # In a production environment, this would typically be served via FastAPI or similar
    # For now, we just confirm the setup is working
    return graph


def create_graph():
    """Entry point for LangGraph server - matches langgraph.json configuration."""
    load_environment()
    return create_chatbot_graph()


if __name__ == "__main__":
    asyncio.run(main())