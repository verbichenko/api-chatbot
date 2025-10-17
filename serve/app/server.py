#!/usr/bin/env python
"""Production server for the API Support Chatbot using LangServe."""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Use vendored patched version of langserve to support checkpointer keys
from src.api_support_chatbot.vendor.langserve_patched import add_routes
from src.api_support_chatbot.chatbot import create_chatbot_graph


# Load environment variables
load_dotenv()

# Create the FastAPI app
app = FastAPI(
    title="API Support Chatbot",
    version="0.1.0",
    description="LangGraph-based chatbot for API support using Azure OpenAI and MCP tools",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the chatbot graph
graph = create_chatbot_graph()

# Add routes for the chatbot
# The chatbot will be available at /chatbot
add_routes(
    app,
    graph,
    path="/chatbot",
    enabled_endpoints=["invoke", "stream", "stream_log", "input_schema", "output_schema"],
    playground_type="chat"
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "API Support Chatbot"}

# Root endpoint with API information
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "API Support Chatbot API",
        "version": "0.1.0",
        "endpoints": {
            "chatbot": "/chatbot",
            "docs": "/docs",
            "health": "/health",
        },
        "description": "LangGraph-based chatbot for API support",
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("LANGSERVE_PORT", "8000"))
    host = os.getenv("LANGSERVE_HOST", "0.0.0.0")
    
    print(f"Starting API Support Chatbot server on {host}:{port}")
    print(f"API documentation available at http://{host}:{port}/docs")
    print(f"Chatbot endpoint available at http://{host}:{port}/chatbot")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
