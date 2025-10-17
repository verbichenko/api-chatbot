# API Support Chatbot

A LangGraph-based multi-agent chatbot for providing API support to customers. The chatbot uses Azure OpenAI and integrates with MCP (Model Context Protocol) servers to access external tools and knowledge sources.

## Architecture

The chatbot consists of four main agents:

1. **Get Request Details Agent**: Processes customer requests, validates scope, and requests additional details if needed
2. **Response Coordinator Agent**: Extracts and delegates requests to response agents
3. **Response Agent**: Uses MCP tools to gather relevant context and create responses
4. **Response Assembler Agent**: Combines all responses into a final comprehensive answer

## Features

- **Multi-Agent Architecture**: Specialized agents for different aspects of request processing
- **Azure OpenAI Integration**: Uses Azure OpenAI models for natural language understanding and generation
- **MCP Tool Integration**: Connects to external MCP servers for accessing documentation and support context
- **Configurable**: Easy configuration management for models, MCP servers, and agent behavior
- **Async Support**: Full asynchronous implementation for scalability

## Setup

1. **Environment Variables**:
   Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

3. **Run the Example Console Chatbot**:
   ```bash
   python example.py
   ```

4. **Run the Production Server** (LangServe):
   ```bash
   python serve.py
   ```
   
   See [QUICKSTART.md](QUICKSTART.md) for a quick start guide and [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Configuration

The chatbot can be configured through environment variables or the `Configuration` class:

- **Azure OpenAI**: Set your Azure OpenAI endpoint, API key, and model deployment names
- **MCP Servers**: Configure connections to MCP servers providing tools like `readme` and `retrieve_support_context`
- **Agent Settings**: Customize agent behavior, retry logic, and response formatting

## Usage

```python
from api_support_chatbot.chatbot import create_chatbot_graph
from api_support_chatbot.configuration import Configuration

# Create configuration
config = Configuration.from_env()

# Create the chatbot graph
graph = create_chatbot_graph()

# Run the chatbot
response = await graph.ainvoke(
    {"messages": [HumanMessage(content="How do I authenticate with your API?")]},
    config={"configurable": config.model_dump()}
)
```

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Format code:
```bash
black src/
ruff check src/
```

Type checking:
```bash
mypy src/
```

## License

MIT License