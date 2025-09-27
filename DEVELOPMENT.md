# API Support Chatbot Development Documentation

## Architecture Overview

The API Support Chatbot is built using LangGraph's multi-agent architecture with four specialized agents:

### 1. Get Request Details Agent (`get_request_details`)
**Purpose**: Analyzes incoming customer requests and determines scope and clarification needs.

**Responsibilities**:
- Validates if the request is within API support scope
- Extracts specific requirements from the customer message
- Identifies needed clarifications
- Calculates confidence score for the analysis

**Input**: Customer messages from the conversation
**Output**: `RequestDetails` object with analysis results
**Next Steps**: Either `ask_clarification` or `coordinate_response`

### 2. Response Coordinator Agent (`coordinate_response`)
**Purpose**: Breaks down complex requests into focused, actionable items for delegation.

**Responsibilities**:
- Analyzes request details from the previous agent
- Creates specific `RequestItem` objects for each aspect of the request
- Categorizes items (auth, endpoints, errors, documentation, etc.)
- Assigns priorities (high, medium, low)
- Determines required context/tools for each item

**Input**: `RequestDetails` from the previous agent
**Output**: List of `RequestItem` objects via `Send` commands
**Next Steps**: Delegates each item to `generate_response` agents in parallel

### 3. Response Agent (`generate_response`)
**Purpose**: Individual agents that handle specific request items using MCP tools.

**Responsibilities**:
- Processes individual `RequestItem` objects
- Connects to MCP servers to access external tools
- Uses `readme_first` resource to understand capabilities
- Uses `retrieve_support_context` tool to gather relevant information
- Generates focused responses based on retrieved context
- Calculates confidence scores for responses

**Input**: Individual `RequestItem` and MCP context
**Output**: `ResponseItem` with generated content and metadata
**Next Steps**: All response items are collected for assembly

### 4. Response Assembler Agent (`assemble_final_response`)
**Purpose**: Combines all response items into a coherent, comprehensive final response.

**Responsibilities**:
- Reviews all `ResponseItem` objects from response agents
- Organizes content in logical flow
- Eliminates redundancy while preserving important details
- Ensures consistency across different responses
- Compiles all sources and references
- Generates helpful follow-up questions

**Input**: Collection of `ResponseItem` objects
**Output**: `FinalResponse` with assembled content
**Next Steps**: Returns final response to customer

## State Management

### Primary States
- **ChatbotState**: Main state containing all conversation and processing data
- **RequestDetails**: Structured analysis of customer requests
- **RequestItem**: Individual request components for delegation
- **ResponseItem**: Generated responses for specific request items
- **FinalResponse**: Final assembled response for the customer

### Agent-Specific States
- **RequestDetailsState**: For the request details agent
- **CoordinatorState**: For the response coordinator
- **ResponseAgentState**: For individual response agents
- **AssemblerState**: For the response assembler

## Configuration

### Azure OpenAI Integration
The chatbot uses Azure OpenAI through the `init_chat_model` function with configurable parameters:
- Endpoint and API key configuration
- Model deployment names
- Temperature and token limits
- Retry logic and error handling

### MCP Integration
Model Context Protocol (MCP) integration provides access to external tools:
- **Connection Types**: Streamable HTTP, stdio, WebSocket, SSE
- **Available Tools**: 
  - `readme_first`: Get capability overview
  - `retrieve_support_context`: Access support documentation and context

### Environment Configuration
All settings can be configured through environment variables:
- Azure OpenAI settings
- MCP server connections
- Chatbot behavior settings (retries, timeouts, clarification enabling)

## Error Handling

### Graceful Degradation
- Each agent includes comprehensive error handling
- Fallback responses when tools fail
- Error state tracking throughout the conversation
- Retry logic for transient failures

### Logging and Monitoring
- Agent action logging for debugging
- Performance monitoring hooks
- Error tracking and reporting
- Conversation flow visibility

## Testing Strategy

### Unit Tests
- Configuration management tests
- Utility function tests
- State validation tests
- Error handling tests

### Integration Tests
- Agent workflow tests
- MCP tool integration tests
- End-to-end conversation tests
- Performance and scalability tests

## Deployment

### LangGraph Server
- Configured via `langgraph.json`
- Graph entry point: `create_graph`
- Environment configuration through `.env`

### Scalability Considerations
- Parallel processing of request items
- Async/await throughout for non-blocking operations
- Configurable concurrency limits
- Timeout handling for external services

## Development Workflow

### Setup
1. Install dependencies: `pip install -e .`
2. Configure environment: `cp .env.example .env`
3. Set Azure OpenAI credentials
4. Configure MCP server connections

### Testing
1. Run unit tests: `pytest`
2. Run example: `python example.py`
3. Start LangGraph dev server: `uvx langgraph dev`

### Customization
- Modify prompts in `prompts.py`
- Adjust agent behavior in `chatbot.py`
- Add new tools through MCP server configuration
- Extend state definitions for new features

## Performance Optimization

### Parallel Processing
- Request items are processed in parallel by response agents
- MCP tool calls are optimized for concurrent execution
- Async operations throughout the pipeline

### Caching Strategy
- Consider caching frequently requested documentation
- Cache MCP tool responses for common queries
- Implement session-based context caching

### Resource Management
- Connection pooling for MCP servers
- Request timeouts and circuit breakers
- Memory usage optimization for large conversations