# API Support Chatbot - AI Agent Instructions

## Architecture: Multi-Agent LangGraph System

This is a **LangGraph-based multi-agent chatbot** with a 4-agent pipeline processing customer API support requests:

1. **Get Request Details** (`get_request_details`) - Validates scope, extracts requirements, asks clarifications (max 3 attempts tracked in state)
2. **Response Coordinator** (`coordinate_response`) - Breaks requests into `RequestItem` objects, each with unique ID via `generate_request_id()`
3. **Response Agent** (`generate_response`) - **Runs in parallel** for each RequestItem using MCP tools with manual tool execution loop (2 iterations max)
4. **Response Assembler** (`assemble_final_response`) - Deferred node that combines all responses into final output

**Critical Flow Pattern**: Agents communicate via `Command` objects (not direct edges). Each agent returns `Command(update={...}, goto="next_node")` or `Command(graph=END, ...)` for errors. The coordinator uses `Send` commands for parallel fan-out to response agents.

## State Management Pattern

- **Main State**: `ChatbotState` (extends `MessagesState`) with reducers for list fields
- **Custom Reducer**: `items_reducer` handles both individual items and lists, supports clearing with `[]`
- **Message Context Splitting**: `split_messages_context()` divides history at last message with `additional_kwargs["artifact"]["final_response"] = True`
- **Conversation Formatting**: Always use `format_conversation_context()` to create `<HISTORICAL CONVERSATION>` and `<CONVERSATION>` blocks for prompts

## MCP Integration (External Tools)

**Setup**: This project integrates with a **custom MCP server** (separate project). Configure in `Configuration.mcp_servers` as `MCPServerConfig` with transport types (streamable_http, stdio, websocket, sse)

**Environment Config**: Set `MCP_API_SUPPORT_SERVER_URL` and `MCP_API_SUPPORT_SERVER_TRANSPORT` in `.env`

**Tool Execution Pattern** (see `generate_response` agent ~line 276):
```python
mcp_client = await initialize_mcp_client(configuration)
tools = await mcp_client.get_tools()
model_with_tools = model.bind_tools(tools)
# Manual tool loop - NOT using create_react_agent
for tool_call in response.tool_calls:
    tool_to_call = next((t for t in tools if t.name == tool_call["name"]), None)
    tool_result = await tool_to_call.ainvoke(tool_call["args"])
```

**Expected MCP Tools**: `readme_first` (capability overview), `retrieve_support_context` (documentation access)

**Important**: MCP server must be running before starting the chatbot. Connection failures will cause response agents to error.

## Configuration & Environment

- **Two-tier Models**: `azure_openai_deployment_name` (default/mini) vs `azure_hq_openai_deployment_name` (high-quality, used by assembler)
- **Model Helper**: Use `_get_azure_chat_model(configuration, hq_model=False)` - handles all Azure OpenAI setup
- **Structured Output**: Chain with `.with_structured_output(RequestDetails/ExtractedRequests/ResponseItem/AssembledResponse)`
- **Config Access**: `Configuration.from_runnable_config(config)` extracts from RunnableConfig in agent functions
- **Critical**: `from_runnable_config()` preserves checkpointer keys (`thread_id`, `checkpoint_ns`, `checkpoint_id`) - only extracts Configuration fields, leaves LangGraph keys untouched

## Testing & Development Workflow

**Setup**: 
- Copy `.env.example` to `.env` and configure Azure OpenAI + MCP server URLs
- Install: `pip install -e .` (or `pip install -e ".[dev]"` for dev tools)
- **Ensure custom MCP server is running** before testing

**Run Console Chatbot**: `python example.py` - Creates unique thread_id per session, runs in loop until "quit"
**Run LangServe Server**: `python serve.py` - Exposes at `/chatbot` with playground at `/chatbot/playground`
**Tests**: `pytest` - Fixtures in `tests/conftest.py` provide `mock_configuration`, `mock_mcp_client`, `sample_messages`

**Dev Tools**: black (formatting), ruff (linting), mypy (type checking)
**Test Client**: `python client_example.py` tests the LangServe server endpoints

## Project-Specific Conventions

**Logging**: Use `log_agent_action(agent_name, action, details_dict)` - currently prints to console, TODO: add proper logging module
**Error Handling**: Every agent wraps in try/except returning `Command(graph=END, update={"messages": [AIMessage(GENERIC_ERROR_MSG)]})` 
**IDs**: Request IDs are 8-char UUIDs from `generate_request_id()` 
**Prompts**: All system prompts in `src/api_support_chatbot/prompts.py` with scope categories (`API_SCOPE_CATEGORIES`, `API_OUT_OF_SCOPE_CATEGORIES`) and product definitions
**Message Metadata**: Final responses marked with `ai_message.additional_kwargs = {"artifact": {"final_response": True}}`

## LangGraph Server Deployment

**Entry Point**: `langgraph.json` points to `create_graph()` in `chatbot.py`
**Checkpointing**: Currently uses `InMemorySaver()` - TODO: add persistent checkpointing for production (e.g., PostgresSaver for Docker deployment)
**Graph Config**: Thread tracking via `{"configurable": {"thread_id": "..."}}`

**LangServe Integration** (`serve.py`):
- Uses `add_routes(app, graph, path="/chatbot", config_keys=["configurable"])`
- `config_keys` must allow checkpointer keys (thread_id, checkpoint_ns, checkpoint_id) to pass through
- `Configuration.from_runnable_config()` filters config dict - extracts only Configuration fields, preserves checkpointer keys

**Docker Deployment** (planned):
- Ensure custom MCP server is accessible from container (network configuration)
- Use environment variables for all configuration (Azure OpenAI, MCP URLs)
- Consider persistent checkpointing for conversation state across restarts
- See `serve.py` for LangServe FastAPI app structure

## Common Pitfalls

- Don't call `create_react_agent` - this project uses custom tool execution loop in response agents
- State updates must use exact field names from `ChatbotState` - typos silently fail
- When adding response items, reducer concatenates - explicitly pass `[]` to clear
- MCP client initialization is async - must `await initialize_mcp_client()`
- Product IDs are critical - coordinator checks `request_details.produtct_id` (note typo in field name)
- **Checkpointer config**: `from_runnable_config()` must NOT consume `thread_id`/`checkpoint_ns`/`checkpoint_id` - these are needed by LangGraph's checkpointer

## Key Files Reference

- `src/api_support_chatbot/chatbot.py` - All 4 agents + graph builder
- `src/api_support_chatbot/state.py` - State classes with Pydantic models
- `src/api_support_chatbot/configuration.py` - Config with `from_env()` and `from_runnable_config()`
- `src/api_support_chatbot/prompts.py` - System prompts with scope/product definitions
- `langgraph.json` - Graph configuration for LangGraph server
