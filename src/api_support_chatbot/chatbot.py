"""Main chatbot implementation with LangGraph multi-agent architecture."""

import asyncio
from typing import Any, Dict, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from src.api_support_chatbot.configuration import Configuration
from src.api_support_chatbot.state import (
    ChatbotInputState,
    ChatbotState,
    RequestDetails,
    RequestItem,
    ExtractedRequests,
    ResponseItem,
    FinalResponse,
    RequestDetailsOutputState,
    CoordinatorOutputState,
    ResponseAgentOutputState,
    AssemblerOutputState,
)
from src.api_support_chatbot.prompts import (
    format_request_details_prompt,
    format_coordinator_prompt,
    format_response_agent_prompt,
    format_assembler_prompt,
    format_greeting_message,
)
from src.api_support_chatbot.utils import (
    get_today_str,
    generate_request_id,
    generate_conversation_id,
    extract_last_human_message,
    log_agent_action,
    create_error_message,
    calculate_confidence_score,
)


# Initialize configurable model
configurable_model = init_chat_model(model_provider="azure_openai",
    configurable_fields=("model", "temperature", "max_tokens", "azure_endpoint", "api_key", "api_version", "azure_deployment"),
)


async def initialize_mcp_client(config: Configuration) -> MultiServerMCPClient:
    """Initialize MCP client with configured servers."""
    mcp_connections = config.get_mcp_connections()
    return MultiServerMCPClient(mcp_connections)


async def get_request_details(
    state: ChatbotState, config: RunnableConfig
) -> Command:
    """
    Agent 1: Get Request Details
    Chat with customer, determine customer problem, check id in scope, and requests clarifications if needed.
    """
    #log_agent_action("GetRequestDetails", "Starting request analysis")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Extract the latest customer message
        last_message = extract_last_human_message(state["messages"])
        # We are at the beginning of a new conversation
        if not last_message:
            return Command(
                graph=END,
                update={
                    "messages": [AIMessage(content=format_greeting_message())],
                    },
                goto=END
                )
        
        # Configure the model for structured output
        azure_params = configuration.get_azure_openai_params()
        model = (
            configurable_model
            .with_structured_output(RequestDetails)
            .with_config({
                "configurable": azure_params,
                "tags": ["get_request_details"]
            })
        )
        
        # Create system prompt
        system_prompt = format_request_details_prompt()
        
        # Analyze the request
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        request_details = await model.ainvoke(messages)
        
        log_agent_action(
            "GetRequestDetails", 
            "Completed conversaton round",
            request_details.model_dump(mode="json")
        )
        
        # Determine next step
        if ( not request_details.valid_request_received and 
            state.get("clarification_attempts", 0) < 5):
            # Condinue dialog to clarify the request
            response = ""
            if request_details.clarifying_question:
                response += str(request_details.clarifying_question)
            if not request_details.clarifying_question and request_details.dialog_message:
                response += str(request_details.dialog_message)
            return Command(
                graph=END,
                update={
                    "messages": [AIMessage(content=response)],
                    "clarification_attempts": state.get("clarification_attempts", 0) + 1
                    },
                goto=END
                )
        else:
            # Valid request received or max clarifications reached,
            # proceed to response coordination

            return Command(
                graph="coordinate_response",
                update={
                    "request_details": request_details,
                    "needs_clarification": False
                },
                goto="coordinate_response"
            )
            
    except Exception as e:
        error_msg = create_error_message(e, "get_request_details")
        log_agent_action("GetRequestDetails", "Error occurred", {"error": error_msg})
        return Command(
            graph=END,
            update={"error_state": error_msg}
        )


async def coordinate_response(
    state: ChatbotState, config: RunnableConfig
) -> List[Send]:
    """
    Agent 2: Response Coordinator
    Breaks down the request into specific items and delegates to response agents.
    """
    log_agent_action("ResponseCoordinator", "Starting request coordination")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        # Delete ?
        request_details = state.get("request_details")
        
        if not request_details:
            raise ValueError("No request details available for coordination")
        
        # Configure the model for structured output
        azure_params = configuration.get_azure_openai_params()
        model = (
            configurable_model
            .with_structured_output(ExtractedRequests)
            .with_config({
                "configurable": azure_params,
                "tags": ["response_coordinator"]
            })
        )
        
        # Create system prompt
        system_prompt = format_coordinator_prompt()
        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        # Generate request items
        request_items = await model.ainvoke(messages)
        
        # Add unique IDs to request items
        for item in request_items.item_list:
            if not item.id:
                item.id = generate_request_id()
            print(item)
        
        log_agent_action(
            "ResponseCoordinator",
            "Created request items",
            {"count": len(request_items), "items": [f"{item.id}: {item.category}" for item in request_items]}
        )

        print("Stopping here...")
        exit(0)


        # Create Send commands for each request item
        sends = []
        for item in request_items:
            sends.append(
                Send(
                    "generate_response",
                    {
                        "request_item": item,
                        "mcp_context": state.get("mcp_context", {})
                    }
                )
            )
        
        return sends
        
    except Exception as e:
        error_msg = create_error_message(e, "coordinate_response")
        log_agent_action("ResponseCoordinator", "Error occurred", {"error": error_msg})
        return [Send(END, {"error_state": error_msg})]


async def generate_response(
    data: Dict[str, Any], config: RunnableConfig
) -> ResponseItem:
    """
    Agent 2.1: Response Agent
    Uses MCP tools to gather context and generate responses for individual request items.
    """
    request_item = data.get("request_item")
    if not request_item:
        raise ValueError("No request item provided to response agent")
    
    log_agent_action("ResponseAgent", f"Generating response for item {request_item.id}")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Initialize MCP client and get tools
        mcp_client = await initialize_mcp_client(configuration)
        tools = await mcp_client.get_tools()
        
        # Configure the model with tools
        azure_params = configuration.get_azure_openai_params()
        model = (
            configurable_model
            .bind_tools(tools)
            .with_config({
                "configurable": azure_params,
                "tags": ["response_agent", f"item_{request_item.id}"]
            })
        )
        
        # Create system prompt with MCP context
        mcp_context = data.get("mcp_context", {})
        system_prompt = format_response_agent_prompt(get_today_str(), mcp_context)
        
        # Create response generation message
        response_prompt = f"""
        Request Item ID: {request_item.id}
        Category: {request_item.category}
        Priority: {request_item.priority}
        Request: {request_item.request_text}
        Context Requirements: {', '.join(request_item.context_requirements)}
        
        Please use the available tools to gather relevant context and generate a comprehensive response.
        Start by using the readme_first resource to understand capabilities, then use retrieve_support_context as needed.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=response_prompt)
        ]
        
        # Generate initial response with tools
        response = await model.ainvoke(messages)
        messages.append(response)
        
        # Extract response content and sources
        response_content = response.content if hasattr(response, 'content') else str(response)
        sources = []
        tools_used = []
        
        # Process tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tools_used.append(tool_call["name"])
                # In a real implementation, you would execute the tools here
                # and add the results to the sources
        
        # Calculate confidence based on available information
        confidence_factors = {
            "context_availability": 0.8 if tools_used else 0.5,
            "request_clarity": min(1.0, len(request_item.request_text) / 50),
            "category_match": 0.9 if request_item.category in ["authentication", "endpoints", "errors"] else 0.7
        }
        confidence = calculate_confidence_score(confidence_factors)
        
        response_item = ResponseItem(
            request_id=request_item.id,
            response_content=response_content,
            sources=sources,
            confidence=confidence,
            requires_followup=confidence < 0.7
        )
        
        log_agent_action(
            "ResponseAgent",
            f"Generated response for item {request_item.id}",
            {
                "confidence": confidence,
                "tools_used": tools_used,
                "requires_followup": response_item.requires_followup
            }
        )
        
        return response_item
        
    except Exception as e:
        error_msg = create_error_message(e, f"generate_response for item {request_item.id}")
        log_agent_action("ResponseAgent", "Error occurred", {"error": error_msg, "item_id": request_item.id})
        
        # Return error response item
        return ResponseItem(
            request_id=request_item.id,
            response_content=f"I apologize, but I encountered an error while processing your request: {error_msg}",
            sources=[],
            confidence=0.0,
            requires_followup=True
        )


async def assemble_final_response(
    state: ChatbotState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """
    Agent 3: Response Assembler
    Combines all response items into a coherent final response.
    """
    log_agent_action("ResponseAssembler", "Starting final response assembly")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        response_items = state.get("response_items", [])
        
        if not response_items:
            error_msg = "No response items available for assembly"
            return Command(
                graph=END,
                update={
                    "error_state": error_msg,
                    "messages": [AIMessage(content="I apologize, but I wasn't able to generate a response to your request. Please try again or rephrase your question.")]
                }
            )
        
        # Configure the model for structured output
        azure_params = configuration.get_azure_openai_params()
        model = (
            configurable_model
            .with_structured_output(FinalResponse)
            .with_config({
                "configurable": azure_params,
                "tags": ["response_assembler"]
            })
        )
        
        # Create system prompt
        system_prompt = format_assembler_prompt(get_today_str())
        
        # Prepare assembly context
        response_summaries = []
        all_sources = []
        
        for item in response_items:
            response_summaries.append(f"Item {item.request_id}: {item.response_content}")
            all_sources.extend(item.sources)
        
        assembly_prompt = f"""
        Customer request: {extract_last_human_message(state["messages"])}
        
        Response items to assemble:
        {"\n".join(response_summaries)}
        
        Please create a comprehensive, well-structured final response that addresses all aspects of the customer's request.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=assembly_prompt)
        ]
        
        # Generate final response
        final_response = await model.ainvoke(messages)
        
        # Add sources to the response content
        if all_sources:
            final_response.response_text += f"\n\n**Sources:**\n" + "\n".join(f"- {source}" for source in set(all_sources))
        
        log_agent_action(
            "ResponseAssembler",
            "Completed final response assembly",
            {
                "response_items_count": len(response_items),
                "sources_count": len(set(all_sources)),
                "satisfaction_score": final_response.satisfaction_score
            }
        )
        
        return Command(
            graph=END,
            update={
                "final_response": final_response,
                "processing_complete": True,
                "messages": [AIMessage(content=final_response.response_text)]
            }
        )
        
    except Exception as e:
        error_msg = create_error_message(e, "assemble_final_response")
        log_agent_action("ResponseAssembler", "Error occurred", {"error": error_msg})
        
        # Provide fallback response
        fallback_content = "I apologize, but I encountered an issue while preparing your response. "
        if state.get("response_items"):
            # Try to provide basic concatenation as fallback
            basic_responses = [item.response_content for item in state["response_items"]]
            fallback_content += "Here's the information I was able to gather:\n\n" + "\n\n".join(basic_responses)
        else:
            fallback_content += "Please try rephrasing your question or contact support directly."
        
        return Command(
            graph=END,
            update={
                "error_state": error_msg,
                "messages": [AIMessage(content=fallback_content)]
            }
        )


def create_chatbot_graph() -> StateGraph:
    """Create and configure the main chatbot graph."""
    
    # Create the main graph
    builder = StateGraph(
        ChatbotState,
        input=ChatbotInputState,
        config_schema=Configuration
    )
    
    # Add nodes
    builder.add_node("get_request_details", get_request_details)
    builder.add_node("coordinate_response", coordinate_response)
    builder.add_node("generate_response", generate_response)
    builder.add_node("assemble_final_response", assemble_final_response)
    
    # Add edges
    builder.add_edge(START, "get_request_details")
    builder.add_edge("coordinate_response", "generate_response")
    builder.add_edge("generate_response", "assemble_final_response")
    builder.add_edge("assemble_final_response", END)
    
    # TODO add persistent checkpointing
    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)


# Graph instance for LangGraph server
graph = create_chatbot_graph()


def create_graph() -> StateGraph:
    """Entry point for LangGraph server configuration."""
    return graph