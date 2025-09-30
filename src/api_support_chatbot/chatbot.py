"""Main chatbot implementation with LangGraph multi-agent architecture."""

import asyncio
from typing import Any, Dict, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
import json


from src.api_support_chatbot.configuration import Configuration
from src.api_support_chatbot.state import (
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
        cl_att = state.get("clarification_attempts", 0)
        max_att = state.get("max_clarification_attempts", 0)

        if ( not request_details.valid_request_received and
            #state.get("clarification_attempts", 0) < state.get("max_clarification_attempts", 5) ):
            cl_att < max_att ):

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
                    "clarification_attempts": cl_att + 1
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
                    "clarification_attempts": 0,
                    "request_items": [], # Reset previous requests
                    "response_items": [], # Reset previous responses
                    #TODO: Add history items here
                    "messages": [AIMessage(content="Working on your request...")],
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
) -> Command:
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
            item.id = generate_request_id()
            print(item)

        # Create Send commands for each request item
       
        if not request_items.item_list:
            return Command(
                graph=END,
                update={
                    "messages": [AIMessage(content="I apologize, but I wasn't able to identify specific requests from your message. Could you please provide more details or clarify your question?")],
                    },
                goto=END
                )
        log_agent_action(
        "ResponseCoordinator",
        "Delegating to response agents",
        {"count": len(request_items.item_list), "items": [f"{item.id}: {item.category}" for item in request_items.item_list]}
        )        
        # Fan out to response agents using Command
        return Command(
            update={"request_items": request_items.item_list},
        )
        
    except Exception as e:
        error_msg = create_error_message(e, "coordinate_response")
        log_agent_action("ResponseCoordinator", "Error occurred", {"error": error_msg})
        return Command(
            graph=END,
            update={"error_state": error_msg},
            goto=END
        )


async def fan_out_requests(state: ChatbotState) -> List[Send]:
    """Create Send commands to fan out to response agents."""
    request_items = state.get("request_items", [])
    if not request_items:
        raise ValueError("No request items to fan out")
    
    sends = []
    for item in request_items:
        sends.append(
            Send(
                "generate_response",
                {"request_item": item},
            )
        )        
    return sends

async def generate_response(
     data: Dict[str, Any], *, config: RunnableConfig
) -> Dict[str, ResponseItem]:
    """
    Agent 2.1: Response Agent
    Uses MCP tools to gather context and generate responses for individual request items.
    """
    request_item = data.get("request_item", None)
    if not request_item:
        raise ValueError("No request item provided to response agent")
    
    log_agent_action("ResponseAgent", f"Generating response for item {request_item.id}")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Initialize MCP client and get tools
        # TODO: Optimize to avoid re-initialization
        mcp_client = await initialize_mcp_client(configuration)
        tools = await mcp_client.get_tools()
        
        # Configure the model with tools
        azure_params = configuration.get_azure_openai_params()

        model = AzureChatOpenAI(
            model = azure_params["azure_deployment"],
            temperature = azure_params["temperature"],
            max_tokens = azure_params["max_tokens"],
            azure_endpoint = azure_params["azure_endpoint"],
            api_key = azure_params["api_key"],
            api_version = azure_params["api_version"]  
        )    
        model_with_tools = model.bind_tools(tools) #, tool_choice="any")
        
        system_prompt = format_response_agent_prompt()
        
        # Create response generation prompt
        response_prompt = f"""
        Request Text: {request_item.request_text}
        Product ID: {request_item.product_id}
        Request Category: {request_item.category}
        """
        
        # Initialize conversation messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=response_prompt)
        ]
        
        # Tool execution loop with maximum iterations
        # create_react_agent from langgraph.prebuilt can be used here instead
        max_iterations = 2
        iteration = 0
        final_response = None
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get model response with potential tool calls
            response = await model_with_tools.ainvoke(messages)
            
            
            # Check if there are tool calls to execute
            if hasattr(response, 'tool_calls') and response.tool_calls:
                messages.append(response)
                # Execute each tool call
                for tool_call in response.tool_calls:
                    try:
                        # Find the tool by name
                        tool_to_call = next((t for t in tools if t.name == tool_call["name"]), None)
                        if tool_to_call:
                            # Execute the tool
                            tool_result = await tool_to_call.ainvoke(tool_call["args"])
                            
                            # Add tool result to messages
                            tool_message = {
                                "role": "tool",
                                "content": str(tool_result),
                                "tool_call_id": tool_call.get("id", "")
                            }
                            messages.append(tool_message)
                        else:
                            # Tool not found
                            tool_message = {
                                "role": "tool", 
                                "content": f"Tool {tool_call['name']} not found",
                                "tool_call_id": tool_call.get("id", "")
                            }
                            messages.append(tool_message)
                    except Exception as e:
                        # Tool execution failed
                        tool_message = {
                            "role": "tool",
                            "content": f"Tool execution failed: {str(e)}",
                            "tool_call_id": tool_call.get("id", "")
                        }
                        messages.append(tool_message)
            else:
                # No more tool calls, break the loop
                final_response = response
                break
        # If we exit the loop without a final response, call the model one last time
        if not final_response:
            final_response = await model.ainvoke(messages)

        response_dict = json.loads(final_response.content)
        if not isinstance(response_dict, dict):
            response_dict = {"response_text": final_response.content, "response_found": False, "confidence": 0.0}
        # Create response item (could not use structured output here due to tool calls)
        response_item = ResponseItem(
            request_id = request_item.id,
            request_text = request_item.request_text,
            product_id = request_item.product_id,
            response_text = response_dict.get("response_text", "No response generated."),
            response_found = response_dict.get("response_found", False),
            confidence = response_dict.get("confidence", 0.0),
        )
        
        log_agent_action(
            "ResponseAgent",
            f"Generated response for item {request_item.id} after {iteration} iterations",
            {
                "Request Text": request_item.request_text[:100] + ("..." if len(request_item.request_text) > 100 else ""),
                "Response Text": response_item.response_content[:100] + ("..." if len(response_item.response_content) > 100 else "") if hasattr(response_item, 'response_content') else "No content",
                "Iterations": iteration
            }
        )
        
        return {"response_items": response_item}
        
    except Exception as e:
        error_msg = create_error_message(e, f"generate_response for item {request_item.id}")
        log_agent_action("ResponseAgent", "Error occurred", {"error": error_msg, "item_id": request_item.id})
        
        # Return error response item
        err_item = ResponseItem(
            request_id = request_item.id,
            response_text = f"I apologize, but I encountered an error while processing your request: {error_msg}",
            response_found = False,
        )
        return {"response_items": err_item}


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
        print("Stopping here...")
        exit(0)

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
        config_schema=Configuration
    )
    
    # Add nodes
    builder.add_node("get_request_details", get_request_details)
    builder.add_node("coordinate_response", coordinate_response)
    builder.add_node("generate_response", generate_response)
    builder.add_node("assemble_final_response", assemble_final_response, defer=True)
    
    # Add edges
    builder.add_edge(START, "get_request_details")
    builder.add_conditional_edges("coordinate_response", fan_out_requests)
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