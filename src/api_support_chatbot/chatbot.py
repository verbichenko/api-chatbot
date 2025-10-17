"""Main chatbot implementation with LangGraph multi-agent architecture."""

from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
import json


from src.api_support_chatbot.configuration import Configuration
from src.api_support_chatbot.state import (
    ChatbotState,
    RequestDetails,
    RequestItem,
    ExtractedRequests,
    ResponseItem,
    AssembledResponse,
)
from src.api_support_chatbot.prompts import (
    format_request_details_prompt,
    format_coordinator_prompt,
    format_response_agent_prompt,
    format_assembler_prompt,
    GENERIC_ERROR_MSG,
)
from src.api_support_chatbot.utils import (
    generate_request_id,
    log_agent_action,
    create_error_message,
)


async def initialize_mcp_client(config: Configuration) -> MultiServerMCPClient:
    """Initialize MCP client with configured servers."""
    mcp_connections = config.get_mcp_connections()
    return MultiServerMCPClient(mcp_connections)

def _get_azure_chat_model(configuration: Configuration, hq_model: bool = False) -> AzureChatOpenAI:
    """Helper function to create an AzureChatOpenAI instance from configuration."""
    if hq_model:
        deployment = configuration.azure_hq_openai_deployment_name
    else:
        deployment = configuration.azure_openai_deployment_name

    return AzureChatOpenAI(
        model = deployment,
        temperature = configuration.model_temperature,
        max_tokens = configuration.max_tokens,
        azure_endpoint = configuration.azure_openai_endpoint,
        api_key = configuration.azure_openai_api_key,
        api_version = configuration.azure_openai_api_version
    )

def split_messages_context(messages: List[BaseMessage]) -> tuple[List[BaseMessage], List[BaseMessage]]:
    """
    Split messages into historical and current context.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Tuple of (historical_messages, messages_to_clarify)
        - historical_messages: Messages from beginning to last final_response (inclusive)
        - messages_to_clarify: Messages after the last final_response
    """
    if not messages:
        return [], []
    
    # Find the last message with final_response metadata set to True
    last_final_response_index = -1
    for i in range(len(messages) - 1, -1, -1):
        message_metadata = messages[i].additional_kwargs.get("artifact", {})
        #message_metadata = getattr(messages[i], 'artifact', {})
        if message_metadata.get('final_response') == True:
            last_final_response_index = i
            break
    
    # If no final_response found, all messages are for clarification
    if last_final_response_index == -1:
        return [], messages
    
    # Split at the last final_response
    historical_messages = messages[:last_final_response_index + 1]
    current_messages = messages[last_final_response_index + 1:]
    
    return historical_messages, current_messages    

def messages_to_text(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a single text block."""
    return "\n\n".join([f"{msg.type}: \n{msg.content}" for msg in messages])

def format_conversation_context(messages: List[BaseMessage]) -> str:
    historical_conversation_msg, conversation_msg = split_messages_context(messages)  
    conversation = messages_to_text(conversation_msg)
    historical_conversation = messages_to_text(historical_conversation_msg)
    prompt = """
    <HISTORICAL CONVERSATION>
    {historical_conversation}
    </HISTORICAL CONVERSATION>

    <CONVERSATION>
    {conversation}
    </CONVERSATION>
    """
    return prompt.format(conversation=conversation, historical_conversation=historical_conversation)


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
        
        # Configure the model for structured output
        model = (
            _get_azure_chat_model(configuration, hq_model=False)
            .with_structured_output(RequestDetails)
            .with_config({
                "tags": ["get_request_details"]
            })
        )
        
        # Create system prompt
        system_prompt = format_request_details_prompt()

        # Split messages to isolate area that we are clairifying
        clarification_text = format_conversation_context(state["messages"])
        # Analyze the request
        messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=clarification_text)]
        
        request_details = await model.ainvoke(messages)
        
        log_agent_action(
            "GetRequestDetails", 
            "Completed conversation round",
            request_details.model_dump(mode="json")
        )
        attempts = state.get("clarification_attempts", 0)
        # Check if we have all necessary details to proceed or exceed max attempts
        if ( not request_details.valid_request_received and
            attempts < state.get("max_clarification_attempts", 3) ):

            # Condinue dialog to clarify the request
            response = ""
            if request_details.clarifying_question:
                response += str(request_details.clarifying_question)
                attempts += 1
            if not request_details.clarifying_question and request_details.info_message:
                response += str(request_details.info_message)
            # Continue clarification loop
            return Command(
                graph=END,
                update={
                    "messages": [AIMessage(content=response)],
                    "clarification_attempts": attempts
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
                    #TODO: Add streaming through AIMessageChunk
                    "messages": [AIMessage(content="Working on your request...")],
                },
                goto="coordinate_response"
            )
            
    except Exception as e:
        error_msg = create_error_message(e, "get_request_details")
        log_agent_action("GetRequestDetails", "Error occurred", {"error": error_msg})
        return Command(
            graph=END,
            update={"messages": [AIMessage(content = f"{GENERIC_ERROR_MSG} {error_msg} ")]},
            goto=END
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
        request_details = state.get("request_details")
        
        if not request_details:
            raise ValueError("No request details available for coordination")
        if not request_details.product_id:
            raise ValueError("No product specified.")
        
        # Configure the model for structured output
        model = (
            _get_azure_chat_model(configuration)
            .with_structured_output(ExtractedRequests)
        )
        # Create system prompt
        system_prompt = format_coordinator_prompt()

        conversation_text = format_conversation_context(state["messages"])

        messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=conversation_text)]
        # Generate request items
        request_items = await model.ainvoke(messages)
        # Add unique IDs to request items
        for item in request_items.item_list:
            item.id = generate_request_id()
            print(item)

        # Create Send commands for each request item
       
        if not request_items.item_list:
            raise ValueError("Unable to comprehend your request.")
        
        log_agent_action(
        "ResponseCoordinator",
        "Delegating to response agents",
        {"count": len(request_items.item_list), "items": [f"{item.id}: {item.category}" for item in request_items.item_list]}
        )        
        # Send list of items to fan out function
        return Command(
            update={"request_items": request_items.item_list},
        )
        
    except Exception as e:
        error_msg = create_error_message(e, "coordinate_response")
        log_agent_action("ResponseCoordinator", "Error occurred", {"error": error_msg})
        return Command(
            graph=END,
            update={"messages": [AIMessage(content = f"{GENERIC_ERROR_MSG} {error_msg} ")]},
            goto=END
        )


async def fan_out_requests(state: ChatbotState) -> List[Send]:
    """Create Send commands to fan out to response agents."""
    request_items = state.get("request_items", [])
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
    Uses MCP tools to gather context and generate responses for individual request item.
    """
    try:
        request_item = data.get("request_item", None)
        if not request_item:
            raise ValueError("No request item provided to response agent")
    
        log_agent_action("ResponseAgent", f"Generating response for item {request_item.id}")
    
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Initialize MCP client and get tools
        try:
            mcp_client = await initialize_mcp_client(configuration)
            tools = await mcp_client.get_tools()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MCP client or retrieve tools: {str(e)}")
        
        # Configure the model with tools
        model = _get_azure_chat_model(configuration)
        model_with_tools = model.bind_tools(tools)
        
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
                # No more tool calls, take the result and break the loop
                final_response = response
                break
        # If we exit the loop without a final response, call the model one last time without tools
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
            response_text = response_dict.get("response_text", "No response found."),
            response_found = response_dict.get("response_found", False),
            confidence = response_dict.get("confidence", 0.0),
        )
        
        log_agent_action(
            "ResponseAgent",
            f"Generated response for item {request_item.id} after {iteration} iterations",
            {
                "Request Text": request_item.request_text[:100] + ("..." if len(request_item.request_text) > 100 else ""),
                "Response Text": response_item.response_text[:100] + ("..." if len(response_item.response_text) > 100 else "") if hasattr(response_item, 'response_text') else "No content",
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
            response_text = f"{GENERIC_ERROR_MSG} {error_msg}",
            response_found = False,
            error = True
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
        qa_pairs = ""
        for item in response_items:
            # Check for error in individual response item
            if item.error:
                raise ValueError(f"{item.response_text}")
            
            response_text = item.response_text if item.response_found and item.response_text else "Could not answer the request"
            qa_pairs += """
            <REQUEST TEXT. PRODUCT ID={product_id}>
            {request_text}
            </REQUEST TEXT>

            <GENERATED RESPONCE. CONFIDENCE={confidence}>
            {response_text}
            </GENARTED RESPONCE>
            """.format(
              product_id = item.product_id,
              request_text = item.request_text,
              confidence = item.confidence,
              response_text = response_text
            )

        if not qa_pairs:
            raise ValueError("No valid response items to assemble.")

        # Configure the model for structured output
        model = (
            _get_azure_chat_model(configuration)
            .with_structured_output(AssembledResponse)
            .with_config({
                "tags": ["response_assembler"]
            })
        )
        
        # Create system prompt
        system_prompt = format_assembler_prompt()
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=qa_pairs)
        ]
        
        # Generate final response
        assembled_response = await model.ainvoke(messages)
        
        # Add sources to the response content
        
        log_agent_action(
            "ResponseAssembler",
            "Completed final response assembly",
            {
                "response_items_count": len(response_items),
                "Response Text": assembled_response.response_text[:100] + ("..." if len(assembled_response.response_text) > 100 else "") if hasattr(assembled_response, 'response_text') else "No content",
            }
        )
        ai_message = AIMessage(content = 
                f"{assembled_response.response_text} \n\n {assembled_response.follow_up_question}"
                )
        ai_message.additional_kwargs = {"artifact": {"final_response": True}}

        return Command(
            update={
                "assembled_response": assembled_response,
                "messages": [ai_message],
            }
        )
        
    except Exception as e:
        error_msg = create_error_message(e, "assemble_final_response")
        log_agent_action("ResponseAssembler", "Error occurred", {"error": error_msg})
        
        return Command(
            graph=END,
            update={
                "error_state": error_msg,
                "messages": [AIMessage(content = f"{GENERIC_ERROR_MSG} {error_msg}")]
            },
            goto=END
        )


class ChatbotInput(BaseModel):
    """Input schema for the chatbot graph."""
    messages: List[BaseMessage]

class ChatbotOutput(BaseModel):
    """Output schema for the chatbot graph."""
    messages: List[BaseMessage]

def create_chatbot_graph() -> StateGraph:
    """Create and configure the main chatbot graph."""
    
    # Create the main graph
    builder = StateGraph(
        state_schema=ChatbotState,
        input_schema=ChatbotInput,
        output_schema=ChatbotOutput
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