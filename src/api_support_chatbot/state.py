"""State definitions for the API Support Chatbot graph."""

import operator
from typing import Annotated, List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# Structured Output Models
class RequestDetails(BaseModel):
    """Structured representation of customer request details."""
    valid_request_received: bool = Field(
        default=False,
        description="Request is clear, complete and in scope. No further clarifications needed."
    )    
    clarifying_question: Optional[str] = Field(
        default="",
        description="A clarifying question to ask the customer if any details are missing."
    )
    dialog_message: Optional[str] = Field(
        default="",
        description="Put here the message to send to the customer other than the clarifying question."
    )
    produtct_id: Optional[str] = Field(
        default=None,
        description="The ID of the product ID the customer is inquiring about."
    )


class RequestItem(BaseModel):
    """Individual request item for delegation."""
    
    id: str = Field(description="Unique identifier for this request item")
    request_text: str = Field(description="Customer request extracted from the conversation")
    category: str = Field(description="Category of the request")
    product_id: Optional[str] = Field(
        default=None,
        description="The product ID relevant to this request item"
        )

class ExtractedRequests(BaseModel):
    """Collection of extracted requests."""
    item_list: List[RequestItem]

class ResponseItem(BaseModel):
    """Response for a single request item."""
    
    request_id: str = Field(description="ID of the request this response addresses")
    response_content: str = Field(description="The response content")
    sources: List[str] = Field(
        default_factory=list,
        description="List of sources used to generate the response"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in the response accuracy"
    )
    requires_followup: bool = Field(
        default=False,
        description="Whether this response requires follow-up"
    )


class FinalResponse(BaseModel):
    """Final assembled response for the customer."""
    
    response_text: str = Field(description="The final response text")
    sources: List[str] = Field(
        default_factory=list,
        description="All sources used in the response"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions for the customer"
    )
    satisfaction_score: float = Field(
        default=1.0,
        description="Expected customer satisfaction score"
    )


# State Definitions
class ChatbotInputState(MessagesState):
    """Input state for the chatbot - only contains messages."""
    pass


class ChatbotState(MessagesState):
    """Main chatbot state containing all conversation and processing data."""
    
    # Request processing
    request_details: Optional[RequestDetails] = None
    request_items: List[RequestItem] = Field(default_factory=list)
    response_items: List[ResponseItem] = Field(default_factory=list)
    final_response: Optional[FinalResponse] = None
    
    # MCP tool context
    mcp_context: Dict[str, Any] = Field(default_factory=dict)
    available_tools: List[str] = Field(default_factory=list)
    
    # Processing flags
    needs_clarification: bool = False
    processing_complete: bool = False
    error_state: Optional[str] = None
    
    # Metadata
    conversation_id: Optional[str] = None
    customer_id: Optional[str] = None
    session_metadata: Dict[str, Any] = Field(default_factory=dict)


class RequestDetailsState(MessagesState):
    """State for the Get Request Details agent."""
    
    request_details: Optional[RequestDetails] = None
    clarification_attempts: int = 0
    max_clarification_attempts: int = 3


class CoordinatorState(MessagesState):
    """State for the Response Coordinator agent."""
    
    request_details: Optional[RequestDetails] = None
    request_items: List[RequestItem] = Field(default_factory=list)
    delegation_complete: bool = False


class ResponseAgentState(MessagesState):
    """State for individual Response Agent instances."""
    
    request_item: Optional[RequestItem] = None
    response_item: Optional[ResponseItem] = None
    mcp_context: Dict[str, Any] = Field(default_factory=dict)
    tools_used: List[str] = Field(default_factory=list)


class AssemblerState(MessagesState):
    """State for the Response Assembler agent."""
    
    request_items: List[RequestItem] = Field(default_factory=list)
    response_items: List[ResponseItem] = Field(default_factory=list)
    final_response: Optional[FinalResponse] = None
    assembly_complete: bool = False


# Output States for subgraphs
class RequestDetailsOutputState(BaseModel):
    """Output state from Request Details agent."""
    
    request_details: RequestDetails
    needs_clarification: bool = False


class CoordinatorOutputState(BaseModel):
    """Output state from Response Coordinator agent."""
    
    request_items: List[RequestItem]


class ResponseAgentOutputState(BaseModel):
    """Output state from Response Agent."""
    
    response_item: ResponseItem


class AssemblerOutputState(BaseModel):
    """Output state from Response Assembler agent."""
    
    final_response: FinalResponse