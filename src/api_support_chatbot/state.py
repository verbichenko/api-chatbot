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
        description="Use if you have received an intelligible input and understood it but need more details to proceed."
    )
    info_message: Optional[str] = Field(
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
    
    request_id: Optional[str] = Field(description="ID of the request this response addresses", default=None)
    request_text: Optional[str] = Field(description="Leave this blank.", default="")
    response_text: Optional[str] = Field(description="Text of the response to the request", default="No response found.")
    response_found: bool = Field(
        default=False,
        description="Whether a valid response was found for the request"
    )
    product_id: Optional[str] = Field(
        default=None,
        description="The product ID relevant to this request item"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in the response accuracy"
    )
    error: bool = Field(
        default=False,
        description="Whether there was an error processing this request"
    )


class AssembledResponse(BaseModel):
    """Final assembled response for the customer."""
    
    response_text: str = Field(description="The final assembled response text")
    
    follow_up_question: Optional[str] = Field(
        default=None,
        description="Suggested follow-up question for the customer"
    )


def items_reducer(current_value, new_value):
    """Reducer function that handles both individual items and lists, and allows clearing."""
    if isinstance(new_value, list) and new_value == []:
        # Clear the list
        return []
    elif isinstance(new_value, list):
        # Add list to current list
        return operator.add(current_value, new_value)
    else:
        # Add individual item to current list
        return current_value + [new_value]


class ChatbotState(MessagesState):
    """Main chatbot state containing all conversation and processing data."""

    clarification_attempts: int
    max_clarification_attempts: int

    request_details: Optional[RequestDetails] = None
    request_items: Annotated[list[RequestItem], items_reducer] = []
    response_items: Annotated[list[ResponseItem], items_reducer] = []
    assembled_response: Optional[AssembledResponse] = None
