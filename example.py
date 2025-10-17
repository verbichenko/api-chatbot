"""Simle console version of the API Support Chatbot for testing purposes."""

import asyncio
import os
from langchain_core.messages import HumanMessage
from src.api_support_chatbot.chatbot import create_chatbot_graph
from src.api_support_chatbot.prompts import GREETING_MESSAGE
import uuid



async def run_example():
    """Run an example conversation with the chatbot."""
    
    
    # Create the chatbot graph
    graph = create_chatbot_graph()
    
    print("API Support Chatbot Example")
    print("=" * 40)
        
    try:
        # Run the chatbot thread_id
        graph_config = {"configurable": {}}
        graph_config["configurable"]["thread_id"] = str(uuid.uuid4())[:8]
        #Show a greeting message
        print(GREETING_MESSAGE)
        # Start conversation loop
        while True:
            user_input = input("You: ")
            if "quit" in user_input:
                break
            # TODO Add streaming: for chunk, meta in app.stream({"messages": [prompt]}, stream_mode="messages"):
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=graph_config
            )
            print(result["messages"][-1].content)
            
    except Exception as e:
        print(f"   [Error: {str(e)}]")
        
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(run_example())