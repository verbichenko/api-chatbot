"""Example usage of the API Support Chatbot."""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.api_support_chatbot.chatbot import create_chatbot_graph
from src.api_support_chatbot.configuration import Configuration
import uuid



async def run_example():
    """Run an example conversation with the chatbot."""
    
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = Configuration.from_env()
    
    # Create the chatbot graph
    graph = create_chatbot_graph()
    
    # Example messages
    examples = [
        "How do I authenticate with your API using OAuth2?",
    ]
    
    print("API Support Chatbot Example")
    print("=" * 40)
    
    for i, question in enumerate(examples, 1):
        
        try:
            # Run the chatbot thread_id
            graph_config = {"configurable": config.model_dump()}
            graph_config["configurable"]["thread_id"] = str(uuid.uuid4())[:8]
            #Show a greeting message
            result = await graph.ainvoke({"messages": []},
                    config=graph_config
            )
            print(result["messages"][-1].content)
            while True:
                user_input = input("You: ")
                if "quit" in user_input:
                    break
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