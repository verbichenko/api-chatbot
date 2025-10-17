"""Example client for testing the LangServe API."""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langserve import RemoteRunnable
import uuid

# Load environment variables
load_dotenv()


async def run_client_example():
    """Run an example conversation with the remote chatbot."""
    
    # Connect to the remote chatbot
    # Make sure the server is running at this URL
    server_url = os.getenv("CHATBOT_SERVER_URL", "http://localhost:8000")
    chatbot = RemoteRunnable(f"{server_url}/chatbot")
    
    print("API Support Chatbot Client")
    print("=" * 40)
    print(f"Connected to: {server_url}")
    print("Type 'quit' to exit")
    print("=" * 40)
    
    # Create a thread ID for the conversation
    thread_id = str(uuid.uuid4())[:8]
    
    try:
        # Start conversation loop
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
            
            # Create configuration
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            
            print("\nAssistant: ", end="", flush=True)
            
            # Stream the response
            async for chunk in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            ):
                if "messages" in chunk and chunk["messages"]:
                    last_message = chunk["messages"][-1]
                    print(last_message.content, end="", flush=True)
            
            print()  # New line after response
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure the server is running and accessible.")


if __name__ == "__main__":
    asyncio.run(run_client_example())
