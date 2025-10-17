#!/usr/bin/env python
"""Simple test script to verify the LangServe server is working."""

import asyncio
import sys
from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage
import uuid



async def test_server(server_url: str = "http://localhost:8000"):
    """Test the LangServe server."""
    
    print(f"Testing server at: {server_url}")
    print("-" * 50)
    
    try:
        # Test 1: Health check
        print("\n1. Testing health endpoint...")
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_url}/health")
            if response.status_code == 200:
                print("   ✓ Health check passed:", response.json())
            else:
                print(f"   ✗ Health check failed: {response.status_code}")
                return False
        
        # Test 2: Connect to chatbot
        print("\n2. Testing chatbot connection...")
        chatbot = RemoteRunnable(f"{server_url}/chatbot")
        print("   ✓ Connected to chatbot")
        
        # Test 3: Simple invoke
        print("\n3. Testing invoke endpoint...")
        config = {"configurable": {}}
        config["configurable"]["thread_id"] = str(uuid.uuid4())[:8]
        response = await chatbot.ainvoke(
            {"messages": [HumanMessage(content="Hello, can you help me?")]},
            config = {"configurable": {"thread_id": "test-456"}}
        )
        
        if "messages" in response and response["messages"]:
            last_message = response["messages"][-1]
            print(f"   ✓ Received response: {last_message.content[:100]}...")
        else:
            print("   ✗ No response received")
            return False
        
        # Test 4: Streaming
        print("\n4. Testing stream endpoint...")
        chunks = []
        async for chunk in chatbot.astream(
            {"messages": [HumanMessage(content="What can you do?")]},
            {"configurable": {"thread_id": "test-456"}}
        ):
            chunks.append(chunk)
        
        if chunks:
            print(f"   ✓ Received {len(chunks)} chunks")
        else:
            print("   ✗ No chunks received")
            return False
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        print("\nMake sure the server is running:")
        print("  python serve.py")
        return False


if __name__ == "__main__":
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    result = asyncio.run(test_server(server_url))
    sys.exit(0 if result else 1)
