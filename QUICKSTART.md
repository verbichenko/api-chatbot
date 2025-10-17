# Quick Start Guide - LangServe Production Deployment

## 1. Install Dependencies

First, install the langserve package along with other dependencies:

```bash
pip install langserve[all] sse-starlette
```

Or reinstall the project with updated dependencies:

```bash
pip install -e .
```

## 2. Ensure Environment Variables are Set

Make sure your `.env` file contains all required configuration:

```bash
# Required Azure OpenAI settings
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
AZURE_HQ_OPENAI_DEPLOYMENT_NAME=...

# Optional server settings
HOST=0.0.0.0
PORT=8000
```

## 3. Start the Server

Simply run:

```bash
python serve.py
```

You should see output like:

```
Starting API Support Chatbot server on 0.0.0.0:8000
API documentation available at http://0.0.0.0:8000/docs
Chatbot endpoint available at http://0.0.0.0:8000/chatbot
```

## 4. Test the Server

### Option A: Use the Web Playground

Open your browser and go to:
```
http://localhost:8000/chatbot/playground
```

### Option B: Use the Python Client

In a new terminal, run:

```bash
python client_example.py
```

### Option C: Check the API Docs

Visit the interactive API documentation:
```
http://localhost:8000/docs
```

## 5. Make a Test Request

Using cURL:

```bash
curl -X POST "http://localhost:8000/chatbot/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{"type": "human", "content": "Hello!"}]
    },
    "config": {
      "configurable": {"thread_id": "test-123"}
    }
  }'
```

## Next Steps

- See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed production deployment options
- Configure CORS settings for your domain in `serve.py`
- Set up monitoring and logging
- Consider using Docker or systemd for production deployment
