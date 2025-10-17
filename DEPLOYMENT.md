# Deployment Guide

This guide explains how to deploy and run the API Support Chatbot in production using LangServe.

## Important: Vendored Dependencies

This project includes a **patched version of langserve** in `src/api_support_chatbot/vendor/langserve_patched/`.

**Why?** The patch allows LangGraph's checkpointer configuration keys (`thread_id`, `checkpoint_ns`, `checkpoint_id`) to pass through without being filtered by Pydantic validation. This is critical for maintaining conversation state across requests.

The vendored package is automatically used when you install the project with `pip install -e .` since it's part of the source tree. No additional configuration is needed.

For details on the patches applied, see `src/api_support_chatbot/vendor/README.md`.

## Prerequisites

1. Python 3.11 or higher
2. All required environment variables configured (see `.env` file)
3. Dependencies installed

## Installation

First, install the required dependencies including LangServe:

```bash
pip install -e .
```

Or install from requirements:

```bash
pip install langserve[all]>=0.3.0 sse-starlette>=2.0.0
```

## Running the Production Server

### Option 1: Using the serve.py script directly

```bash
python serve.py
```

The server will start on `http://0.0.0.0:8000` by default.

### Option 2: Using Uvicorn directly

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Option 3: With custom host and port

Set environment variables:

```bash
export HOST=0.0.0.0
export PORT=8080
python serve.py
```

Or use command line arguments with uvicorn:

```bash
uvicorn serve:app --host 127.0.0.1 --port 8080
```

## API Endpoints

Once the server is running, the following endpoints are available:

### Main Endpoints

- **`/`** - Root endpoint with API information
- **`/health`** - Health check endpoint
- **`/docs`** - Interactive API documentation (Swagger UI)
- **`/redoc`** - Alternative API documentation (ReDoc)

### Chatbot Endpoints

The chatbot is exposed at `/chatbot` with the following sub-endpoints:

- **`POST /chatbot/invoke`** - Invoke the chatbot with a single input
- **`POST /chatbot/batch`** - Batch invoke with multiple inputs
- **`POST /chatbot/stream`** - Stream responses from the chatbot
- **`POST /chatbot/stream_log`** - Stream detailed logs

### Playground

LangServe automatically provides a playground UI at:
- **`/chatbot/playground`** - Interactive chatbot playground

## Example Usage

### Using the Python Client

Run the example client:

```bash
python client_example.py
```

### Using cURL

Invoke endpoint:

```bash
curl -X POST "http://localhost:8000/chatbot/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"type": "human", "content": "How do I authenticate with the API?"}
      ]
    },
    "config": {
      "configurable": {
        "thread_id": "test-123"
      }
    }
  }'
```

Stream endpoint:

```bash
curl -X POST "http://localhost:8000/chatbot/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"type": "human", "content": "What endpoints are available?"}
      ]
    },
    "config": {
      "configurable": {
        "thread_id": "test-123"
      }
    }
  }'
```

### Using the LangServe Client (Python)

```python
from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage

# Connect to the remote chatbot
chatbot = RemoteRunnable("http://localhost:8000/chatbot")

# Invoke
response = chatbot.invoke({
    "messages": [HumanMessage(content="Hello!")]
}, config={"configurable": {"thread_id": "test-123"}})

print(response)

# Stream
for chunk in chatbot.stream({
    "messages": [HumanMessage(content="Tell me about the API")]
}, config={"configurable": {"thread_id": "test-123"}}):
    print(chunk)
```

### Using the LangChain.js Client

```javascript
import { RemoteRunnable } from "@langchain/core/runnables/remote";

const chatbot = new RemoteRunnable({
  url: "http://localhost:8000/chatbot"
});

const response = await chatbot.invoke({
  messages: [{ type: "human", content: "Hello!" }]
}, {
  configurable: { thread_id: "test-123" }
});

console.log(response);
```

## Production Deployment

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY serve.py .
COPY .env .

RUN pip install -e .

EXPOSE 8000

CMD ["python", "serve.py"]
```

Build and run:

```bash
docker build -t api-chatbot .
docker run -p 8000:8000 api-chatbot
```

### Using Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - HOST=0.0.0.0
      - PORT=8000
    restart: unless-stopped
```

Run:

```bash
docker-compose up -d
```

### Using Systemd (Linux)

Create a systemd service file `/etc/systemd/system/api-chatbot.service`:

```ini
[Unit]
Description=API Support Chatbot
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/api-chatbot
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python /path/to/api-chatbot/serve.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable api-chatbot
sudo systemctl start api-chatbot
```

### Using Supervisor

Create a supervisor config `/etc/supervisor/conf.d/api-chatbot.conf`:

```ini
[program:api-chatbot]
command=/path/to/venv/bin/python /path/to/api-chatbot/serve.py
directory=/path/to/api-chatbot
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/api-chatbot.err.log
stdout_logfile=/var/log/api-chatbot.out.log
```

Reload supervisor:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start api-chatbot
```

### Behind a Reverse Proxy (Nginx)

Nginx configuration:

```nginx
server {
    listen 80;
    server_name chatbot.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for streaming
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## Configuration

### Environment Variables

Required environment variables (set in `.env` file):

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment
AZURE_HQ_OPENAI_DEPLOYMENT_NAME=your-hq-deployment

# MCP Configuration
MCP_SERVER_FILESYSTEM_PATH=/path/to/filesystem
MCP_SERVER_PUPPETEER_PATH=/path/to/puppeteer

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8000
```

### CORS Configuration

By default, CORS is enabled for all origins. For production, update the CORS settings in `serve.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to your domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Monitoring and Logging

### Health Checks

The `/health` endpoint can be used for monitoring:

```bash
curl http://localhost:8000/health
```

### Logging

Uvicorn logs are output to stdout by default. For production, configure logging:

```bash
uvicorn serve:app --log-config logging.conf
```

### Metrics

Consider adding metrics with:
- Prometheus with `prometheus-fastapi-instrumentator`
- DataDog with `ddtrace`
- Application Insights for Azure

## Security Considerations

1. **API Keys**: Never commit `.env` files. Use environment variables or secret management services.
2. **CORS**: Restrict allowed origins in production.
3. **Rate Limiting**: Consider adding rate limiting with `slowapi` or similar.
4. **Authentication**: Add authentication middleware if needed.
5. **HTTPS**: Always use HTTPS in production with SSL/TLS certificates.

## Troubleshooting

### Server won't start

1. Check that all environment variables are set
2. Verify the port is not already in use
3. Check firewall settings

### LangServe import errors

Make sure langserve is installed:

```bash
pip install langserve[all]
```

### Streaming issues

Ensure your reverse proxy (if any) supports HTTP/1.1 upgrade and has appropriate timeout settings.

## Performance Tuning

For production use, consider:

1. **Workers**: Run multiple Uvicorn workers
   ```bash
   uvicorn serve:app --workers 4
   ```

2. **Async Workers**: Use Gunicorn with Uvicorn workers
   ```bash
   gunicorn serve:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

3. **Caching**: Implement response caching for common queries

4. **Load Balancing**: Use a load balancer for horizontal scaling

## Resources

- [LangServe Documentation](https://python.langchain.com/docs/langserve)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
