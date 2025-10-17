# Vendored Dependencies

This directory contains patched versions of third-party packages that are included with the project for redistribution.

## langserve_patched (v0.3.2)

**Original Package**: `langserve==0.3.2`  
**License**: MIT (see langserve_patched/LICENSE or original package)

### Patches Applied

#### 1. Config Validation Bypass in `api_handler.py`

**File**: `langserve_patched/api_handler.py`  
**Lines**: ~193-196  
**Reason**: Allow checkpointer configuration keys (`thread_id`, `checkpoint_ns`, `checkpoint_id`) to pass through without Pydantic validation

**Original Code**:
```python
elif isinstance(config, Mapping):
    config_dicts.append(model(**config).model_dump())
```

**Patched Code**:
```python
elif isinstance(config, Mapping):
    #PATCHED: Removed pydantic validation to allow passing checkpointer keys
    #config_dicts.append(model(**config).model_dump())
    config_dicts.append(dict(config))
```

**Impact**: This allows LangGraph's checkpointer keys to pass through the configuration without being filtered out by Pydantic validation. This is essential for the multi-agent chatbot to maintain conversation state across requests.

### Usage

Import from the vendored package instead of the original:

```python
# Instead of:
# from langserve import add_routes

# Use:
from api_support_chatbot.vendor.langserve_patched import add_routes
```

### Updating

If you need to update to a newer version of langserve:

1. Install the new version: `pip install langserve==<version>`
2. Copy to vendor: `cp -r .venv/lib/python3.13/site-packages/langserve src/api_support_chatbot/vendor/langserve_patched`
3. Remove cache: `find src/api_support_chatbot/vendor/langserve_patched -type d -name "__pycache__" -exec rm -rf {} +`
4. Re-apply patches as documented above
5. Test thoroughly
6. Update version number in this README
