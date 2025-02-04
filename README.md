# pydantic-ai-langfuse

pydantic-ai-langfuse extends [pydantic-ai-slim](https://pypi.org/project/pydantic-ai-slim/) to integrate Langfuse tracking into your OpenAI model interactions. By incorporating our Langfuse OpenAI model settings, you can easily label, track, and filter your generations using enriched metadata.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install pydantic-ai-langfuse
```

## Environment Setup

Before running your model, you need to set up the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `LANGFUSE_PUBLIC_KEY`: Your Langfuse public key.
- `LANGFUSE_SECRET_KEY`: Your Langfuse secret key.
- `LANGFUSE_HOST`: Your Langfuse host endpoint.

## Quickstart

Below is a complete Python example showing how to set up and use the LangfuseOpenAIModel with extra model settings. This example uses the synchronous `run_sync` method with basic error handling and retries built in.

```python
import os

from langfuse.openai import AsyncOpenAI
from pydantic_ai import Agent

from pydantic_ai_langfuse import LangfuseOpenAIModel

for var in [
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
]:
    if var not in os.environ:
        raise OSError(f"Missing env variable: {var}")

weather_agent = Agent(
    model=LangfuseOpenAIModel("gpt-4o", openai_client=AsyncOpenAI()),
    system_prompt="Be concise: reply with one sentence.",
    retries=2,
)

result = weather_agent.run_sync(
    "What the weather like in Medolago BG?",
    model_settings={
        "name": "weather_query",
        "metadata": {"location": "medolago", "query_type": "weather"},
        "session_id": "testoneditest",
        "user_id": "user123",
        "tags": ["weather", "italy", "query"],
    },
)

print("Response:", result.data)

```
