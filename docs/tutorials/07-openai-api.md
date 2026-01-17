# OpenAI API Client

Use OpenAI SDK with llama-server for drop-in compatibility.

**Level**: Advanced | **Time**: 15 minutes | **VRAM Required**: 5-10 GB

---

## Setup

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)
```

## Chat Completions

```python
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Streaming

```python
stream = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/07-openai-api-client-llcuda-v2-2-0)
