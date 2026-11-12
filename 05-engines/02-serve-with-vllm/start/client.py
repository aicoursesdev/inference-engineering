"""Call any OpenAI-compatible server: llama-server, vLLM, SGLang, Ollama.

    uv run --with openai python client.py http://127.0.0.1:8080/v1 MODEL
"""
import sys

from openai import OpenAI

base_url, model = sys.argv[1], sys.argv[2]
client = OpenAI(base_url=base_url, api_key="not-needed-locally")
reply = client.chat.completions.create(
    model=model,
    temperature=0,
    max_tokens=60,
    messages=[{"role": "user",
               "content": "In one sentence: why does my Kochi rain jacket "
                          "leak at the shoulder seams? /no_think"}],
)
print(reply.choices[0].message.content)
print(reply.usage)
