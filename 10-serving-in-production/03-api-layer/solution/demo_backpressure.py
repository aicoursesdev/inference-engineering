"""Ten clients at once against a gateway that allows 2 in flight and 4 waiting."""
import asyncio
import time

import httpx

BODY = {"model": "m", "max_tokens": 150, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Describe the monsoon in 100 words."}]}


async def one(client, i, t0):
    r = await client.post("http://127.0.0.1:8090/v1/chat/completions", json=BODY)
    return i, r.status_code, time.monotonic() - t0, r.headers.get("retry-after")


async def main():
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=60) as client:
        for i, code, t, ra in sorted(await asyncio.gather(*(one(client, i, t0) for i in range(10))), key=lambda x: x[2]):
            print(f"client {i}: {code} after {t:4.1f} s" + (f" (Retry-After: {ra})" if ra else ""))

asyncio.run(main())
