import os
import time
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
    timeout=300.0,
)

start = time.perf_counter()

try:
    stream = client.chat.completions.create(
        model="deepseek-ai/deepseek-v4-pro",
        messages=[{"role": "user", "content": "Rispondi solo con ok"}],
        temperature=0.0,
        max_tokens=20,
        stream=True,
    )

    print("Stream aperto, attendo chunk...")

    full_text = ""
    first_chunk_time = None

    for chunk in stream:
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start
            print(f"\nPrimo chunk dopo {first_chunk_time:.2f}s\n")

        if not getattr(chunk, "choices", None):
            continue

        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_text += delta

    print("\n\nFine stream.")
    print("Testo completo:", repr(full_text))
    print(f"Tempo totale: {time.perf_counter() - start:.2f}s")

except Exception as exc:
    print("Errore:")
    print(type(exc).__name__)
    print(exc)