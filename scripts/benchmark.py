"""
Latency benchmark for the local model.

Measures, over a few prompts:
  * TTFT  — time to first token (how long until the reply starts appearing).
  * Total — wall-clock time for the whole reply.
  * tok/s — generated tokens per second (from Ollama's own eval_count/eval_duration).

Run it AFTER `ollama serve` is up and the model is pulled:

    python scripts/benchmark.py

Copy the printed table into your README under "Latency benchmarks". Do NOT make
numbers up — the whole point is to report your own hardware's real performance.
"""
import json
import statistics
import time

import httpx

from app.config import settings

PROMPTS = [
    "Hi, what membership plans do you offer?",
    "Can I freeze my membership for two months?",
    "What time is the Saturday strength class and can I book it?",
]


def run_one(prompt: str) -> dict:
    url = f"{settings.OLLAMA_HOST}/api/chat"
    payload = {
        "model": settings.MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": settings.TEMPERATURE, "num_predict": settings.MAX_TOKENS},
    }

    start = time.perf_counter()
    ttft = None
    final = {}

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if ttft is None and data.get("message", {}).get("content"):
                    ttft = time.perf_counter() - start
                if data.get("done"):
                    final = data
                    break

    total = time.perf_counter() - start
    eval_count = final.get("eval_count", 0)
    eval_dur_s = final.get("eval_duration", 0) / 1e9  # ns -> s
    tok_per_s = (eval_count / eval_dur_s) if eval_dur_s else 0.0

    return {
        "prompt": prompt,
        "ttft_s": ttft or 0.0,
        "total_s": total,
        "gen_tokens": eval_count,
        "tok_per_s": tok_per_s,
    }


def main() -> None:
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Ollama: {settings.OLLAMA_HOST}\n")
    print(f"{'TTFT (s)':>9} | {'Total (s)':>9} | {'Tokens':>6} | {'tok/s':>6} | prompt")
    print("-" * 78)

    rows = []
    for p in PROMPTS:
        r = run_one(p)
        rows.append(r)
        print(
            f"{r['ttft_s']:>9.2f} | {r['total_s']:>9.2f} | "
            f"{r['gen_tokens']:>6} | {r['tok_per_s']:>6.1f} | {p[:32]}"
        )

    print("-" * 78)
    print(
        f"{'avg':>9} | "
        f"{statistics.mean(r['total_s'] for r in rows):>9.2f} | "
        f"{'':>6} | "
        f"{statistics.mean(r['tok_per_s'] for r in rows):>6.1f} |"
    )


if __name__ == "__main__":
    main()
