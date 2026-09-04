"""One request over HTTP, timed the same way the CLI times in-process (M2).

The difference between this number and ``tinyserve generate`` is what the
server layer costs. Same prompt, same chat template, same max_tokens, or
the comparison means nothing.
"""

import argparse
import json
import statistics
import time

import httpx

from tinyserve.engine.runner import Runner
from tinyserve.prompts import PROMPTS


def one_request(url: str, prompt: str, max_tokens: int) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    with httpx.Client(timeout=300) as client:
        with client.stream("POST", url, json={
            "prompt": prompt, "max_tokens": max_tokens, "stream": True,
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[len("data: "):]
                if body == "[DONE]":
                    break
                payload = json.loads(body)
                choice = payload["choices"][0]
                if choice["text"] and ttft is None:
                    ttft = time.perf_counter() - t0
                if choice["finish_reason"]:
                    tokens = payload["usage"]["completion_tokens"]
    return ttft or 0.0, time.perf_counter() - t0, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="medium")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    runner = Runner.load()
    prompt = runner.format_prompt(PROMPTS[args.prompt])

    ttfts, tpss = [], []
    for i in range(args.runs):
        ttft, total, tokens = one_request(args.url, prompt, args.max_tokens)
        decode_tps = (tokens - 1) / (total - ttft) if total > ttft and tokens > 1 else 0.0
        ttfts.append(ttft)
        tpss.append(decode_tps)
        print(f"[run {i + 1}] ttft={ttft * 1000:.0f}ms tokens={tokens} decode={decode_tps:.1f}tok/s")

    print(f"[median of {args.runs}] ttft={statistics.median(ttfts) * 1000:.0f}ms "
          f"decode={statistics.median(tpss):.1f}tok/s")


if __name__ == "__main__":
    main()
