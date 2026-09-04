"""N concurrent clients against a running server. Built on day 3, frozen.

Every row it prints is comparable with every other row it ever printed,
because the prompt set, the request shape and the timing method never
change. Change the *server* between rows, never this file.

    python -m tinyserve.bench.harness --users 8 --runs 3
    python -m tinyserve.bench.harness --users 8 --prompt long --shared-system

Per request it records TTFT (first SSE text frame) and completion tokens
(from the final frame's usage — real token counts, not segment counts).
Across the run: aggregate tok/s, per-user tok/s, MLX's peak memory (reset
at the start of each run; RSS does not see Metal buffers) and the server's
peak RSS sampled by pid every 100 ms.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass

import httpx
import psutil

from tinyserve.prompts import PROMPTS, SYSTEM_PROMPT


@dataclass
class RequestResult:
    user: int
    prompt: str
    ttft_s: float
    total_s: float
    completion_tokens: int
    finish_reason: str

    @property
    def decode_tps(self) -> float:
        # Tokens after the first, over the time after the first.
        if self.completion_tokens < 2 or self.total_s <= self.ttft_s:
            return 0.0
        return (self.completion_tokens - 1) / (self.total_s - self.ttft_s)


@dataclass
class RunResult:
    users: int
    wall_s: float
    total_tokens: int
    aggregate_tps: float
    per_user_tps_median: float
    ttft_median_ms: float
    ttft_p95_ms: float
    peak_rss_gb: float
    mlx_peak_gb: float
    finish_reasons: dict


async def one_request(client: httpx.AsyncClient, url: str, user: int, prompt: str,
                      prompt_text: str, max_tokens: int) -> RequestResult:
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    reason = "?"
    async with client.stream("POST", url, json={
        "prompt": prompt_text, "max_tokens": max_tokens, "stream": True,
    }) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[len("data: "):]
            if body == "[DONE]":
                break
            payload = json.loads(body)
            if "error" in payload:
                reason = "error"
                continue
            choice = payload["choices"][0]
            if choice.get("text") and ttft is None:
                ttft = time.perf_counter() - t0
            if choice.get("finish_reason"):
                reason = choice["finish_reason"]
                tokens = payload.get("usage", {}).get("completion_tokens", 0)
    return RequestResult(user, prompt, ttft or 0.0, time.perf_counter() - t0, tokens, reason)


async def sample_rss(pid: int, stop: asyncio.Event, out: list) -> None:
    proc = psutil.Process(pid)
    while not stop.is_set():
        try:
            out.append(proc.memory_info().rss)
        except psutil.Error:
            break
        await asyncio.sleep(0.1)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


async def run_once(base_url: str, users: int, prompt_keys: list[str],
                   prompt_texts: dict[str, str], max_tokens: int) -> RunResult:
    async with httpx.AsyncClient(timeout=600) as client:
        health = (await client.get(f"{base_url}/health")).json()
        pid = health["pid"]
        await client.post(f"{base_url}/stats/reset-peak")
        stop = asyncio.Event()
        samples: list[int] = []
        sampler = asyncio.create_task(sample_rss(pid, stop, samples))

        url = f"{base_url}/v1/completions"
        t0 = time.perf_counter()
        results = await asyncio.gather(*(
            one_request(client, url, u, key, prompt_texts[key], max_tokens)
            for u, key in zip(range(users), prompt_keys)
        ))
        wall = time.perf_counter() - t0
        stop.set()
        await sampler
        mlx_peak = (await client.get(f"{base_url}/stats")).json().get("mlx_peak_gb", 0.0)

    total = sum(r.completion_tokens for r in results)
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r.finish_reason] = reasons.get(r.finish_reason, 0) + 1
    return RunResult(
        users=users,
        wall_s=wall,
        total_tokens=total,
        aggregate_tps=total / wall if wall else 0.0,
        per_user_tps_median=statistics.median(r.decode_tps for r in results),
        ttft_median_ms=statistics.median(r.ttft_s for r in results) * 1000,
        ttft_p95_ms=percentile([r.ttft_s for r in results], 95) * 1000,
        peak_rss_gb=max(samples, default=0) / 1024**3,
        mlx_peak_gb=mlx_peak,
        finish_reasons=reasons,
    )


def build_prompts(base_url: str, users: int, prompt: str, shared_system: bool):
    """Apply the chat template client-side so the model sees exactly the
    tokens the in-process CLI feeds it. /v1/completions stays a raw-prompt
    endpoint, which is what OpenAI compatibility means."""
    from tinyserve.engine.runner import Runner  # tokenizer only; weights load too — once

    runner = Runner.load()
    system = SYSTEM_PROMPT if shared_system else None
    keys = sorted(PROMPTS) if prompt == "mixed" else [prompt]
    texts = {k: runner.format_prompt(PROMPTS[k], system=system) for k in keys}
    per_user = [keys[u % len(keys)] for u in range(users)]
    return per_user, texts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="tinyserve.bench.harness")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--prompt", default="mixed",
                    help="short|medium|long or 'mixed' (round-robin)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--shared-system", action="store_true",
                    help="prepend the frozen SYSTEM_PROMPT to every user (M7)")
    ap.add_argument("--cooldown", type=float, default=20.0,
                    help="seconds to idle between runs (fanless M1)")
    ap.add_argument("--json", default=None, help="append results as JSON lines")
    args = ap.parse_args(argv)

    per_user, texts = build_prompts(args.url, args.users, args.prompt, args.shared_system)
    runs: list[RunResult] = []
    for i in range(args.runs):
        if i:
            time.sleep(args.cooldown)
        r = asyncio.run(run_once(args.url, args.users, per_user, texts, args.max_tokens))
        runs.append(r)
        print(f"[run {i + 1}] users={r.users} wall={r.wall_s:.1f}s tokens={r.total_tokens} "
              f"aggregate={r.aggregate_tps:.1f}tok/s per-user={r.per_user_tps_median:.1f}tok/s "
              f"ttft_med={r.ttft_median_ms:.0f}ms ttft_p95={r.ttft_p95_ms:.0f}ms "
              f"mlx_peak={r.mlx_peak_gb:.2f}GB rss={r.peak_rss_gb:.2f}GB finish={r.finish_reasons}")

    med = lambda f: statistics.median(f(r) for r in runs)  # noqa: E731
    print()
    print("| users | prompt | aggregate tok/s | per-user tok/s | TTFT median | TTFT p95 | MLX peak | peak RSS |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| {args.users} | {args.prompt}{' +system' if args.shared_system else ''} "
          f"| {med(lambda r: r.aggregate_tps):.1f} | {med(lambda r: r.per_user_tps_median):.1f} "
          f"| {med(lambda r: r.ttft_median_ms):.0f} ms | {med(lambda r: r.ttft_p95_ms):.0f} ms "
          f"| {med(lambda r: r.mlx_peak_gb):.2f} GB | {med(lambda r: r.peak_rss_gb):.2f} GB |")

    if args.json:
        with open(args.json, "a") as f:
            for r in runs:
                f.write(json.dumps({"args": vars(args), **asdict(r)}) + "\n")


if __name__ == "__main__":
    main()
