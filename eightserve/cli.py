"""Run the loop in-process and print what it cost.

Every timer here brackets an mx.eval() inside the runner, so the numbers
describe work actually done rather than graphs actually built.
"""

import argparse
import time
from dataclasses import dataclass, asdict

import psutil

from eightserve.engine.runner import Runner, DEFAULT_MODEL
from eightserve.engine.sampler import sample
from eightserve.prompts import PROMPTS


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float
    prefill_s: float
    decode_s: float
    decode_tps: float
    prefill_tps: float
    peak_rss_gb: float
    peak_mlx_gb: float


def generate(runner, instruction, max_tokens=128, temperature=0.0, top_p=1.0,
             on_text=None) -> GenResult:
    proc = psutil.Process()
    ids = runner.encode(runner.format_prompt(instruction))

    detok = runner.tokenizer.detokenizer
    detok.reset()

    cache = runner.new_cache()

    t0 = time.perf_counter()
    logits = runner.prefill(ids, cache)
    prefill_s = time.perf_counter() - t0

    pieces = []
    generated = 0
    ttft_s = None
    peak_rss = proc.memory_info().rss
    t_decode_start = time.perf_counter()

    for _ in range(max_tokens):
        tok = sample(logits, temperature=temperature, top_p=top_p)
        if tok == runner.eos_id:
            break

        if ttft_s is None:
            # First token is prefill plus one decode step — this is what a
            # user experiences as the pause before text appears.
            ttft_s = time.perf_counter() - t0

        detok.add_token(tok)
        segment = detok.last_segment
        if segment:
            pieces.append(segment)
            if on_text:
                on_text(segment)

        generated += 1
        peak_rss = max(peak_rss, proc.memory_info().rss)
        logits = runner.decode_step(tok, cache)

    detok.finalize()
    if detok.last_segment:
        pieces.append(detok.last_segment)
        if on_text:
            on_text(detok.last_segment)

    decode_s = time.perf_counter() - t_decode_start

    return GenResult(
        text="".join(pieces),
        prompt_tokens=len(ids),
        generated_tokens=generated,
        ttft_s=ttft_s or 0.0,
        prefill_s=prefill_s,
        decode_s=decode_s,
        decode_tps=generated / decode_s if decode_s else 0.0,
        prefill_tps=len(ids) / prefill_s if prefill_s else 0.0,
        peak_rss_gb=peak_rss / 1024**3,
        peak_mlx_gb=runner.peak_memory_gb(),
    )


def main():
    ap = argparse.ArgumentParser(prog="eightserve.cli")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="medium")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="stats only, no text")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    runner = Runner.load(args.model)
    instruction = PROMPTS[args.prompt]

    results = []
    for i in range(args.runs):
        printer = None if args.quiet else (lambda s: print(s, end="", flush=True))
        r = generate(runner, instruction, args.max_tokens,
                     args.temperature, args.top_p, on_text=printer)
        if not args.quiet:
            print()
        results.append(r)
        print(f"[run {i+1}] prompt={r.prompt_tokens}tok "
              f"generated={r.generated_tokens}tok "
              f"ttft={r.ttft_s*1000:.0f}ms "
              f"prefill={r.prefill_tps:.1f}tok/s "
              f"decode={r.decode_tps:.1f}tok/s "
              f"rss={r.peak_rss_gb:.2f}GB "
              f"mlx_peak={r.peak_mlx_gb:.2f}GB")

    if args.runs > 1:
        med = sorted(r.decode_tps for r in results)[len(results) // 2]
        print(f"[median of {args.runs}] decode={med:.1f}tok/s")


if __name__ == "__main__":
    main()
