"""tinyserve command line.

    tinyserve generate --prompt medium          the loop in-process, timed (M1)
    tinyserve serve --max-batch 8 --kv-gb 0.5   the HTTP server
    python -m tinyserve.bench.harness --users 8 N concurrent clients (M3+)

Every timer in ``generate`` brackets an mx.eval() inside the runner, so
the numbers describe work actually done rather than graphs actually built.
"""

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import psutil

from tinyserve.config import DEFAULT_MODEL, EngineConfig
from tinyserve.prompts import PROMPTS


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


def generate(runner, instruction, max_tokens=128, temperature=0.0, top_p=1.0,
             on_text=None, system=None) -> GenResult:
    """The single-sequence loop of Session 1: prefill, then decode_step
    until EOS or max_tokens. Contiguous cache, no scheduler, no threads."""
    from tinyserve.engine.sampler import sample

    proc = psutil.Process()
    ids = runner.encode(runner.format_prompt(instruction, system=system))

    detok = runner.new_detokenizer()
    cache = runner.new_cache()

    t0 = time.perf_counter()
    logits = runner.prefill(ids, cache)
    prefill_s = time.perf_counter() - t0

    pieces, generated, ttft_s = [], 0, None
    peak_rss = proc.memory_info().rss
    t_decode_start = time.perf_counter()

    for _ in range(max_tokens):
        tok = sample(logits, temperature=temperature, top_p=top_p)
        if tok in runner.eos_ids:
            break
        if ttft_s is None:
            # First token = prefill + one sample. This is the pause a user
            # sees before text appears.
            ttft_s = time.perf_counter() - t0
        detok.add_token(tok)
        if detok.last_segment:
            pieces.append(detok.last_segment)
            if on_text:
                on_text(detok.last_segment)
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
    )


def cmd_generate(args) -> None:
    from tinyserve.engine.runner import Runner

    runner = Runner.load(args.model)
    instruction = PROMPTS[args.prompt] if args.prompt in PROMPTS else args.prompt

    results = []
    for i in range(args.runs):
        printer = None if args.quiet else (lambda s: print(s, end="", flush=True))
        r = generate(runner, instruction, args.max_tokens, args.temperature,
                     args.top_p, on_text=printer)
        if not args.quiet:
            print()
        results.append(r)
        print(f"[run {i + 1}] prompt={r.prompt_tokens}tok "
              f"generated={r.generated_tokens}tok "
              f"ttft={r.ttft_s * 1000:.0f}ms "
              f"prefill={r.prefill_tps:.1f}tok/s "
              f"decode={r.decode_tps:.1f}tok/s "
              f"rss={r.peak_rss_gb:.2f}GB")

    if args.runs > 1:
        print(f"[median of {args.runs}] "
              f"decode={statistics.median(r.decode_tps for r in results):.1f}tok/s "
              f"prefill={statistics.median(r.prefill_tps for r in results):.1f}tok/s "
              f"ttft={statistics.median(r.ttft_s for r in results) * 1000:.0f}ms")


def cmd_serve(args) -> None:
    import uvicorn

    # The uvicorn factory takes no arguments, so the config rides on env vars.
    os.environ["TINYSERVE_MODEL"] = args.model
    os.environ["TINYSERVE_MAX_BATCH"] = str(args.max_batch)
    os.environ["TINYSERVE_SCHEDULING"] = args.scheduling
    os.environ["TINYSERVE_KV_BACKEND"] = args.kv_backend
    os.environ["TINYSERVE_KV_GB"] = str(args.kv_gb)
    os.environ["TINYSERVE_BLOCK_SIZE"] = str(args.block_size)
    os.environ["TINYSERVE_PREFIX_CACHING"] = "0" if args.no_prefix_caching else "1"
    uvicorn.run("tinyserve.server.app:get_app", factory=True,
                host=args.host, port=args.port, log_level="warning")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="tinyserve")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run the single-sequence loop in-process")
    g.add_argument("--prompt", default="medium",
                   help="short|medium|long from the frozen set, or literal text")
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--runs", type=int, default=1)
    g.add_argument("--quiet", action="store_true", help="stats only, no text")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("serve", help="start the OpenAI-compatible server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--max-batch", type=int, default=EngineConfig.max_batch_size)
    s.add_argument("--scheduling", choices=["continuous", "static"],
                   default=EngineConfig.scheduling)
    s.add_argument("--kv-backend", choices=["paged", "padded"],
                   default=EngineConfig.kv_backend)
    s.add_argument("--kv-gb", type=float, default=EngineConfig.kv_budget_gb)
    s.add_argument("--block-size", type=int, default=EngineConfig.block_size)
    s.add_argument("--no-prefix-caching", action="store_true")
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
