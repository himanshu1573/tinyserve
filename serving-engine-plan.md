# The 8 GB Serving Engine — Study & Build Plan

**Goal:** Build a readable LLM serving engine on the M1 MacBook Air, producing one headline sentence for blog #2:
> "My laptop serves 8 people almost as fast as it serves 1."

**Time budget:** ~25–35 hours total. Ten sessions of ~2.5–3 hrs + ~3.5 hrs study tonight. Evenings-only pace: done in ~2 weeks (~Aug 28–30).

**The rule from post #1, applied here:** every build step must end in a measurement you write down. The measurements ARE the blog post. The code is just how you get them.

---

## 0. What you're building

An OpenAI-compatible HTTP server around **Qwen2.5-1.5B-Instruct (4-bit)** running on **MLX**, with a scheduler doing continuous batching and a block-based KV cache allocator written by hand. Target: ~800–1,200 lines of Python. Working name suggestion: `tinyserve`.

**Why MLX** (not llama.cpp or PyTorch): MLX runs on the M1 GPU from Python, and `mlx-lm` exposes its KV cache as plain arrays you can manage yourself. llama.cpp's server hides everything you're trying to learn; PyTorch-on-MPS is slower and clunkier on 8 GB.

---

## 1. Study plan for tonight (~3.5 hours)

Ordered by importance. If you only do two, do the first two.

| Time | What | What you're extracting |
|---|---|---|
| ~75 min | **mlx-lm source**: `pip install mlx-lm`, read `mlx_lm/generate.py` (the `generate_step` loop), `mlx_lm/models/cache.py` (the `KVCache` class), skim `mlx_lm/models/qwen2.py` | Exactly how one decode step works on your machine; where you'll cut in to replace their cache with your paged one. Highest-value hour of the night. |
| ~60 min | **nano-vllm source**: github.com/GeeeekExplorer/nano-vllm — read `scheduler.py`, `block_manager.py`, `sequence.py`. Ignore all CUDA/Triton parts. | The architecture you're porting to Apple Silicon: what a Sequence carries, what the scheduler decides per step, allocate/free/fork on a block table. Note the class names — you'll mirror them. |
| ~30 min | **Re-read Anyscale's continuous batching post** — as a builder this time | Sketch the loop on paper: admit → step → stream → evict. |
| ~30 min | **FastAPI streaming + asyncio**: `StreamingResponse` docs, SSE format (`data: {...}\n\n`), `asyncio.Queue` | The one pattern the server hangs on: handlers `await queue.get()` while one engine thread pushes tokens into per-request queues. |
| ~15 min | **vLLM paper §4 only** (arxiv 2309.06180), Figures 6–8 | Refresh the block-table data structure. Skip evals. |

**Explicitly NOT studying:** CUDA/Triton/FlashAttention internals (wrong hardware), tensor parallelism (one chip), tokenizer internals (mlx-lm ships the HF tokenizer), Metal kernel programming (MLX ops are your floor — hitting their limits is a blog finding, not a rabbit hole).

---

## 2. The components (~800–1,200 lines)

```
tinyserve/
├── engine/
│   ├── runner.py        # MLX wrapper: step(batch, caches) → logits   (~120 lines)
│   ├── sequence.py      # request state: tokens, status, block table  (~60 lines)
│   ├── scheduler.py     # continuous batching loop — THE HEART        (~150 lines)
│   ├── block_manager.py # paged KV: free list, block tables, refcounts (~150 lines)
│   └── sampler.py       # greedy, temperature, top-p                  (~40 lines)
├── server/
│   └── app.py           # FastAPI, /v1/completions, SSE streaming     (~120 lines)
└── bench/
    └── harness.py       # N concurrent clients, TTFT / tok/s / RAM    (~100 lines)
```

- **runner.py** — loads `mlx-community/Qwen2.5-1.5B-Instruct-4bit` via `mlx_lm.load()`. One method: batch of next-token IDs + each sequence's KV cache → next-token logits. Start by calling mlx-lm's own machinery per-sequence; replace internals as you go. Day 4's hard part: a genuinely *batched* forward over different-length sequences needs padding + attention mask — the M1-specific work nano-vllm never had to do.
- **sequence.py** — dataclass: id, prompt tokens, generated tokens, status (WAITING/RUNNING/FINISHED), sampling params, block table, and the asyncio.Queue its tokens stream through. No logic.
- **scheduler.py** — infinite loop in ONE dedicated thread (MLX calls block; keep them off the event loop): (1) admit WAITING sequences while there's room, (2) prefill new ones, (3) one decode step for whole batch, (4) sample + push tokens to queues, (5) evict finished, free blocks.
- **block_manager.py** — the promise from post #1. Fixed blocks of 16 tokens, free list, per-sequence block table (logical → physical). `allocate(seq)` on growth, `free(seq)` on eviction; stretch: `fork(seq)` with refcounts for prefix sharing + copy-on-write.
- **server/app.py** — OpenAI-compatible `POST /v1/completions` with SSE. Compatibility is worth it: any existing client UI can point at your engine = the demo GIF for LinkedIn.
- **bench/harness.py** — build BEFORE batching. Fires N concurrent requests with a fixed prompt set; records TTFT, per-user tok/s, aggregate tok/s, peak RAM (psutil). It's your llama-bench: exists from day 3, never changes after, or before/after numbers aren't comparable.

---

## 3. Build roadmap — ten sessions

Each session ends with a measurement written into `MEASUREMENTS.md`. No number = not done.

| Day | Build | Measurement to record |
|---|---|---|
| 1 | Repo + mlx-lm installed; bare generate loop (no server): load, tokenize, decode step-by-step, print tokens | **M1:** tok/s of your loop vs llama.cpp's 26 — your new baseline |
| 2 | FastAPI + SSE streaming, single request end-to-end; curl shows live tokens | **M2:** TTFT + tok/s through HTTP — what did the server layer cost? |
| 3 | Benchmark harness. No engine changes today. | **M3:** 1/2/4/8 concurrent users vs the *serial* engine — the "before" table |
| 4–5 | Static batching: padded batched forward + attention mask in MLX. Hardest step — two sessions. | **M4:** throughput at batch 1/2/4/8 — **the headline number lives here** |
| 6 | Continuous batching: admit/evict every step | **M5:** mixed-length workload: continuous vs static batching |
| 7–8 | Paged KV block manager; swap in under the runner | **M6:** peak RAM + max concurrent long conversations, paged vs naive contiguous |
| 9 | Stretch: prefix sharing with refcounts — or long-context decay experiment (tok/s as context grows) | **M7:** 8 users sharing one system prompt: blocks with vs without sharing |
| 10 | README, demo GIF, tidy repo, turn MEASUREMENTS.md into blog draft skeleton | **M8:** final before/after table — serial vs finished engine |

**Risk note (days 4–5):** if MLX batched attention over ragged sequences fights you, fall back honestly: loop over sequences but share the weights-read via MLX's lazy evaluation, measure it, write down that it didn't give the full win and why. Post #1's best moments were honest gaps between theory and measurement — a fallback is content, not failure.

---

## 4. Known gotchas

- **8 GB budget:** model ≈ 1 GB, macOS + browser ≈ 3–4 GB → KV budget ≈ 1–2 GB. That constraint is exactly why the paged allocator matters — it's the story. Track with psutil from day one; close Chrome while benchmarking.
- **MLX is lazy:** nothing computes until `mx.eval()`. Time around `mx.eval()`, not the Python call, or every measurement lies.
- **Blocking vs async:** one engine thread, `asyncio.Queue` per request, `loop.call_soon_threadsafe` to hand tokens across. Never call the model inside a request handler.
- **Fixed benchmark inputs:** freeze the prompt set on day 3 (haiku-length, paragraph, essay prompt). Changing prompts mid-project invalidates comparisons.
- **Thermals:** M1 Air has no fan. Run each benchmark 3× after cooldown, report the median — or day-8 numbers will mysteriously beat day-4 numbers.

---

## 5. Capture as you go

Two files in the repo from the first commit:

- `MEASUREMENTS.md` — every number, dated, with the exact command that produced it
- `SURPRISES.md` — one line every time something confuses you or a prediction misses. The "26 vs 860" moment of this project is hiding in this file.

**Tomorrow starts here:** create repo → `pip install mlx-lm fastapi uvicorn psutil` → bare generate loop that prints its own tok/s.

---

*After the build: budget 4–6 hrs for writing the post. LinkedIn opener, same formula as last time: "I made my 8 GB laptop serve 8 people at once. It was barely slower than serving one."*
