# eightserve — Design (Sessions 1–2)

**Date:** 2026-08-19
**Status:** approved approach, pending implementation plan

## Goal

Build a readable LLM serving engine on an 8 GB M1 MacBook Air, producing one
defensible sentence for blog post #2:

> "My laptop serves 8 people almost as fast as it serves 1."

The code is how you get the measurements. The measurements are the post.

## Constraints

| Constraint | Value | Consequence |
|---|---|---|
| Unified memory | 8 GB total | ~1 GB model + 3–4 GB macOS/browser leaves 1–2 GB for KV. This is why the paged allocator matters — it is the story, not a detail. |
| Memory bandwidth | ~68 GB/s | Decode ceiling ≈ 68 / model_bytes. Every tok/s number gets compared against this. |
| Cooling | M1 Air is fanless | Every benchmark: 3 runs after cooldown, report the median. Otherwise later days mysteriously beat earlier ones. |
| MLX evaluation | Lazy | Time around `mx.eval()`, never around the Python call. |

## Scope

**In scope now (Sessions 1–2):** repo scaffold, environment, the decode loop we
own, HTTP + SSE streaming, measurements M1 and M2.

**Out of scope now:** batching (Sessions 4–5), continuous batching (6), paged KV
(7–8), prefix sharing (9). Their files are created on their days. No empty stubs
today — scaffolding a file before it has behaviour reads as progress that has not
happened.

## Approach decision: we own the decode loop

`mlx_lm.load()` supplies weights and tokenizer. Everything after that is ours:
cache construction, prefill, decode step, sampling. We never call
`mlx_lm.generate_step`.

Rejected: wrapping `generate_step` and replacing it later. Two reasons.

1. **M1 must measure our code.** A baseline taken from mlx-lm's loop and a final
   number taken from eightserve's loop are two different programs. The before/after
   table stops being defensible, and that table is the post.
2. **Session 4–5 needs the raw call in hand.** A batched forward over ragged
   sequences means calling `model(ids, cache=caches)` with our own padding and
   mask. Owning that call makes Session 4 an extension; wrapping it makes Session 4
   a rewrite — of the riskiest session in the schedule.

Cost: roughly 30 extra minutes in Session 1.

## Architecture

```
eightserve/
├── engine/
│   ├── runner.py     # owns MLX: load, prefill, decode_step
│   ├── sequence.py   # request state: tokens, status, params, output queue
│   ├── sampler.py    # greedy / temperature / top-p
│   └── engine.py     # engine thread + request intake
├── server/
│   └── app.py        # FastAPI, POST /v1/completions, SSE
└── cli.py            # python -m eightserve.cli generate --prompt ...
```

### Component contracts

**`runner.py`** — the only module that imports `mlx`. Loads
`mlx-community/Qwen2.5-1.5B-Instruct-4bit`.

- `load(model_id) -> Runner`
- `prefill(token_ids) -> (logits_for_last_position, cache)` — one forward over the
  whole prompt, `mx.eval()` before returning so the caller can time it truthfully
- `decode_step(token_id, cache) -> logits` — one token forward, cache mutated in
  place

Sessions 4–5 add `decode_batch(token_ids, caches)` alongside `decode_step`. The
single-sequence path stays as the reference implementation to diff against.

**`sequence.py`** — a dataclass, no logic: id, prompt tokens, generated tokens,
status (`WAITING` / `RUNNING` / `FINISHED`), sampling params, output queue. Gains a
block table in Session 7.

**`sampler.py`** — pure functions over logits. Greedy, temperature, top-p. No MLX
state, no I/O; the one module fully testable without loading a model.

**`engine.py`** — one dedicated thread owning the runner. Pulls sequences off an
intake queue, runs them, pushes tokens back to each sequence's `asyncio.Queue` via
`loop.call_soon_threadsafe`. Batch size is 1 in Session 2; the shape is what
Session 6 needs.

**`server/app.py`** — OpenAI-compatible `POST /v1/completions` with SSE frames
(`data: {...}\n\n`, terminated by `data: [DONE]`). Compatibility is deliberate: any
existing chat UI can point at it, which is the demo for the post.

## Data flow

**Session 1, in-process:**

```
cli → Runner.load → tokenize → prefill(ids) → [decode_step → sample → emit]* → stop
```

Stop conditions: EOS token, or `max_tokens` reached.

**Session 2, over HTTP:**

```
client ─HTTP─> handler ─intake queue─> engine thread ─(call_soon_threadsafe)─> asyncio.Queue ─SSE─> client
```

The request handler never touches MLX. This is a hard rule, not a preference: a
blocking model call on the event loop stalls every other connection, and Session 6
depends on the separation already existing.

## Measurements

Recorded in `MEASUREMENTS.md` with the exact command and the date. A session
without a number is not finished.

**M1 — the baseline.** Four numbers on the frozen prompt set, median of 3 after
cooldown:

- roofline ceiling: 68 GB/s ÷ actual on-disk model bytes
- `llama-bench` on `qwen2.5-1.5b-instruct-q4_k_m.gguf`, run fresh today
- eightserve decode tok/s
- eightserve prefill tok/s

The llama.cpp number is re-run rather than quoted from post #1 so the comparison is
between two programs measured on the same machine on the same day.

**M2 — what the server layer cost.** TTFT and tok/s through HTTP against the same
prompts in-process. The delta is the answer.

**Frozen prompt set**, fixed in Session 1 and never changed — changing prompts
mid-project invalidates every comparison:

- short: a haiku request (~15 prompt tokens)
- medium: a paragraph explanation (~80 tokens)
- long: an essay request (~400 tokens)

## Testing

Fast tests where they are cheap and meaningful:

- `sampler` — greedy is deterministic; temperature and top-p tested with a fixed
  seed; top-p filtering is pure logic with exact expected sets
- `sequence` — status transitions, stop conditions
- SSE frame formatting — exact bytes, including the `[DONE]` terminator

One integration test loads the real model and generates a few tokens, behind a
`slow` marker. MLX is not mocked: a mocked forward pass tests nothing worth
testing, and would pass while the real path is broken.

## Capture files

Live from the first scaffold commit.

- `MEASUREMENTS.md` — every number, dated, with the command that produced it
- `SURPRISES.md` — one line whenever something confuses or a prediction misses.
  Post #1's "26 vs 860" moment came from exactly this file.
- `LEARNING-LOG.md` — what was built, why that way, and what the mlx-lm and
  nano-vllm source revealed, explained at the point we touched it. This is the
  re-readable one; the other two are raw data.

## Risks

**Python 3.14 has no mlx-lm wheel.** System Python is 3.14.3. Mitigation: `uv venv
--python 3.12`. Verify the import before writing any engine code — discovering this
after the loop is written wastes a session.

**Model download is ~1 GB.** On a slow connection this eats Session 1. Mitigation:
start the download first, write the sampler and its tests while it runs.

**Session 4–5 batched attention may resist.** Not today's problem, but the fallback
is decided now: loop over sequences sharing the weights read via MLX's lazy
evaluation, measure it, and write down honestly that it did not give the full win
and why. Post #1's best moments were gaps between theory and measurement. A
documented fallback is content, not failure.
