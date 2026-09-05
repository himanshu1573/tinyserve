# Measurements

Every number here has a date and the exact command that produced it.
Median of 3 runs after cooldown unless stated otherwise.

## Machine

- Apple M1, 8 GB unified memory, macOS 26.5
- Memory bandwidth: ~68 GB/s (the number every ceiling is computed from)

---

## Conditions for every number below

- All applications closed (the M1 Air is fanless and shares 8 GB with the OS)
- Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, 28 layers, 2 KV heads,
  head_dim 128, fp16 KV → **28 KB of KV per token**
- Weights resident, by MLX's own counter: **0.81 GB**
- `temperature=0.0` everywhere, so a run is reproducible and token counts
  are comparable
- Every server-side number is the median of 3 runs, each preceded by an
  unmeasured warmup run and separated by a 12 s cooldown. The warmup is not
  optional — see `SURPRISES.md`.
- Prefix caching is **off** for M3-M6 and M8-serial, so that batching and
  paging are measured without a second effect mixed in. M7 is the
  measurement of prefix caching itself.

---

## M1 — the serial baseline, 2026-09-05

```text
Command: .venv/bin/python -m tinyserve.cli generate --prompt medium \
             --max-tokens 128 --runs 3 --quiet
Why:     The in-process loop with no server, no scheduler, no batching.
         Every later number is compared against this one.
Result:  decode 34.0 tok/s | prefill 243.9 tok/s | TTFT 214 ms   (median of 3)
         52 prompt tokens, 101 generated, identical in all three runs.
```

| run | TTFT | prefill | decode | generated |
|---|---|---|---|---|
| 1 | 211 ms | 248.2 tok/s | 34.6 tok/s | 101 |
| 2 | 214 ms | 243.9 tok/s | 34.0 tok/s | 101 |
| 3 | 218 ms | 239.1 tok/s | 33.7 tok/s | 101 |

**Prefill is 7× faster per token than decode** (244 vs 34 tok/s), from the
same weights on the same chip. That ratio is the entire thesis of post #1
measured on this machine: prefill reads the weights once for 52 tokens,
decode reads them once per token.

### What the ceiling is

The M1's memory bandwidth is ~68 GB/s and decoding one token requires
reading all 0.81 GB of resident weights:

```text
ceiling = 68 GB/s ÷ 0.81 GB/token = 84 tok/s
measured = 34 tok/s = 40% of the roofline
```

So the loop leaves ~60% on the table — attention overhead, the sampler,
Python per-step, and MLX kernel launches. **That gap is not the interesting
number.** The interesting number is that the ceiling is 84 tok/s *no matter
how good the code gets*, for **one** user. The only way past 84 is to make
one weight read serve more than one user, which is what the rest of this
file measures.

For reference, post #1 measured llama.cpp at 26 tok/s on this machine.
That was a different build and quantization and was not re-run here, so
treat it as context, not a like-for-like comparison.

---

## M2 — what the server layer costs, 2026-09-05

```text
Command: tinyserve serve --max-batch 8 --kv-backend padded --no-prefix-caching
         driver: 1 user, medium prompt, 128 max tokens, warmup + median of 3
Why:     Same single-sequence work as M1, now through HTTP, SSE, an
         asyncio.Queue and the engine thread. The difference is the tax.
Result:  decode 34.9 tok/s | TTFT 411 ms      (M1 in-process: 34.0 | 214 ms)
```

**Decode through the whole server stack is 34.9 tok/s against 34.0 tok/s
in-process — the server layer costs nothing measurable per token.** That is
the expected answer and it is worth stating plainly: at 34 tok/s a token
takes 29 ms, and FastAPI + SSE + a queue handoff are microseconds. The
server cannot be the bottleneck when the GPU is busy for 29 ms per token.

TTFT is the honest caveat: 411 ms vs 214 ms. Note this is not the same
prompt work — the harness sends the long prompt for its first user while M1
used medium — and the first measurement of it was pure cold-start artifact
(`SURPRISES.md`). What is left after warmup is prefill of a longer prompt
plus one queue handoff.

---

## M3 — the "before" table: a serial engine under load, 2026-09-05

```text
Command: tinyserve serve --max-batch 1 --no-prefix-caching
         driver: N users, mixed prompts (long/medium/short round-robin)
Why:     This is the engine every naive local server is: one request at a
         time. It is the baseline the whole project exists to beat.
Result:  see table
```

| users | aggregate tok/s | per-user tok/s | TTFT median | TTFT p95 | wall |
|---|---|---|---|---|---|
| 1 | 28.8 | 35.2 | 0.8 s | 0.8 s | 4.4 s |
| 2 | 24.0 | 27.6 | 3.4 s | 6.0 s | 9.6 s |
| 4 | 22.7 | 26.6 | 8.2 s | 11.8 s | 16.5 s |
| 8 | 22.7 | 27.2 | 14.9 s | 28.2 s | 31.8 s |

**Adding users to a batch-1 engine makes everything worse and nothing
better.** Aggregate throughput *falls* from 28.8 to 22.7 tok/s, because the
extra users add scheduling and queue work while contributing no extra
parallel work. Meanwhile TTFT grows linearly with the queue: the eighth user
waits **14.9 seconds** at the median and **28.2 seconds** at p95 before
seeing a single character.

This is the shape of the problem. Eight users are not being served; they are
standing in a line. Every weight read serves exactly one of them.

---

## M4 — static batching, 2026-09-05

```text
Command: tinyserve serve --max-batch 8 --scheduling static \
             --kv-backend padded --no-prefix-caching
Why:     The plan expected the headline number here. It is not here, and
         why it is not here is more interesting than the number.
Result:  see table
```

| users | aggregate tok/s | per-user tok/s | TTFT median | TTFT p95 | scheduler steps* |
|---|---|---|---|---|---|
| 1 | 23.2 | 27.9 | 1.0 s | 1.0 s | 396 |
| 2 | 24.1 | 27.6 | 3.4 s | 5.9 s | 714 |
| 4 | 24.6 | 15.0 | 6.3 s | 7.5 s | 792 |
| 8 | 25.7 | 11.9 | 8.1 s | 19.2 s | 1188 |

*steps are cumulative over warmup + 3 runs, and are the diagnostic here.

**Static batching does not batch requests that arrive while a batch is
running.** At 2 users it took **exactly the same 714 steps as the serial
engine** — no batching happened at all — and produced the same throughput
(24.1 vs 24.0) and the same TTFT (3.44 s vs 3.45 s).

The mechanism is one line in `_admit`: in static mode it returns
immediately if anything is running. The first request is admitted alone,
begins its prefill, and every request that lands during that prefill waits
until it *finishes*. Only then are the survivors admitted together — which
is why 4 and 8 users show partial batching (792 and 1188 steps against the
serial engine's 1179 and 2280) and why per-user throughput collapses to
11.9 tok/s while aggregate barely moves.

This is not an implementation bug. It is the actual pathology of static
batching, and it is the reason continuous batching exists. The plan
budgeted two sessions for static batching expecting the headline; the
honest finding is that static batching is nearly worthless for
interactive serving, and the measurement that proves it is the step count.

---

## M5 — continuous vs static, mixed-length workload, 2026-09-05

```text
Command: tinyserve serve --max-batch 8 --scheduling continuous \
             --kv-backend padded --no-prefix-caching
Why:     The same workload and backend as M4, changing only the admission
         policy. This isolates continuous batching itself.
Result:  8 users: aggregate 27.6 tok/s | per-user 5.2 | TTFT 3.6 s
```

| 8 users, padded backend | aggregate | per-user | TTFT median | TTFT p95 |
|---|---|---|---|---|
| serial (batch 1) | 22.7 | 27.2 | 14.9 s | 28.2 s |
| static batching | 25.7 | 11.9 | 8.1 s | 19.2 s |
| **continuous batching** | **27.6** | **5.2** | **3.6 s** | **5.5 s** |

**Continuous batching is a latency win, not a throughput win.** TTFT falls
4.2× against the serial engine (14.9 s → 3.6 s) and p95 falls 5.1×
(28.2 s → 5.5 s). Aggregate throughput rises only 22% (22.7 → 27.6 tok/s).

That is not what the theory predicts. If a batched decode step amortized the
one weight read across 8 rows, aggregate throughput should have multiplied,
not moved 22%. Something is preventing the batch from amortizing the read —
see M9.

---

## M6 — paged vs padded KV, 2026-09-05

```text
Command: tinyserve serve --max-batch 8 --kv-backend {paged,padded} \
             --kv-gb {0.5,0.05} --no-prefix-caching
         plus a dedicated pressure test sampling /stats at ~7 Hz during the run
Why:     What does the hand-written block allocator actually buy on a
         machine with no PagedAttention kernel?
Result:  it costs throughput and buys enforcement, not capacity — see below
```

### What paging costs

Single user, medium prompt, prefix caching off, so the only variable is the
backend:

| backend | decode tok/s | TTFT |
|---|---|---|
| padded | 34.9 | 411 ms |
| paged | 28.1 | 441 ms |

**Paging costs 19% of single-stream decode throughput.** The reason is in
`paged_cache.py`: with no custom Metal kernel, every step gathers each row's
whole context out of the slot pool, per layer. On a bandwidth-bound machine
that gather competes directly with the weight read. At 8 users the gap
narrows to ~2.5% (26.9 vs 27.6 tok/s) because the gather is amortized across
the batch.

This is the opposite of the trade PagedAttention makes on a CUDA GPU, where
the kernel reads blocks in place and paging is nearly free. **Without the
kernel, paging is a memory-capacity feature bought with memory bandwidth.**

### What paging buys

Both backends, 8 users, a 0.05 GB (1872-token) budget, `/stats` sampled
during the run rather than after it:

| backend | peak users running | preemptions | peak KV held | over budget? |
|---|---|---|---|---|
| padded | 8 of 8 | 0 | 44.0 MB | no |
| paged | 8 of 8 | 0 | 53.7 MB (the whole pool) | no |

**Honest result: this workload never created memory pressure, so it did not
demonstrate the capacity win.** The model emits EOS after ~180 tokens
regardless of `max_tokens=384`, so no context ever grew large enough to
strain the budget. Both backends held all 8 users comfortably.

What the run does show is a real structural difference: **the paged pool is
preallocated in full** (53.7 MB reserved whether used or not, and 536 MB at
the default 0.5 GB budget) while **the padded cache grows on demand** to only
what it needs (44.0 MB). On an 8 GB machine that matters: paged reserves the
budget, padded borrows against it.

The capacity advantage of paging is arithmetic and holds regardless of
whether this benchmark provoked it. At a 1872-token budget with this mixed
workload, the padded backend must pad every row to the longest (320 slots →
5 rows fit) while the paged backend rounds each sequence to 16 (1744 slots
for all 8). Demonstrating it empirically needs a workload whose sequences
actually stay long — an open item, not a claim.

---

## M7 — prefix sharing, 2026-09-05

```text
Command: tinyserve serve --max-batch 8 [--no-prefix-caching]
         8 users, each request prefixed with the same SYSTEM_PROMPT
Why:     Does sharing blocks between users pay for the refcounts?
Result:  1.74x on TTFT between users — after removing an effect that
         made it look like 3.0x
```

### The measurement, corrected

The campaign's first pass warmed the server with the same prompts it then
measured, so a "hit" could be a request hitting **its own previous run** —
a response cache, not cross-user sharing. It reported 3.0×. The corrected
experiment gives each arm a freshly started server and exactly one burst of
8 users, so the only sharing possible is between users:

| 8 users, one cold burst | TTFT median | TTFT max | aggregate | hits |
|---|---|---|---|---|
| prefix caching off | 4,073 ms | 6,340 ms | 26.3 tok/s | 0 / 50 |
| prefix caching on | **2,339 ms** | 2,841 ms | 30.1 tok/s | 38 / 50 |

**Cross-user prefix sharing is worth 1.74× on TTFT and +14% on aggregate
throughput.** Cheap for what it is: a hash per full block and a refcount.

### The caveat that matters more than the number

A third arm ran with prefix caching on and **no** shared system prompt:
TTFT 2,184 ms, 26 hits. Statistically the same as the shared-system arm.

So the win above is **not** mostly the shared system preamble. The frozen
prompt set has 3 prompts and the benchmark has 8 users, so users 0/3/6 send
byte-identical prompts and hit each other's blocks completely. The system
preamble contributes 12 extra hits and no measurable latency improvement —
it is ~28 tokens against prompts of 69-157, too small to move TTFT.

Isolating the value of a shared system prompt properly needs a prompt set
with 8 *distinct* user questions behind one preamble. The frozen set cannot
express that, and changing it would invalidate every number above it in this
file. Recorded as a limitation rather than papered over.

---

## M9 — where the batching win actually is, 2026-09-05

M5 raised the question this answers. Continuous batching cut latency 4×
but moved aggregate throughput only 22%, which contradicts the premise of
batching: one weight read should serve the whole batch. So the forward was
timed on its own — no server, no scheduler, no queues, no sampling — at a
fixed 256 tokens of context per row, 24 steps per point.

```text
Command: .venv/bin/python scratchpad/microbench.py   (see repo issue notes)
Why:     Isolate the batched forward from everything else in the engine.
Result:  two regimes with a knee at B~8
```

| batch | padded ms/step | vs B=1 | ms per token | aggregate tok/s |
|---|---|---|---|---|
| 1 | 34.9 | 1.00× | 34.93 | 28.6 |
| 2 | 61.9 | 1.77× | 30.97 | 32.3 |
| 4 | 111.6 | 3.19× | 27.89 | 35.8 |
| 8 | 191.1 | 5.47× | 23.89 | 41.9 |
| 10 | 193.7 | 5.54× | 19.37 | 51.6 |
| 12 | 206.9 | 5.92× | 17.24 | 58.0 |
| 14 | 196.1 | 5.62× | 14.01 | 71.4 |
| 16 | 200.5 | 5.74× | 12.53 | 79.8 |
| 24 | 209.9 | 6.01× | 8.75 | 114.3 |
| 32 | 211.7 | 6.06× | 6.62 | **151.1** |

**Below B≈8, step time grows almost linearly with batch size.** Eight rows
cost 5.5× one row. Every extra user buys a nearly proportional extra cost,
so batching returns almost nothing — which is exactly what M4, M5 and M8
measured at the server level.

**From B≈8 to B=32, step time is flat.** 191 ms for 8 rows, 212 ms for 32:
four times the work for 11% more time. In this regime every extra user is
nearly free and aggregate throughput scales almost linearly, reaching
151 tok/s — **5.3× the single-user rate**, and well past the 84 tok/s
single-stream roofline from M1, which is precisely what serving more than
one user at a time is supposed to achieve.

**The engine's default `max_batch_size` is 8 — the worst point on this
curve.** It sits at the end of the regime where batching does nothing,
just before the regime where it pays for everything. Every throughput
number in M4-M8 was measured there.

The same shape holds for the paged backend, shifted 5-15% slower, and the
gap widens with batch size (24 rows: 241 vs 210 ms; 32 rows: 249 vs 212 ms)
because the per-step context gather grows with batch × context.

### What causes the knee — measured vs inferred

**Measured:** the two regimes and the knee location. Reproducible across
both backends and both sweeps.

**Inferred, not proven:** the likely cause is GPU occupancy in MLX's
quantized matmul. Below ~8 rows the kernel appears to process rows in a
way that repeats the weight traffic per row; past that the batch fills the
machine and the weight read is genuinely shared. Confirming this needs
Metal-level profiling, which is outside what this project set out to do.
The engineering consequence does not depend on the explanation.

### The prediction, tested at the server level

If the knee is real, raising `--max-batch` should help at a load big enough
to fill the batch. Same 16 users, same workload, only the cap changed:

| config | aggregate | TTFT median | TTFT p95 |
|---|---|---|---|
| `--max-batch 8`, 16 users | 29.4 tok/s | 8.1 s | 30.1 s |
| `--max-batch 16`, 16 users | **37.1 tok/s** | **6.5 s** | **10.4 s** |
| `--max-batch 32`, 32 users | 48.5 tok/s | 13.8 s | 22.7 s |

Confirmed: +26% aggregate throughput **and** 2.9× better p95 latency purely
from letting the batch grow. `EngineConfig.max_batch_size` was changed from
8 to 16 as a result — the only code change this measurement campaign caused.

The server reaches 37 tok/s where the isolated forward reaches 80 at the
same batch size. The gap is everything the microbenchmark excluded: prefills
interleaving with decode, `max_prefill_per_step` limiting admission,
sequences finishing at different times so the batch is rarely full, plus
sampling, detokenization and SSE per token. **Half the theoretical batching
win is lost to the scheduler, not to the GPU** — which is where the next
session's work would go.

---

## M8 — the final table, 2026-09-05

The serial engine of M3 against the finished engine (continuous batching,
paged KV, prefix caching), 8 users, mixed prompts:

| | serial (batch 1) | final engine | change |
|---|---|---|---|
| aggregate throughput | 22.7 tok/s | 32.0 tok/s | **1.41×** |
| TTFT median | 14.9 s | 1.2 s | **12.2× faster** |
| TTFT p95 | 28.2 s | 1.8 s | **15.4× faster** |
| per-user throughput | 27.2 tok/s | 5.3 tok/s | 5.1× slower |
| wall clock | 31.8 s | 21.2 s | 1.50× |

Single user, for the cost of all that machinery:

| | serial | final engine |
|---|---|---|
| aggregate | 28.8 tok/s | 27.3 tok/s |
| TTFT | 832 ms | 164 ms |

**The engine costs a single user nothing (27.3 vs 28.8 tok/s, ~5%) and pays
that back 12× in latency for eight.**

### The sentence this project set out to write

The plan wanted: *"My laptop serves 8 people almost as fast as it serves 1."*

**That sentence is false on this machine, and the measurements say why.**
Per-user throughput at 8 users is 5.3 tok/s against 27-35 tok/s for one
user: six times slower, not "almost as fast". Aggregate throughput rises
only 1.41×, because at batch 8 this chip is in the regime where a decode
step costs time proportional to the batch (M9).

What is true, and is a better sentence:

> **My laptop makes eight people wait 1.2 seconds instead of 15 to see
> their first word — and the throughput win everyone talks about doesn't
> start until batch 16.**

Continuous batching on this hardware is a **latency** technology. It turns a
queue into a batch, and the queue was the thing making users wait. The
throughput story needs a bigger batch than 8, and the numbers to prove it
are in M9.
