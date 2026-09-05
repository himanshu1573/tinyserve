# Learning Log

What was built, why it was built that way, and what the source of
mlx-lm and nano-vllm actually revealed — written at the moment we
touched each piece.

`MEASUREMENTS.md` holds the numbers. `SURPRISES.md` holds the
confusions. This file holds the understanding.

---

## Sampler — why greedy is the default

`temperature=0.0` is the default and every measurement run uses it. With
sampling on, two runs of the same prompt produce different token counts and
different text, so tok/s numbers stop being comparable. Determinism is a
measurement requirement here, not a quality preference.

The top-p filter keeps tokens while `cumulative - ordered < top_p` rather
than `cumulative < top_p`. The difference matters when a single token holds
more probability than top_p: the naive form keeps nothing and the sample is
undefined. This form always keeps at least the top token.

## Runner — what mlx-lm actually gives you

`mlx_lm.load()` returns `(model, tokenizer)`. The model is a plain callable:
`model(ids, cache=cache)` returns logits of shape `(batch, seq_len, vocab)`.
That is the entire interface the generation loop needs — `generate_step` in
`mlx_lm/generate.py` is a convenience wrapper around exactly this call plus
sampling, and we replace it with our own thirty lines.

`make_prompt_cache(model)` returns a **list with one cache object per layer**,
not a single object. Each one holds that layer's K and V and grows as tokens
are appended. This is the thing Session 7 replaces with a block-based
allocator, and the list-per-layer shape is why the block table is per-layer too.

**Prefill and decode are the same call with a different sequence length.**
Prefill passes N tokens and gets N positions of logits back; we keep only
the last. Decode passes 1 token. Nothing else differs — no separate code
path, no special mode. The two phases from post #1 turn out to be one
function with a different input shape, which explains why prefill is
compute-bound (lots of tokens, one weight read) and decode is
memory-bound (one token, one weight read).

**MLX is lazy.** `model(...)` returns instantly and computes nothing. Without
`mx.eval(out)` inside prefill and decode_step, the timers in the CLI would
measure graph construction and the whole first measurement would be fiction.

## Engine — why a thread and not just `await`

MLX calls block. `model(...)` builds a graph instantly, but `mx.eval()` sits
on the GPU until the work is done, and nothing in MLX is awaitable. Calling
it inside a FastAPI handler would stall the event loop for the whole
forward — every other user's SSE stream would freeze while one user's token
was computed.

So the engine owns exactly one thread, and that thread is the only place
the model is ever touched. Requests reach it through a plain
`queue.Queue`; tokens come back through one `asyncio.Queue` per request,
written with `loop.call_soon_threadsafe`. That call is not decoration: an
`asyncio.Queue` is not thread-safe, and touching it directly from the
engine thread corrupts the event loop's internal state in ways that
surface much later as a hung request.

The loop blocks on intake only when there is no work (`block = not
sched.has_work()`). With work in flight it drains the queue without
blocking and goes straight into the next step, so a request arriving
mid-batch waits at most one decode step — a few tens of milliseconds —
rather than a poll interval. This is why there is no `sleep` anywhere in
the loop: a sleep would be a floor on admission latency.

## Scheduler — what continuous batching actually decides

The whole thing is twelve lines: `step()` calls `_admit()` then
`_decode()`. Everything interesting is in what `_admit` is allowed to do.

**Static** returns immediately if anything is running: a batch is formed,
run to completion, and only then is the next one admitted. **Continuous**
admits on every single step, so a request that arrives while eight others
are mid-generation joins the very next forward. The difference between the
two modes is literally `if self.scheduling == "static" and self.running:
return`, which is a good measure of how much of "continuous batching" is
architecture rather than algorithm.

Two bounds keep it honest. `max_batch_size` caps the rows in a forward.
`max_prefill_per_step` caps how many waiting sequences may be prefilled
before the running batch gets its decode step — without it, a burst of
arrivals starves the users already streaming, because prefill is far more
expensive per call than decode. That is the scheduler's real job: not
"go fast", but decide whose latency to spend.

Preemption is the release valve. Before a decode step every running
sequence must be able to append one token; if the pool cannot provide a
slot, the *newest* sequence is evicted, its blocks are freed, and it goes
back to the front of the waiting queue to be re-prefilled later. Evicting
the newest rather than the oldest keeps the users who have waited longest
making progress, and re-prefill is the cost paid for admitting optimistically.

## Padded batch cache — the shape that makes one forward possible

To decode B sequences of different lengths in a single forward they must
share one tensor, and the new token for every row must land in the same
column. That forces **left** padding:

    row 0:  [_ _ t t t t]     next token goes in column 6
    row 1:  [t t t t t t]     next token goes in column 6
    row 2:  [_ _ _ _ t t]     next token goes in column 6

Right padding would put each row's write at a different index and there
would be no single slice to assign. Left padding costs a mask — a row must
not attend to its own padding — and `make_mask` builds it from two
conditions: causal (`j <= query position`) and not-padding (`j >=
left_pad[b]`).

The expensive part is churn. `extend()` admits a sequence by concatenating
a row, and if the newcomer is longer than everything already in the batch,
**every existing row is re-padded and copied**. `filter()` evicts by
copying the survivors. Admission and eviction are therefore O(whole cache),
which is exactly the cost that motivates the block allocator — and the
reason static batching is a natural fit for this backend while continuous
batching is not.

## Block manager — paging, and the two bugs hiding in it

Blocks are 16 tokens. A sequence holds a block table mapping logical block
i to a physical block id, so its KV can be scattered across the pool and
still be addressed. Free blocks live in a list; `free()` walks the table in
reverse decrementing refcounts, returning a block only when its count
reaches zero. That refcount is what makes sharing possible at all.

Prefix sharing hashes each **full** block's tokens together with the
previous block's hash, so a hash identifies a whole prefix path and not
just 16 tokens in isolation — two different conversations that happen to
contain the same 16 tokens in different positions must not collide. A hit
bumps the refcount and skips recomputing those positions entirely.

Two subtleties in this file are worth more than the rest of it:

1. **A full block is not shareable the moment it is full.** A token is
   appended to the sequence one step *before* its K/V is written (the write
   happens when it is fed into the next forward). Registering a block for
   sharing at append time means a later hit could read a slot holding
   zeros for that last position. So registration is deferred to
   `may_append`, one step later, when the write has provably happened.
2. **A fully cached prompt still needs one token to prefill.**
   `num_cached_tokens` is clamped to `num_tokens - 1`, because prefill
   produces the logits the first sampled token comes from — with nothing
   left to compute there are no logits. The recomputed last token writes
   identical K/V into a slot that already holds it, so it is safe as well
   as necessary.

## Paged cache — PagedAttention without a kernel

vLLM's PagedAttention is a custom CUDA kernel that reads scattered blocks
*inside* the attention computation. There is no such kernel here, and
writing one in Metal was ruled out as the wrong rabbit hole, so paging is
done the honest way: one flat pool per layer, `slot = block_id *
block_size + i`, and a `BatchPlan` computed before each forward that says
where this step's K/V is written and which slots the context is gathered
from.

The gather is the price. `self.keys[self.plan.slot_table]` copies every
row's whole context, per layer, per step. On a machine whose bottleneck is
memory bandwidth, paging therefore costs bandwidth to save capacity —
which is the opposite trade from the one PagedAttention makes on a GPU
with a kernel. The measurements show what that costs; it is the most
Apple-Silicon-specific finding in the project.

The reason the unmodified mlx-lm model runs on top of it at all is that
`PagedKVCache` implements the same three things mlx-lm's own cache does:
`update_and_fetch`, an `offset` (here a per-row array, which
`mx.fast.rope` accepts), and `make_mask`. Nothing in the model knows its
cache is paged.

## Server — the whole OpenAI compatibility is the shape of a dict

`/v1/completions` streaming is four moving parts: a `Sequence` carrying its
own `asyncio.Queue` and the loop it belongs to, an async generator that
awaits that queue, `StreamingResponse`, and `sse_frame` putting `data: ` in
front of a JSON blob and `\n\n` after it. The engine thread never touches
FastAPI and FastAPI never touches the model; the queue is the only contact
surface, which is why the request handler can be cancelled (client hangs up)
without the engine noticing or caring.

Compatibility itself turned out to be cheap — matching OpenAI's response
shape is a dict literal with the right keys, not a protocol — and it buys the
demo: any existing chat UI can point at this server. The one non-obvious
piece is `usage`: `completion_tokens` counts generated tokens *excluding*
EOS, because that is what OpenAI reports and what a benchmark divides by.

Detokenization has to be incremental and stateful. A token is not a
character: a multi-byte UTF-8 sequence or a partial word arrives across
several tokens, so the engine keeps one mlx-lm detokenizer per sequence and
emits `last_segment` only when it is non-empty. Streaming raw `decode(token)`
per token would emit replacement characters mid-word. This is also why TTFT
is measured on the first non-empty *text* segment rather than the first
token: it is the first moment a user sees something.

## Batching — the thing the whole project was built to observe

The premise of batching is that decode is memory-bound: one read of the
weights, one token per user, so B users should cost roughly what 1 user
costs. Every serving-engine explanation says this, and it is why continuous
batching is supposed to be free throughput.

**On this chip it is only true above a batch size of about 8.** Timing the
forward alone (M9): 1 row costs 35 ms, 8 rows cost 191 ms — 5.5× the time
for 8× the work, so nearly nothing is shared. Then 32 rows cost 212 ms:
four times the work of 8 rows for 11% more time, so almost everything is
shared. Two regimes, and `max_batch_size = 8` had been sitting exactly on
the boundary, in the half where batching does nothing.

The lesson generalises past this laptop: **"decode is memory-bound" is a
statement about the hardware, not about your code.** Whether a batched
matmul actually amortises the weight read depends on the kernel and on
whether the batch is large enough to fill the machine. Below that size you
pay for batching (bigger masks, more cache churn, more scheduling) and
receive none of its benefit. The only way to know where your threshold is
is to time the forward alone, at several batch sizes, on the machine you
intend to serve from.

The second half of the lesson is where the rest of the win went. At batch
16 the isolated forward does 80 tok/s and the actual server does 37. The
missing half is not the GPU: it is prefill interleaving with decode,
`max_prefill_per_step` throttling admission, and sequences finishing at
different times so the batch is rarely full. Once the kernel stops being
the bottleneck, **the scheduler becomes the bottleneck** — which is
presumably why real serving engines put so much work into chunked prefill
and admission policy.
