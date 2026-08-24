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
