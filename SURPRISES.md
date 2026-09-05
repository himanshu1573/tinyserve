# Surprises

One line whenever something confuses me or a prediction misses.
Post #1's "26 vs 860" moment came out of a file like this.

---

**2026-09-05 — Prefix caching breaks greedy determinism.** Same prompt,
`temperature=0.0`, same server: the first request generated 101 tokens, every
repeat generated 86. With `--no-prefix-caching` all three runs generated 101,
byte-identical to the in-process CLI. So it is the prefix cache, not the
server or the scheduler. A hit is *mathematically* identical — the stored K/V
was computed from these exact tokens — but a 52-token prefill and a 1-token
prefill-over-cached-prefix are different tensor shapes, so the GEMMs reduce
in a different order and the last bits differ. Greedy sampling is an argmax,
and an argmax over two nearly-tied logits will flip on a 1e-3 difference,
after which the sequences diverge completely. Predicted "caching is free and
invisible"; it is neither. Consequence for this repo: every throughput
comparison runs with prefix caching **off** except M7, which is about it.

**2026-09-05 — psutil RSS is the wrong memory metric for MLX.** The same
server, doing the same work, reported peak RSS anywhere from 0.12 GB to
1.07 GB across runs, while MLX's own `get_peak_memory()` sat rock-steady at
1.32 GB. Weights are mmapped and Metal buffers are not attributed to RSS the
way heap memory is, so the number wanders with page-cache pressure rather
than with allocation. Every memory claim in MEASUREMENTS.md is MLX's counter;
RSS is recorded next to it only to show it disagrees.

**2026-09-05 — What looked like "the server layer costs 130ms of TTFT" was
cold start.** In-process TTFT was 214 ms, over HTTP it was 338-583 ms, and
the obvious story was FastAPI/SSE/queue overhead. It was not: MLX compiles
graphs lazily on first use of each tensor shape, and every measurement was
paying that once per freshly started server. One unmeasured warmup request
before the timed runs brought HTTP TTFT to 195 ms — within noise of
in-process. The lesson is the one from the mx.eval() gotcha wearing a
different hat: on a lazy framework, the first of anything is not a
measurement.
