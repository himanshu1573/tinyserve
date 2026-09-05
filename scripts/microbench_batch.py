"""Where does the batching win go?

If a batched decode step amortizes the weight read, step time should be
nearly FLAT in batch size B: one read of 0.81 GB serves all B rows.
If step time instead grows linearly with B, the weight read is not being
amortized and 'batching' is just doing B sequential matvecs.

This times the forward alone -- no scheduler, no server, no queues.
"""
import time
import mlx.core as mx


from tinyserve.engine.runner import Runner
from tinyserve.engine.backends import make_backend
from tinyserve.engine.sequence import Sequence

r = Runner.load()
CTX = 256          # tokens of context each row already holds
REPS = 24


def time_steps(fn, reps=REPS):
    fn()                                    # warm the shapes
    mx.eval(mx.zeros(1))
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000   # ms per step


print(f"model weights resident: {r.memory_stats()['mlx_active_gb']:.2f} GB")
print(f"context per row: {CTX} tokens, {REPS} steps per point\n")
print(f"{'backend':8} {'B':>3} {'ms/step':>8} {'tok/s':>8} {'ms/token':>9} {'vs B=1':>7}")

for backend_name in ("padded", "paged"):
    base = None
    for B in (1, 2, 4, 8, 10, 12, 14, 16, 24, 32):
        backend = make_backend(backend_name, r.kv_spec, int(0.9 * 1024**3),
                                   prefix_caching=False)
        seqs = []
        for i in range(B):
            s = Sequence(id=f"s{i}", prompt_tokens=list(range(1000 + i * CTX, 1000 + (i + 1) * CTX)),
                         max_tokens=64)
            backend.admit(s)
            caches = backend.prefill_caches(s)
            r.prefill(s.all_tokens()[s.num_cached_tokens:], caches)
            backend.after_prefill(s, caches)
            s.append(42)
            backend.may_append(s)
            seqs.append(s)

        def step(seqs=seqs, backend=backend):
            caches = backend.decode_caches(seqs)
            r.decode_batch([s.last_token for s in seqs], caches)

        ms = time_steps(step)
        if base is None:
            base = ms
        print(f"{backend_name:8} {B:>3} {ms:>8.1f} {B / ms * 1000:>8.1f} "
              f"{ms / B:>9.2f} {ms / base:>6.2f}x")
    print()
