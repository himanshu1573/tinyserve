"""Token sampling. Pure functions over logits — no model state, no I/O."""

import mlx.core as mx


def sample(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """Pick the next token id from a 1-D logits array of shape (vocab,).

    temperature == 0.0 means greedy (argmax), which is what every
    measurement run uses: sampling noise would make two runs of the same
    prompt incomparable.
    """
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits of shape (vocab,), got {logits.shape}")

    if temperature == 0.0:
        return int(mx.argmax(logits).item())

    scaled = logits * (1.0 / temperature)

    if top_p >= 1.0:
        return int(mx.random.categorical(scaled).item())

    probs = mx.softmax(scaled, axis=-1)
    order = mx.argsort(-probs)
    ordered = probs[order]
    cumulative = mx.cumsum(ordered)

    # Keep every token up to and including the one that crosses top_p, so
    # the nucleus is never empty even when one token holds all the mass.
    keep = cumulative - ordered < top_p
    filtered = mx.where(keep, ordered, mx.zeros_like(ordered))

    choice = mx.random.categorical(mx.log(filtered + 1e-10))
    return int(order[choice].item())


def sample_batch(logits: mx.array, seqs) -> list[int]:
    """One token per row of a (B, vocab) logits array.

    The all-greedy case is a single argmax and a single .tolist(), which
    matters: with eight users, eight separate .item() calls are eight
    GPU round-trips per step.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits of shape (B, vocab), got {logits.shape}")
    if logits.shape[0] != len(seqs):
        raise ValueError(f"{logits.shape[0]} logit rows for {len(seqs)} sequences")

    if all(s.temperature == 0.0 for s in seqs):
        return [int(t) for t in mx.argmax(logits, axis=-1).tolist()]

    return [
        sample(logits[i], temperature=s.temperature, top_p=s.top_p)
        for i, s in enumerate(seqs)
    ]
