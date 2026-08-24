"""Token sampling. Pure functions over a logits vector — no model state."""

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
