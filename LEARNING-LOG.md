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
