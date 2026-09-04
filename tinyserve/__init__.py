"""tinyserve — a readable LLM serving engine for an 8 GB M1.

Reference implementation of the full ten-session plan: an owned decode loop
on MLX, an engine thread, OpenAI-compatible SSE streaming, a continuous
batching scheduler, and a hand-written paged KV cache with prefix sharing.
"""
