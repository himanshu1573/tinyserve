"""The frozen prompt set. Fixed 2026-08-19, never changed.

Changing any string here makes every measurement recorded before the
change incomparable to every measurement after it. That is the whole
reason this is a module and not an argument.
"""

PROMPTS = {
    "short": "Write a haiku about a laptop fan that isn't.",
    "medium": (
        "Explain in one paragraph why generating a single token from a "
        "language model requires reading the entire model out of memory."
    ),
    "long": (
        "Write a detailed essay of several paragraphs about the memory "
        "hierarchy of a modern computer. Cover registers, the cache "
        "levels, main memory, and storage. For each level, explain its "
        "approximate size, its approximate latency, and the reason that "
        "level exists at all rather than being merged into its neighbour. "
        "Then explain what the phrase 'memory bandwidth bound' means for "
        "a program whose inner loop reads more bytes than it performs "
        "arithmetic operations, and why adding faster arithmetic units to "
        "such a program produces no speedup whatsoever."
    ),
}
