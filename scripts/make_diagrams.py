"""Render the README explainer diagrams: why tinyserve exists, what it is,
and what one scheduler step does.

Same deal as make_figures.py: plain SVG written by hand, once per theme, so
the README can pick the right one for the reader's GitHub theme.

    python scripts/make_diagrams.py
"""

from make_figures import OUT, THEMES, svg, txt

FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"


# --- small drawing helpers --------------------------------------------------

def box(x, y, w, h, t, *, fill=None, stroke=None, rx=6):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill or t["band"]}" stroke="{stroke or t["ref"]}" stroke-width="1"/>')


def badge(cx, cy, n, t, color=None):
    """A numbered circle, like the step markers in a flow chart."""
    return (f'<circle cx="{cx}" cy="{cy}" r="9" fill="{color or t["series"]}"/>'
            + txt(cx, cy + 3.5, str(n), "#ffffff", 10, anchor="middle", weight=700))


def arrow(x1, y1, x2, y2, t, color=None, dashed=False, width=1.5):
    c = color or t["ink"]
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
            f'stroke-width="{width}"{dash} marker-end="url(#head-{t["_name"]})"/>')


def marker(t):
    return (f'<defs><marker id="head-{t["_name"]}" markerWidth="8" markerHeight="8" '
            f'refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L8,4 L0,8 z" fill="{t["ink"]}"/></marker></defs>')


def lines(x, y, rows, t, size=10.5, gap=14, anchor="middle", fill=None):
    return "".join(txt(x, y + i * gap, s, fill or t["ink2"], size, anchor=anchor)
                   for i, s in enumerate(rows))


def step_box(x, y, w, h, n, title, body, t, accent=None):
    """A flow-chart step: badge, bold title, a few lines of plain text."""
    return (box(x, y, w, h, t)
            + badge(x + 16, y + 17, n, t, accent)
            + txt(x + 30, y + 21, title, t["ink"], 12, weight=600)
            + lines(x + 12, y + 42, body, t, anchor="start"))


# --- diagram 1: why it exists ----------------------------------------------

def figure_why(theme):
    t = dict(THEMES[theme], _name=theme)
    w, h = 880, 400
    b = [f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>', marker(t),
         txt(14, 26, "Why tinyserve exists", t["ink"], 13, weight=600),
         txt(14, 42, "A 1.5B model has to read about 1 GB of weights to make one token. "
                     "That read is the slow part, not the math.", t["ink2"], 10.5)]

    # -- row A: the problem
    y = 70
    b.append(txt(14, y + 8, "THE PROBLEM", t["bar2"], 11, weight=600))
    b.append(txt(110, y + 8, "a plain server does one request at a time", t["ink2"], 10.5))
    # queue of eight users
    for i in range(8):
        x = 14 + i * 58
        b.append(box(x, y + 24, 50, 30, t, fill=t["bar2"], stroke=t["bar2"]))
        b.append(txt(x + 25, y + 43, f"user {i + 1}", "#ffffff", 10.5, anchor="middle", weight=600))
    b.append(arrow(486, y + 39, 522, y + 39, t))
    b.append(box(530, y + 16, 116, 46, t))
    b.append(lines(588, y + 35, ["the laptop", "serves one user"], t, anchor="middle"))
    b.append(arrow(654, y + 39, 690, y + 39, t))
    b.append(txt(698, y + 35, "user 1 gets an answer right away.", t["ink"], 10.5))
    b.append(txt(698, y + 49, "user 8 waits 28 s to see a word.", t["ink"], 10.5, weight=600))
    b.append(txt(14, y + 78, "Everyone stands in line. Each one waits for all the users in front of them.",
                 t["ink2"], 10.5))

    b.append(f'<line x1="14" y1="{y + 96}" x2="{w - 14}" y2="{y + 96}" stroke="{t["grid"]}"/>')

    # -- row B: the fix
    y = 186
    b.append(txt(14, y + 8, "THE FIX", t["series"], 11, weight=600))
    b.append(txt(80, y + 8, "a serving engine puts everyone into one batch and runs them together",
                 t["ink2"], 10.5))
    for i in range(8):
        col, row = divmod(i, 4)
        x, yy = 14 + col * 56, y + 24 + row * 24
        b.append(box(x, yy, 50, 20, t, fill=t["series"], stroke=t["series"], rx=3))
        b.append(txt(x + 25, yy + 14, f"user {i + 1}", "#ffffff", 10, anchor="middle", weight=600))
    b.append(txt(70, y + 134, "one batch of 8", t["ink2"], 10.5, anchor="middle", weight=600))
    b.append(arrow(134, y + 70, 172, y + 70, t))
    b.append(box(180, y + 44, 130, 52, t))
    b.append(lines(245, y + 66, ["the laptop serves", "all 8 in one pass"], t, anchor="middle"))
    b.append(arrow(318, y + 70, 356, y + 70, t))
    b.append(txt(366, y + 58, "everyone sees a first word in 1.2 s, not 15 s.", t["ink"], 10.5, weight=600))
    b.append(txt(366, y + 74, "each user gets about 5 tokens a second.", t["ink"], 10.5))
    b.append(txt(366, y + 90, "new users join the batch mid-flight instead of waiting behind it.",
                 t["ink2"], 10.5))
    b.append(txt(366, y + 104, "that is continuous batching. The paged KV cache is what lets",
                 t["ink2"], 10.5))
    b.append(txt(366, y + 118, "8 users' memory fit in the 8 GB next to the model.", t["ink2"], 10.5))

    # -- the question
    y = 340
    b.append(box(14, y - 8, w - 28, 50, t, fill=t["surface"], stroke=t["series"]))
    b.append(txt(w / 2, y + 12, "The question this project was built to answer:",
                 t["ink2"], 10.5, anchor="middle"))
    b.append(txt(w / 2, y + 30, "Can one 8 GB laptop serve eight people almost as fast as it serves one?",
                 t["ink"], 12.5, anchor="middle", weight=600))
    return svg(w, h, "".join(b))


# --- diagram 2: what it is ---------------------------------------------------

def figure_what(theme):
    t = dict(THEMES[theme], _name=theme)
    w, h = 880, 420
    b = [f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>', marker(t),
         txt(14, 26, "What tinyserve is: the path of one request", t["ink"], 13, weight=600),
         txt(14, 42, "An OpenAI-compatible server around a small model on the M1 GPU. "
                     "Every box is one file in the repo.", t["ink2"], 10.5)]

    bw, bh, gap, y = 150, 104, 22, 70
    xs = [24 + i * (bw + gap) for i in range(5)]
    steps = [
        ("Client", ["curl, or any", "OpenAI client", "POST /v1/completions"]),
        ("HTTP server", ["server/app.py", "checks the request,", "turns text into tokens"]),
        ("Engine", ["engine/engine.py", "one thread owns the GPU,", "one queue per request"]),
        ("Scheduler", ["engine/scheduler.py", "continuous batching:", "admit, prefill,", "decode, evict"]),
        ("Runner", ["engine/runner.py", "MLX on the M1 GPU", "Qwen2.5-1.5B, 4-bit"]),
    ]
    for i, (x, (title, body)) in enumerate(zip(xs, steps)):
        b.append(step_box(x, y, bw, bh, i + 1, title, body, t))
        if i < 4:
            b.append(arrow(x + bw + 3, y + bh / 2, x + bw + gap - 3, y + bh / 2, t))

    # the block manager hangs off the scheduler
    x4 = xs[3]
    by = y + bh + 44
    b.append(arrow(x4 + bw / 2, y + bh + 2, x4 + bw / 2, by - 3, t))
    b.append(arrow(x4 + bw / 2 + 10, by - 2, x4 + bw / 2 + 10, y + bh + 3, t, color=t["ink2"], dashed=True))
    b.append(step_box(x4 - 10, by, bw + 20, 112, 6, "Block manager",
                      ["engine/block_manager.py", "KV cache in 16-token blocks:", "a free list, one block table",
                       "per user, and shared blocks", "for shared prompts"], t))
    b.append(txt(x4 + bw / 2 + 18, y + bh + 26, "asks for blocks,", t["ink2"], 9.5))
    b.append(txt(x4 + bw / 2 + 18, y + bh + 38, "gives them back", t["ink2"], 9.5))

    # the return path: tokens stream back
    ry = by + 112 + 30
    x5c, x1c = xs[4] + bw / 2, xs[0] + bw / 2
    b.append(f'<path d="M{x5c},{y + bh + 2} L{x5c},{ry} L{x1c},{ry} L{x1c},{y + bh + 8}" '
             f'fill="none" stroke="{t["series"]}" stroke-width="1.5" '
             f'marker-end="url(#head-{theme})"/>')
    b.append(txt(x5c - 8, ry + 22, "each new token goes straight back to the waiting client, "
                                    "one at a time, as a stream (server/sse.py)",
                 t["series"], 10.5, anchor="end", weight=600))
    b.append(txt(x5c - 8, ry + 37, "the client sees words appear while the rest is still being made",
                 t["ink2"], 10.5, anchor="end"))

    # the benchmark harness, feeding the front door
    hx, hy = xs[1], by
    b.append(step_box(hx, hy, bw + 60, 96, 7, "Benchmark",
                      ["bench/harness.py", "fires 8 users at once and", "records first-token time,",
                       "tok/s and peak RAM"], t, accent=t["bar2"]))
    b.append(arrow(hx + 40, hy - 3, hx + 40, y + bh + 3, t, color=t["bar2"]))
    b.append(txt(hx + 48, hy - 16, "acts as 8 clients", t["bar2"], 9.5))
    return svg(w, h, "".join(b))


# --- diagram 3: one scheduler step -------------------------------------------

def figure_step(theme):
    t = dict(THEMES[theme], _name=theme)
    w, h = 880, 340
    b = [f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>', marker(t),
         txt(14, 26, "What happens in one scheduler step", t["ink"], 13, weight=600),
         txt(14, 42, "The engine runs this loop again and again. Each loop gives every running user "
                     "one new token.", t["ink2"], 10.5)]

    bw, bh, gap, y = 250, 118, 40, 66
    xs = [24 + i * (bw + gap) for i in range(3)]
    steps = [
        ("Admit", ["Take waiting requests while there is", "room in the batch and free KV blocks.",
                   "Run the whole prompt in one pass", "(prefill) and send the first token.",
                   "New users join mid-flight. Nobody", "waits for the batch to finish."]),
        ("Decode", ["One forward pass for all running", "users together. Each one gets one",
                    "token. The 1 GB weight read is", "paid once and shared by all of them.",
                    "This is where batching pays off."]),
        ("Evict", ["Users who hit a stop token or their", "max_tokens are done. Their KV blocks",
                   "go back to the free list, so the next", "step has room to admit more."]),
    ]
    for i, (x, (title, body)) in enumerate(zip(xs, steps)):
        b.append(step_box(x, y, bw, bh, i + 1, title, body, t))
        if i < 2:
            b.append(arrow(x + bw + 4, y + bh / 2, x + bw + gap - 4, y + bh / 2, t))
    # loop back
    ly = y + bh + 18
    b.append(f'<path d="M{xs[2] + bw / 2},{y + bh + 2} L{xs[2] + bw / 2},{ly} L{xs[0] + bw / 2},{ly} '
             f'L{xs[0] + bw / 2},{y + bh + 8}" fill="none" stroke="{t["ink2"]}" stroke-width="1.5" '
             f'stroke-dasharray="4 3" marker-end="url(#head-{theme})"/>')
    b.append(txt(w / 2, ly - 6, "next step", t["ink2"], 10, anchor="middle"))

    # the KV cache, as blocks
    ky = ly + 34
    b.append(txt(14, ky, "THE KV CACHE, AS BLOCKS", t["ink2"], 11, weight=600))
    b.append(txt(190, ky, "every user's memory is a list of 16-token blocks, handed out from one pool",
                 t["ink2"], 10.5))
    cell, cg = 40, 6
    owners = ["shared", "shared", "A", "A", "A", "B", "B", "C", "C", "C", "C",
              "free", "free", "free", "free", "free", "free", "free"]
    colors = {"shared": t["series"], "A": t["bar2"], "B": "#3d9a6b", "C": "#8a63d2", "free": t["band"]}
    for i, o in enumerate(owners):
        x = 14 + i * (cell + cg)
        fill = colors[o]
        stroke = t["ref"] if o == "free" else fill
        b.append(box(x, ky + 14, cell, 28, t, fill=fill, stroke=stroke, rx=4))
        label = {"shared": "sys", "free": ""}.get(o, o)
        b.append(txt(x + cell / 2, ky + 32, label, "#ffffff" if o != "free" else t["ink2"],
                     10, anchor="middle", weight=600))
    lg = ky + 62
    items = [("shared", "shared system prompt: one copy, 3 users point at it"),
             ("A", "user A"), ("B", "user B"), ("C", "user C"), ("free", "free, ready for the next user")]
    x = 14
    for key, label in items:
        fill = colors[key]
        b.append(box(x, lg - 9, 12, 12, t, fill=fill, stroke=t["ref"] if key == "free" else fill, rx=2))
        b.append(txt(x + 18, lg + 1, label, t["ink2"], 10))
        x += 18 + len(label) * 5.6 + 22
    b.append(txt(14, lg + 24, "No big fixed slab per user, so memory is not wasted on space a short "
                              "answer never uses. That is what \"paged\" means here.", t["ink2"], 10.5))
    return svg(w, h, "".join(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"why-{theme}.svg").write_text(figure_why(theme))
        (OUT / f"what-it-is-{theme}.svg").write_text(figure_what(theme))
        (OUT / f"one-step-{theme}.svg").write_text(figure_step(theme))
    print(f"wrote 6 files to {OUT}")


if __name__ == "__main__":
    main()
