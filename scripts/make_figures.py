"""Render the README figures from the M9 and M8 measurements.

Two figures, each emitted twice (light and dark) so the README can serve the
right one to the reader's GitHub theme via <picture>. They live in assets/
because docs/ is gitignored. No dependencies: the
SVG is written directly, from the numbers recorded in MEASUREMENTS.md.

    python scripts/make_figures.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets"

# --- the data, straight out of MEASUREMENTS.md -----------------------------

# M9: the isolated decode step, padded backend, 256 tokens of context per row
BATCH = [1, 2, 4, 8, 10, 12, 14, 16, 24, 32]
MS_STEP = [34.9, 61.9, 111.6, 191.1, 193.7, 206.9, 196.1, 200.5, 209.9, 211.7]
TOKS = [28.6, 32.3, 35.8, 41.9, 51.6, 58.0, 71.4, 79.8, 114.3, 151.1]
KNEE = 8

# M8: eight users, serial engine vs the finished engine
TTFT = [("median", 14.9, 1.2), ("p95", 28.2, 1.8)]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e7e6e2",
                  series="#2a78d6", band="#f2f1ed", ref="#9a9992", bar2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#2f2f2c",
                 series="#3987e5", band="#242422", ref="#7d7c75", bar2="#d95926"),
}

W, H = 880, 330
PAD_L, PAD_R, PAD_T, PAD_B = 52, 46, 52, 46
PANEL_W = (W - 26) / 2


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, fill, size=11, anchor="start", weight=400, opacity=1.0):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" opacity="{opacity}" '
            f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"'
            f'>{esc(s)}</text>')


def log2(v):
    return (v.bit_length() - 1) if isinstance(v, int) and v and not (v & (v - 1)) else __import__("math").log2(v)


def panel(ox, title, subtitle, ys, ymax, unit, t, label_idx, note):
    """One line panel: batch size (log2 x) against ys."""
    x0, x1 = ox + PAD_L, ox + PANEL_W - PAD_R
    y0, y1 = PAD_T + 18, H - PAD_B
    lo, hi = log2(1), log2(32)

    def px(b):
        return x0 + (log2(b) - lo) / (hi - lo) * (x1 - x0)

    def py(v):
        return y1 - (v / ymax) * (y1 - y0)

    p = [txt(ox + 14, 26, title, t["ink"], 13, weight=600),
         txt(ox + 14, 42, subtitle, t["ink2"], 10.5)]

    # the regime band: everything at or below the knee
    p.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{px(KNEE)-x0:.1f}" '
             f'height="{y1-y0:.1f}" fill="{t["band"]}"/>')

    # gridlines + y labels
    for i in range(5):
        v = ymax * i / 4
        y = py(v)
        p.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        p.append(txt(x0 - 8, y + 3.5, f"{v:g}", t["ink2"], 10, anchor="end"))
    p.append(txt(x0 - 8, y0 - 12, unit, t["ink2"], 10, anchor="end"))

    # x ticks at the powers of two
    for b in (1, 2, 4, 8, 16, 32):
        p.append(txt(px(b), y1 + 18, str(b), t["ink2"], 10, anchor="middle"))
    p.append(txt((x0 + x1) / 2, y1 + 36, "batch size (rows in one decode step)",
                 t["ink2"], 10, anchor="middle"))

    # the knee marker
    p.append(f'<line x1="{px(KNEE):.1f}" y1="{y0:.1f}" x2="{px(KNEE):.1f}" '
             f'y2="{y1:.1f}" stroke="{t["ref"]}" stroke-width="1" stroke-dasharray="3 3"/>')

    pts = " ".join(f"{px(b):.1f},{py(v):.1f}" for b, v in zip(BATCH, ys))
    p.append(f'<polyline points="{pts}" fill="none" stroke="{t["series"]}" '
             f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for b, v in zip(BATCH, ys):
        p.append(f'<circle cx="{px(b):.1f}" cy="{py(v):.1f}" r="4" '
                 f'fill="{t["series"]}" stroke="{t["surface"]}" stroke-width="2"/>')

    # selective direct labels only, anchored so they never leave the panel
    # or land on the y-axis numbers
    for b, v in zip(BATCH, ys):
        if b in label_idx:
            up, anchor = label_idx[b]
            dx = {"start": 9, "end": -9, "middle": 0}[anchor]
            p.append(txt(px(b) + dx, py(v) + (-10 if up else 17), f"{v:g}", t["ink"],
                         10.5, anchor=anchor, weight=600))

    p.append(txt(x0 + 7, y0 + 14, note[0], t["ink2"], 9.5, weight=600))
    p.append(txt(px(KNEE) + 7, y0 + 14, note[1], t["ink2"], 9.5, weight=600))
    return "".join(p)


def figure_batching(theme):
    t = THEMES[theme]
    body = [f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']
    body.append(panel(0, "A decode step costs almost the same for 8 rows as for 32",
                      "milliseconds per step — lower is better",
                      MS_STEP, 240, "ms", t,
                      {1: (True, "start"), 8: (True, "middle"), 32: (True, "end")},
                      ("cost grows with the batch", "cost is flat")))
    body.append(panel(PANEL_W + 30, "…so past batch 8, throughput scales for free",
                      "aggregate tokens/sec across all rows — higher is better",
                      TOKS, 160, "tok/s", t,
                      {1: (True, "start"), 8: (False, "middle"), 32: (True, "end")},
                      ("batching buys nothing", "extra users ~free")))
    return svg(W, H, "".join(body))


def figure_ttft(theme):
    t = THEMES[theme]
    w, h = 880, 210
    x0, x1 = 150, w - 90
    body = [f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>',
            txt(14, 26, "Eight users, time to first token", t["ink"], 13, weight=600),
            txt(14, 42, "serial engine (one request at a time) vs continuous batching + paged KV",
                t["ink2"], 10.5)]
    vmax = 30.0
    y = 68
    for label, before, after in TTFT:
        for name, val, color in (("serial", before, t["bar2"]), ("tinyserve", after, t["series"])):
            bw = max(3.0, (val / vmax) * (x1 - x0))
            body.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="22" rx="4" fill="{color}"/>')
            body.append(txt(x0 - 10, y + 15.5, f"{label} · {name}", t["ink2"], 10.5, anchor="end"))
            body.append(txt(x0 + bw + 8, y + 15.5, f"{val:g} s", t["ink"], 11, weight=600))
            y += 28
        y += 10
    body.append(txt(x0, h - 16, "12× faster at the median, 15× at p95 — "
                                "aggregate throughput rose only 1.41×",
                    t["ink2"], 10.5))
    return svg(w, h, "".join(body))


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">{body}</svg>\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"batching-regimes-{theme}.svg").write_text(figure_batching(theme))
        (OUT / f"ttft-{theme}.svg").write_text(figure_ttft(theme))
    print(f"wrote 4 files to {OUT}")


if __name__ == "__main__":
    main()
