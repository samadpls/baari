"""Write assets/hero.svg, the animated README banner.

The text is drawn as outlines because GitHub shows README images through a proxy that loads no web
fonts. Fonts (from github.com/google/fonts) go in data/fonts: Bricolage.ttf, Geist.ttf and
NotoNastaliqUrdu.ttf.

    uv run --with uharfbuzz --with fonttools python assets/make_hero.py
"""

import random
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONTS = Path("data/fonts")
INK, MUTED, GREEN, AMBER, AMBER_TEXT, CREAM, RULE = (
    "#17150F",
    "#4A463D",
    "#0E6B4F",
    "#F2B544",
    "#9A5F0C",
    "#F7F1E3",
    "#C9BFA9",
)
W, H, LOOP = 1280, 400, 10
TALK = [(0, 25), (40, 65)]  # percent of the loop
BARS = [
    (20, 32),
    (36, 64),
    (52, 44),
    (68, 20),
]  # x and height in the 120-unit logo; the logo is the finished state


def num(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


class Font:
    def __init__(self, file, **axes):
        face = hb.Face(hb.Blob.from_file_path(FONTS / file))
        self.font, self.unit = hb.Font(face), 1 / face.upem
        self.font.set_variations(axes)

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {})
        return zip(buf.glyph_infos, buf.glyph_positions)

    def width(self, text, size, tracking=0):
        return sum(p.x_advance * size * self.unit + tracking for _, p in self._shape(text))

    def path(self, text, x, y, size, tracking=0):
        s = size * self.unit
        pen = SVGPathPen(None, ntos=num)
        for info, p in self._shape(text):
            self.font.draw_glyph_with_pen(
                info.codepoint, TransformPen(pen, (s, 0, 0, -s, x + p.x_offset * s, y - p.y_offset * s))
            )
            x += p.x_advance * s + tracking
        return pen.getCommands()


def talking(t):
    return any(a <= t < b for a, b in TALK)


def bar_keyframes(i, base):
    rng = random.Random(i)
    frames = []
    for k in range(41):
        t = k * 2.5
        if t in (0, 100) or t >= 70:
            h = base
        elif talking(t):
            h = rng.uniform(14, 84)
        else:
            h = 10
        frames.append(f"{num(t)}%{{transform:scaleY({h / base:.3f})}}")
    return f"@keyframes bar{i}{{{''.join(frames)}}}"


def fade(name, windows, base):
    """Opacity 1 inside each (start, end) percent window, with 1% fades."""
    stops = {0.0: 0.0, 100.0: 0.0}
    for a, b in windows:
        stops |= {max(a - 1, 0): 0.0, a: 1.0, b - 1: 1.0, b: 0.0}
    if windows[0][0] == 0:
        stops[0.0] = 1.0
    body = "".join(f"{num(t)}%{{opacity:{o:g}}}" for t, o in sorted(stops.items()))
    return f"@keyframes {name}{{{body}}}.{name}{{opacity:{base};animation:{name} {LOOP}s linear infinite}}"


def main():
    title = Font("Bricolage.ttf", wght=800, opsz=96)
    urdu = Font("NotoNastaliqUrdu.ttf", wght=600)
    text = Font("Geist.ttf", wght=400)
    bold = Font("Geist.ttf", wght=600)

    x0 = 432
    title_w = title.width("baari", 112, -3)
    parts = [
        f'<path fill="{INK}" d="{title.path("baari", x0, 158, 112, -3)}"/>',
        f'<path fill="{GREEN}" d="{urdu.path("باری", x0 + title_w + 30, 148, 64)}"/>',
        f'<path fill="{MUTED}" d="{text.path("Knows when an Urdu speaker has finished talking", x0, 216, 30)}"/>',
    ]
    status = [
        ("talk", MUTED, "Speaking", TALK, 0),
        ("wait", GREEN, "Pause, mid-sentence: keep listening", [(26, 39)], 0),
        ("done", AMBER_TEXT, "Pause, finished: reply now", [(66, 98)], 1),
    ]
    css = []
    for name, color, words, windows, base in status:
        parts.append(
            f'<g class="{name}"><circle cx="{x0 + 8}" cy="271" r="7" fill="{color}"/>'
            f'<path fill="{color}" d="{bold.path(words, x0 + 26, 281, 28)}"/></g>'
        )
        css.append(fade(name, windows, base))

    links = ["github.com/samadpls", "samadpls.com", "Open source, BSD-2-Clause"]
    x = x0
    for k, link in enumerate(links):
        if k:
            parts.append(f'<circle cx="{num(x + 14)}" cy="327" r="2.5" fill="{RULE}"/>')
            x += 28
        parts.append(f'<path fill="{GREEN}" d="{bold.path(link, x, 334, 21)}"/>')
        x += bold.width(link, 21)
    assert x < W - 60 and x0 + title_w + 30 + urdu.width("باری", 64) < W - 60

    bars = "".join(
        f'<rect class="b{i}" x="{bx}" y="{60 - h / 2}" width="10" height="{h}" rx="5"/>'
        for i, (bx, h) in enumerate(BARS)
    )
    css += [
        bar_keyframes(i, h)
        + f".b{i}{{transform-box:fill-box;transform-origin:center;animation:bar{i} {LOOP}s ease-in-out infinite}}"
        for i, (_, h) in enumerate(BARS)
    ]
    css.append(
        "@keyframes listen{0%,25%{opacity:.25}29%{opacity:.85}33%{opacity:.3}37%{opacity:.85}40%,65%{opacity:.25}66%,100%{opacity:0}}"
        f".listen{{opacity:0;animation:listen {LOOP}s linear infinite}}"
    )
    css.append(
        "@keyframes stop{0%,65%{transform:scale(0)}69%{transform:scale(1.35)}72%,97%{transform:scale(1)}100%{transform:scale(0)}}"
        f".stop{{transform-box:fill-box;transform-origin:center;animation:stop {LOOP}s ease-out infinite}}"
    )
    css.append("@media (prefers-reduced-motion:reduce){*{animation:none!important}}")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="baari: knows when an Urdu speaker has finished talking">
<title>baari: knows when an Urdu speaker has finished talking</title>
<defs><radialGradient id="glow" cx="235" cy="200" r="430" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#E4EFE6"/><stop offset="1" stop-color="#E4EFE6" stop-opacity="0"/></radialGradient></defs>
<style>{"".join(css)}</style>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="24" fill="#FBF7EF" stroke="#ECE4D3"/>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="24" fill="url(#glow)"/>
<g transform="translate(140 105) scale({190 / 120:.4f})">
<rect x="2" y="2" width="116" height="116" rx="30" fill="{GREEN}"/>
<g fill="{CREAM}">{bars}</g>
<circle class="listen" cx="92" cy="60" r="8" fill="{CREAM}"/>
<circle class="stop" cx="92" cy="60" r="8" fill="{AMBER}"/>
</g>
{"".join(parts)}
</svg>
"""
    out = Path("assets/hero.svg")
    out.write_text(svg)
    print(out, f"{len(svg) // 1024} KB")


if __name__ == "__main__":
    main()
