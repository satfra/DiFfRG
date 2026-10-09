#!/usr/bin/env python3
"""Make Doxygen's generated CSS digestible by doxysphinx's libsass scoper.

Doxygen >= 1.13 emits CSS relative-color functions such as
``hsl(from #f8d1cc h s calc(l + -3))``. These are valid CSS but not valid SCSS,
so the (old) libsass that doxysphinx uses to scope the stylesheet fails with
"Function hsl is missing argument $saturation".

We rewrite every ``<func>(from <color> ...)`` expression to its base ``<color>``
in-place. The small lightness/saturation adjustments are dropped, which is
cosmetically negligible for the embedded API pages. This is a no-op on older
Doxygen output that does not use the relative-color syntax.

Usage: fix_doxygen_css.py <doxygen-html-dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_FUNCS = ("hsl", "hsla", "rgb", "rgba", "hwb", "lab", "lch", "oklab", "oklch", "color")
_OPEN = re.compile(r"\b(?:" + "|".join(_FUNCS) + r")\(\s*from\s+", re.IGNORECASE)


def _strip_relative_colors(css: str) -> tuple[str, int]:
    out: list[str] = []
    i = 0
    count = 0
    while True:
        m = _OPEN.search(css, i)
        if not m:
            out.append(css[i:])
            break
        out.append(css[i : m.start()])
        # Walk from the function's '(' to its matching ')', honouring nesting
        # (the channel list often contains calc(...)).
        open_idx = css.index("(", m.start())
        depth = 0
        j = open_idx
        while j < len(css):
            if css[j] == "(":
                depth += 1
            elif css[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        inner = css[open_idx + 1 : j]
        toks = inner.split(None, 2)  # ["from", "<color>", "<channels...>"]
        base = toks[1] if len(toks) > 1 else "inherit"
        out.append(base)
        count += 1
        i = j + 1
    return "".join(out), count


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    html_dir = Path(argv[1])
    if not html_dir.is_dir():
        print(f"error: not a directory: {html_dir}", file=sys.stderr)
        return 1
    total = 0
    for css in html_dir.glob("*.css"):
        text = css.read_text(encoding="utf-8", errors="replace")
        fixed, n = _strip_relative_colors(text)
        if n:
            css.write_text(fixed, encoding="utf-8")
            total += n
    print(f"fix_doxygen_css: rewrote {total} relative-color expression(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
