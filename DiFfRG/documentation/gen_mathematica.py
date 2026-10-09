#!/usr/bin/env python3
"""Generate MyST-Markdown reference pages for the DiFfRG Mathematica package.

The pages are produced purely by scanning the package ``.m`` sources for
``Symbol::usage = "..."`` definitions -- no Wolfram kernel is invoked.  This is
the auto-generated replacement for the previously hand-written notebook docs.

Usage::

    python gen_mathematica.py --src ../Mathematica/DiFfRG --out _generated_mathematica

Each source file becomes one reference page; an ``index.md`` ties them together.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- configuration ----------------------------------------------------------

# A ``Symbol::usage = "..."`` definition. The body allows escaped characters
# (``\"`` for literal quotes, ``\[Name]`` for Wolfram named characters) and may
# span multiple lines, so DOTALL is used. Matching ``::usage`` specifically
# excludes message tags such as ``MakeKernel::MissingKey``.
USAGE_RE = re.compile(
    r'(?P<sym>[A-Za-z$][A-Za-z0-9$]*)::usage\s*=\s*"(?P<body>(?:[^"\\]|\\.)*)"',
    re.DOTALL,
)

# A small map of the Wolfram named characters that actually appear in the usage
# strings, plus a few common neighbours. Unknown ``\[Name]`` fall back to their
# bare name (see ``unescape``).
NAMED_CHARS = {
    "Pi": "π",
    "Element": "∈",
    "CapitalZeta": "ℤ",  # the source uses CapitalZeta to mean the integers
    "Integers": "ℤ",
    "Sigma": "σ",
    "CapitalSigma": "Σ",
    "Pi": "π",
    "CapitalPi": "Π",
    "Eta": "η",
    "Rho": "ρ",
    "Phi": "φ",
    "CapitalPhi": "Φ",
    "Lambda": "λ",
    "CapitalLambda": "Λ",
    "Mu": "μ",
    "Nu": "ν",
    "Alpha": "α",
    "Beta": "β",
    "Gamma": "γ",
    "CapitalGamma": "Γ",
    "Delta": "δ",
    "CapitalDelta": "Δ",
    "Theta": "θ",
    "Tau": "τ",
    "Omega": "ω",
    "CapitalOmega": "Ω",
    "Infinity": "∞",
    "Rule": "→",
    "RuleDelayed": "⧴",
    "Equal": "=",
    "Integral": "∫",
    "PartialD": "∂",
    "Sqrt": "√",
    "Dagger": "†",
    "Transpose": "ᵀ",
    "Element": "∈",
    "LineSeparator": "\n",
    "IndentingNewLine": "\n",
}

NAMED_CHAR_RE = re.compile(r"\\\[([A-Za-z][A-Za-z0-9]*)\]")

# Human-friendly titles per source file (relative to the package root). Files
# not listed fall back to a title derived from the path.
FILE_TITLES = {
    "DiFfRG.m": "DiFfRG (main package)",
    "CodeTools.m": "CodeTools",
    "CodeTools/Directory.m": "CodeTools · Directory",
    "CodeTools/Export.m": "CodeTools · Export",
    "CodeTools/MakeKernel.m": "CodeTools · MakeKernel",
    "CodeTools/Regulator.m": "CodeTools · Regulator",
    "CodeTools/TemplateParameterGeneration.m": "CodeTools · TemplateParameterGeneration",
    "CodeTools/Utils.m": "CodeTools · Utils",
}


# --- extraction --------------------------------------------------------------


def unescape(body: str) -> str:
    """Turn a raw Wolfram usage-string body into readable Unicode text."""
    body = NAMED_CHAR_RE.sub(lambda m: NAMED_CHARS.get(m.group(1), m.group(1)), body)
    body = body.replace('\\"', '"')
    return body.strip()


class Symbol:
    def __init__(self, name: str, signature: str, description: str):
        self.name = name
        self.signature = signature
        self.description = description


def extract_file(path: Path) -> list[Symbol]:
    """Return the (deduplicated) symbols documented in a single ``.m`` file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    symbols: list[Symbol] = []
    seen: set[str] = set()
    for m in USAGE_RE.finditer(text):
        name = m.group("sym")
        body = unescape(m.group("body"))
        if not body:
            # Placeholder usages such as ``Foo::usage = ""`` carry no docs.
            continue
        if name in seen:
            continue
        seen.add(name)
        lines = body.split("\n", 1)
        signature = lines[0].strip()
        description = lines[1].strip() if len(lines) > 1 else ""
        symbols.append(Symbol(name, signature, description))
    return symbols


# --- rendering ---------------------------------------------------------------


def slug(rel: str) -> str:
    return rel.replace("/", "_").replace(".m", "").lower()


def title_for(rel: str) -> str:
    return FILE_TITLES.get(rel, rel.replace(".m", "").replace("/", " · "))


def render_page(title: str, symbols: list[Symbol]) -> str:
    out = [f"# {title}", ""]
    out.append(f"*{len(symbols)} symbol{'s' if len(symbols) != 1 else ''}, "
               "auto-generated from package `::usage` strings.*")
    out.append("")
    for s in symbols:
        out.append(f"## `{s.name}`")
        out.append("")
        out.append("```mathematica")
        out.append(s.signature)
        out.append("```")
        out.append("")
        if s.description:
            out.append(s.description)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_index(entries: list[tuple[str, str, int]]) -> str:
    """``entries`` is a list of (page-filename, title, symbol-count)."""
    total = sum(n for _, _, n in entries)
    out = [
        "# Mathematica API",
        "",
        "The DiFfRG Mathematica package provides symbolic derivation of fRG flow "
        "equations and generation of C++ integration kernels.",
        "",
        f"The reference below is auto-generated from the `::usage` strings in the "
        f"package sources ({total} symbols across {len(entries)} modules).",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
    ]
    for fname, title, n in entries:
        out.append(fname.replace(".md", ""))
    out += ["```", ""]
    out.append("| Module | Symbols |")
    out.append("| --- | --- |")
    for fname, title, n in entries:
        out.append(f"| [{title}]({fname}) | {n} |")
    out.append("")
    return "\n".join(out)


# --- driver ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default=str(Path(__file__).parent / ".." / "Mathematica" / "DiFfRG"),
        help="Root of the Mathematica package (contains DiFfRG.m).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "_generated_mathematica"),
        help="Output directory for generated MyST pages.",
    )
    args = parser.parse_args(argv)

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if not src.is_dir():
        print(f"error: source directory not found: {src}", file=sys.stderr)
        return 1

    m_files = sorted(
        p for p in src.rglob("*.m") if "Tests" not in p.relative_to(src).parts
    )
    if not m_files:
        print(f"error: no .m files under {src}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str, int]] = []
    total_symbols = 0
    for path in m_files:
        rel = path.relative_to(src).as_posix()
        symbols = extract_file(path)
        if not symbols:
            continue
        title = title_for(rel)
        fname = slug(rel) + ".md"
        (out / fname).write_text(render_page(title, symbols), encoding="utf-8")
        entries.append((fname, title, len(symbols)))
        total_symbols += len(symbols)

    # Order the main package first, then the rest alphabetically by title.
    entries.sort(key=lambda e: (not e[0].startswith("diffrg."), e[1]))
    (out / "index.md").write_text(render_index(entries), encoding="utf-8")

    print(
        f"gen_mathematica: wrote {len(entries)} pages, "
        f"{total_symbols} symbols to {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
